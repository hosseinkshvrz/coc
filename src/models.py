import os

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, wilcoxon
import statsmodels.formula.api as smf

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_PATH, 'data')


def filter_data(df):
    # filter 1: bots
    df = df.dropna(subset=['gender', 'ethnicity'])
    # filter 2: non-person
    df = df[df['is_person'] == True]
    # filter 4: without CoC
    df = df[~df['adoption_date'].isnull()]
    df = df[df['pull_merged'] == True]
    return df


def prepare_data(df):
    df['pull_creation_date'] = pd.to_datetime(df['pull_creation_date'], format='%Y-%m-%d %H:%M:%S')
    df['adoption_date'] = pd.to_datetime(df['adoption_date'], format='%Y-%m-%d %H:%M:%S')
    df['start_date'] = pd.to_datetime(df['start_date'], format='%Y-%m-%d %H:%M:%S')
    df['end_date'] = pd.to_datetime(df['end_date'], format='%Y-%m-%d %H:%M:%S')

    df = filter_data(df)

    return df


def manual_group_by(df, series):
    counts = []
    for i, d in enumerate(series):
        if i == 0:
            counts.append(len(df[df['pull_creation_date'] < d]))
        else:
            counts.append(len(df[df['pull_creation_date'] < d]) - sum(counts[:i]))
    frame = pd.DataFrame({'pull_creation_date': series,
                          'repository_url': [df['repository_url'].tolist()[0] for i in range(len(counts))],
                          'exc_count': counts})
    return frame


def get_data(pdf, project, distinct, interval):
    pdf = pdf[pdf['repository_url'] == project]
    before = pdf[
        (pdf['pull_creation_date'] > pdf['start_date']) & (pdf['pull_creation_date'] < pdf['adoption_date'])]
    after = pdf[
        (pdf['pull_creation_date'] > pdf['adoption_date']) & (pdf['pull_creation_date'] < pdf['end_date'])]

    if distinct:
        before = before.drop_duplicates('user_login')
        after = after.drop_duplicates('user_login')

    before_inclusive = before.groupby([pd.Grouper(key='pull_creation_date', freq=interval),
                                       pd.Grouper(key='repository_url')]).size().reset_index(name='inc_count')
    before_exclusive = before[~((before['gender'] == 'male') & (before['ethnicity'] == 'white'))]
    before_exclusive = manual_group_by(before_exclusive, before_inclusive['pull_creation_date'])
    before = pd.merge(before_inclusive, before_exclusive, how='outer')
    before['ratio'] = before['exc_count'] / before['inc_count']

    after_inclusive = after.groupby([pd.Grouper(key='pull_creation_date', freq=interval),
                                     pd.Grouper(key='repository_url')]).size().reset_index(name='inc_count')
    after_exclusive = after[~((after['gender'] == 'male') & (after['ethnicity'] == 'white'))]
    after_exclusive = manual_group_by(after_exclusive, after_inclusive['pull_creation_date'])
    after = pd.merge(after_inclusive, after_exclusive, how='outer')
    after['ratio'] = after['exc_count'] / after['inc_count']

    print()

    return before, after


def fit_model(train, test):
    train_num = train.copy(deep=True)
    test_num = test.copy(deep=True)

    train_num['pull_creation_date'] = train_num['pull_creation_date'].values.astype(np.int64) // 10 ** 9
    X = sm.add_constant(train_num['pull_creation_date'])
    model = sm.OLS(train_num['ratio'], X)
    tr_results = model.fit()
    print(tr_results.summary())
    print(len(tr_results.params))

    train_predict = tr_results.params[0] + tr_results.params[1] * train_num['pull_creation_date']

    test_num['pull_creation_date'] = test_num['pull_creation_date'].values.astype(np.int64) // 10 ** 9
    X = sm.add_constant(test_num['pull_creation_date'])
    model = sm.OLS(test_num['ratio'], X)
    te_results = model.fit()
    print(te_results.summary())
    test_predict = te_results.params[0] + te_results.params[1] * test_num['pull_creation_date']

    return train_predict, test_predict, tr_results.params[1], te_results.params[1]


def mixed_effect(df):
    df_copy = df.copy(deep=True)

    # df_copy = df_copy[
    #     (df_copy['pull_creation_date'] > df_copy['start_date']) & (df_copy['pull_creation_date'] < df_copy['end_date'])]
    df_copy['coc_added'] = (df_copy['pull_creation_date'] >= df_copy['adoption_date']).astype(int)
    df_copy['has_coc'] = (df_copy['adoption_date'].isnull() == False).astype(int)

    df_copy = df_copy.drop_duplicates('user_login')
    df_copy = df_copy.groupby([pd.Grouper(key='pull_creation_date', freq='3M'),
                               pd.Grouper(key='repository_url'),
                               pd.Grouper(key='coc_added')]).size().reset_index(name='count')
    cols = ['repository_url', 'project', 'forks', 'stars', 'total_contributors', 'size', 'age']
    train = pd.merge(df_copy, df[cols].drop_duplicates(), how='left')
    md = smf.mixedlm('count ~ coc_added + stars + size + age', train, groups=train['project'])
    mdf = md.fit()
    print(mdf.summary())

    md = smf.mixedlm('count ~ pull_creation_date + coc_added + stars + size + age', train, groups=train['project'])
    mdf = md.fit()
    print(mdf.summary())

    print()


def test_sig(df):
    print('\n\n\n')
    print(df)
    print('\n\n\n')
    stat, p = mannwhitneyu(df['before_slope'], df['after_slope'], alternative='greater')
    print('contributor rate slopes MannWhitneyU Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    # if p > alpha:
    #     print('Same distribution (fail to reject H0)')
    # else:
    #     print('Different distribution (reject H0)')

    # print('\n\n\n')
    stat, p = wilcoxon(df['before_slope'], df['after_slope'], alternative='greater')
    print('contributor rate slopes Wilcoxon Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    # alpha = 0.05
    # if p > alpha:
    #     print('Same distribution (fail to reject H0)')
    # else:
    #     print('Different distribution (reject H0)')


def run(df, distinct=False, interval='3M'):
    df = prepare_data(df)
    # mixed_effect(df)
    projects = df['repository_url'].unique().tolist()
    projects = {projects[i]: i + 1 for i in range(len(projects))}
    df['project'] = [projects[p] for p in df['repository_url']]
    print()
    slopes = pd.DataFrame({'project': [p for p in projects.keys()]})
    tr_slopes, te_slopes = [], []
    for p in projects.keys():
        print(p)
        train, test = get_data(df, p, distinct, interval)
        train_predict, test_predict, tr_slope, te_slope = fit_model(train, test)
        tr_slopes.append(tr_slope)
        te_slopes.append(te_slope)
        plt.figure(figsize=(16, 8))
        plt.scatter(train['pull_creation_date'], train['ratio'], color='cornflowerblue', label='Before CoC Contribution')
        plt.scatter(test['pull_creation_date'], test['ratio'], color='indianred', label='After CoC Contribution')
        plt.plot(train['pull_creation_date'], train_predict, color='cornflowerblue', linewidth=3)
        plt.plot(test['pull_creation_date'], test_predict, color='indianred', linewidth=3)
        plt.xlabel("time")
        plt.ylabel("ratio of non-white non-male contributions to total contributions")
        plt.legend(loc="upper left")
        plt.savefig(data_path + '/figs/{}_{}.png'.format(p.split('/')[1], interval))
        # plt.savefig(data_path + '/figs/{}_noninclusive_distinct_{}.png'.format('all', interval))
        plt.close()
    slopes['before_slope'] = tr_slopes
    slopes['after_slope'] = te_slopes
    test_sig(slopes)


if __name__ == '__main__':
    df = pd.read_csv(data_path + '/All_Pulls.csv')
    run(df)
