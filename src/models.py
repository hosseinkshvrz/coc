import os

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, wilcoxon
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_PATH, 'data')


def filter_data(df):
    # filter 1: bots
    df = df.dropna(subset=['gender', 'ethnicity'])
    # filter 2: non-person
    df = df[df['is_person'] == True]
    return df


def prepare_data(df):
    df['pull_creation_date'] = pd.to_datetime(df['pull_creation_date'], format='%Y-%m-%d %H:%M:%S')
    df['adoption_date'] = pd.to_datetime(df['adoption_date'], format='%Y-%m-%d %H:%M:%S')
    df['start_date'] = pd.to_datetime(df['start_date'], format='%Y-%m-%d %H:%M:%S')
    df['end_date'] = pd.to_datetime(df['end_date'], format='%Y-%m-%d %H:%M:%S')
    df = filter_data(df)

    return df


def add_blau_index(df):
    df_pr = df.groupby('repository_url').size().reset_index(name='count')
    df_wm = df[(df['gender'] == 'male') & (df['ethnicity'] == 'white')]
    df_wm = df_wm.groupby('repository_url').size().reset_index(name='wm_count')
    df_wm = pd.merge(df_wm, df_pr)
    df_wm['wm_ratio'] = df_wm['wm_count'] / df_wm['count']
    df_nwnm = df[~((df['gender'] == 'male') & (df['ethnicity'] == 'white'))]
    df_nwnm = df_nwnm.groupby('repository_url').size().reset_index(name='nwnm_count')
    df_nwnm = pd.merge(df_nwnm, df_pr)
    df_nwnm['nwnm_ratio'] = df_nwnm['nwnm_count'] / df_nwnm['count']
    df_bl = pd.merge(df_wm, df_nwnm)[['repository_url', 'wm_ratio', 'nwnm_ratio']]
    df_bl['blau_index'] = 1 - ((df_bl['wm_ratio'] ** 2) + (df_bl['nwnm_ratio'] ** 2))
    df = pd.merge(df, df_bl[['repository_url', 'blau_index']], how='left')

    return df


def test_blau_with_without(df):
    df = df[(df['pull_creation_date'] > pd.to_datetime('2020-06-08'))]  # last 12 month data
    df['has_coc'] = (df['adoption_date'].isnull() == False).astype(int)
    df = df.drop_duplicates(['repository_url', 'user_login'])
    df = add_blau_index(df)
    cols = ['repository_url', 'project', 'stars', 'size', 'age', 'has_coc', 'blau_index']
    df = df.drop_duplicates(cols)[cols]

    X = df[cols[2:-1]]
    y = df[cols[-1]]

    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    tr_results = model.fit()
    print(tr_results.summary())

    md = smf.mixedlm('blau_index ~  stars + size + age + has_coc', df, groups=df['project'])
    mdf = md.fit()
    print(mdf.summary())


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


def get_before_after_data(df, interval):
    before = df[
        (df['pull_creation_date'] > df['start_date']) & (df['pull_creation_date'] < df['adoption_date'])]
    after = df[
        (df['pull_creation_date'] > df['adoption_date']) & (df['pull_creation_date'] < df['end_date'])]

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

    return before, after


def test_significance(df, column_a, column_b, mode, test):
    stat, p = mannwhitneyu(df[column_a], df[column_b], alternative='less')
    print('{} ratio {} MannWhitneyU Statistics={:.4f}, p={:.4f}'.format(mode, test, stat, p))
    stat, p = wilcoxon(df[column_a], df[column_b], alternative='less')
    print('{} ratio {} Wilcoxon Statistics={:.4f}, p={:.4f}'.format(mode, test, stat, p))


def plot_fig(before_df, after_df, before_prd, after_prd, project, mode, test, interval):
    plt.figure(figsize=(16, 8))
    plt.scatter(before_df['pull_creation_date'], before_df['ratio'], color='cornflowerblue', label='Before CoC')
    plt.scatter(after_df['pull_creation_date'], after_df['ratio'], color='indianred', label='After CoC')
    plt.plot(before_df['pull_creation_date'], before_prd, color='cornflowerblue', linewidth=3)
    plt.plot(after_df['pull_creation_date'], after_prd, color='indianred', linewidth=3)
    plt.xlabel('time')
    plt.ylabel('ratio of non-white non-male {} to total {}'.format(mode, mode))
    plt.legend(loc='upper left')
    plt.savefig(data_path + '/figs/{}_{}_{}_{}.png'.format(project.split('/')[1], mode, test, interval))
    plt.close()


def fit_model(X, y):
    model = sm.OLS(y, X)
    result = model.fit()
    # print(result.summary())

    return result


def test_contributors_slopes(df, projects, interval):
    print('\n\n\t\t***** test contributors slopes before and after CoC. *****\n\n')
    mode = 'contributors'
    test = 'slopes'
    projects = {p: projects[p] for p in df[~df['adoption_date'].isnull()]['repository_url']}
    slopes = pd.DataFrame({'project': [p for p in projects.keys()]})
    b_slopes, a_slopes = [], []
    for p in projects.keys():
        pdf = df[df['repository_url'] == p].drop_duplicates(['user_login'])
        before, after = get_before_after_data(pdf, interval)

        X = before['pull_creation_date'].values.astype(np.int64) // 10 ** 9
        X = sm.add_constant(X)
        y = before['ratio']
        b_result = fit_model(X, y)
        before_prd = b_result.params[0] + b_result.params[1] * X[:, 1]
        b_slopes.append(b_result.params[1])

        X = after['pull_creation_date'].values.astype(np.int64) // 10 ** 9
        X = sm.add_constant(X)
        y = after['ratio']
        a_result = fit_model(X, y)
        after_prd = a_result.params[0] + a_result.params[1] * X[:, 1]
        a_slopes.append(a_result.params[1])

        plot_fig(before, after, before_prd, after_prd, p, mode, test, interval)

    slopes['before_slope'] = b_slopes
    slopes['after_slope'] = a_slopes
    test_significance(slopes, 'before_slope', 'after_slope', mode, test)


def test_contributions_slopes(df, projects, interval):
    print('\n\n\t\t***** test contributions slopes before and after CoC. *****\n\n')
    mode = 'contributions'
    test = 'slopes'
    projects = {p: projects[p] for p in df[~df['adoption_date'].isnull()]['repository_url']}
    slopes = pd.DataFrame({'project': [p for p in projects.keys()]})
    b_slopes, a_slopes = [], []
    for p in projects.keys():
        pdf = df[df['repository_url'] == p]
        before, after = get_before_after_data(pdf, interval)

        X = before['pull_creation_date'].values.astype(np.int64) // 10 ** 9
        X = sm.add_constant(X)
        y = before['ratio']
        b_result = fit_model(X, y)
        before_prd = b_result.params[0] + b_result.params[1] * X[:, 1]
        b_slopes.append(b_result.params[1])

        X = after['pull_creation_date'].values.astype(np.int64) // 10 ** 9
        X = sm.add_constant(X)
        y = after['ratio']
        a_result = fit_model(X, y)
        after_prd = a_result.params[0] + a_result.params[1] * X[:, 1]
        a_slopes.append(a_result.params[1])

        plot_fig(before, after, before_prd, after_prd, p, mode, test, interval)

    slopes['before_slope'] = b_slopes
    slopes['after_slope'] = a_slopes
    test_significance(slopes, 'before_slope', 'after_slope', mode, test)


def test_contributors_residuals(df, projects, interval):
    print('\n\n\t\t***** test contributors residuals before and after CoC. *****\n\n')
    mode = 'contributions'
    test = 'residuals'
    projects = {p: projects[p] for p in df[~df['adoption_date'].isnull()]['repository_url']}
    residuals = pd.DataFrame({'project': [p for p in projects.keys()]})
    b_res_mean, a_res_mean = [], []
    b_res_var, a_res_var = [], []
    for p in projects.keys():
        print('{}'.format(p))
        pdf = df[df['repository_url'] == p]
        before, after = get_before_after_data(pdf, interval)

        X = before['pull_creation_date'].values.astype(np.int64) // 10 ** 9
        X = sm.add_constant(X)
        y = before['ratio']
        b_result = fit_model(X, y)
        before_prd = b_result.params[0] + b_result.params[1] * X[:, 1]
        before_residuals = y - before_prd
        b_res_mean.append(before_residuals.mean())
        b_res_var.append(before_residuals.var())

        X = after['pull_creation_date'].values.astype(np.int64) // 10 ** 9
        X = sm.add_constant(X)
        y = after['ratio']
        # a_result = fit_model(X, y)
        after_prd = b_result.params[0] + b_result.params[1] * X[:, 1]
        after_residuals = y - after_prd
        a_res_mean.append(after_residuals.mean())
        a_res_var.append(after_residuals.var())

        stat, pvalue = mannwhitneyu(before_residuals, after_residuals, alternative='less')
        print('{} ratio {} MannWhitneyU Statistics={:.4f}, p={:.4f}\n'.format(mode, test, stat, pvalue))
        plot_fig(before, after, before_prd, after_prd, p, mode, test, interval)

    residuals['before_res_mean'] = b_res_mean
    residuals['before_res_var'] = b_res_var
    residuals['after_res_mean'] = a_res_mean
    residuals['after_res_var'] = a_res_var
    print('\t\tresidual means\n')
    test_significance(residuals, 'before_res_mean', 'after_res_mean', mode, test)
    print('\n\t\tresidual variances\n')
    test_significance(residuals, 'before_res_var', 'after_res_var', mode, test)


def test_contributions_residuals(df, projects, interval):
    print('\n\n\t\t***** test contributions residuals before and after CoC. *****\n\n')
    mode = 'contributors'
    test = 'residuals'
    projects = {p: projects[p] for p in df[~df['adoption_date'].isnull()]['repository_url']}
    residuals = pd.DataFrame({'project': [p for p in projects.keys()]})
    b_res_mean, a_res_mean = [], []
    b_res_var, a_res_var = [], []
    for p in projects.keys():
        print('{}'.format(p))
        pdf = df[df['repository_url'] == p].drop_duplicates(['user_login'])
        before, after = get_before_after_data(pdf, interval)

        X = before['pull_creation_date'].values.astype(np.int64) // 10 ** 9
        X = sm.add_constant(X)
        y = before['ratio']
        b_result = fit_model(X, y)
        before_prd = b_result.params[0] + b_result.params[1] * X[:, 1]
        before_residuals = y - before_prd
        b_res_mean.append(before_residuals.mean())
        b_res_var.append(before_residuals.var())

        X = after['pull_creation_date'].values.astype(np.int64) // 10 ** 9
        X = sm.add_constant(X)
        y = after['ratio']
        # a_result = fit_model(X, y)
        after_prd = b_result.params[0] + b_result.params[1] * X[:, 1]
        after_residuals = y - after_prd
        a_res_mean.append(after_residuals.mean())
        a_res_var.append(after_residuals.var())

        stat, pvalue = mannwhitneyu(before_residuals, after_residuals, alternative='less')
        print('{} ratio {} MannWhitneyU Statistics={:.4f}, p={:.4f}\n'.format(mode, test, stat, pvalue))
        plot_fig(before, after, before_prd, after_prd, p, mode, test, interval)

    residuals['before_res_mean'] = b_res_mean
    residuals['before_res_var'] = b_res_var
    residuals['after_res_mean'] = a_res_mean
    residuals['after_res_var'] = a_res_var
    print('\t\tresidual means\n')
    test_significance(residuals, 'before_res_mean', 'after_res_mean', mode, test)
    print('\n\t\tresidual variances\n')
    test_significance(residuals, 'before_res_var', 'after_res_var', mode, test)


def set_periods(df):
    start_dates, end_dates = [], []
    last_date = datetime.datetime(2020, 1, 1)
    adoption_dates = df['adoption_date'].tolist()
    creation_dates = df['creation_date'].tolist()
    for i in range(len(adoption_dates)):
        a = adoption_dates[i]
        c = creation_dates[i]
        if isinstance(a, str):
            adoption = datetime.datetime.strptime(a, '%Y-%m-%d %H:%M:%S')
            creation = datetime.datetime.strptime(c, '%Y-%m-%d %H:%M:%S')
            if last_date - adoption == adoption - creation:
                start_dates.append(creation.strftime('%Y-%m-%d %H:%M:%S'))
                end_dates.append(last_date.strftime('%Y-%m-%d %H:%M:%S'))
            elif last_date - adoption < adoption - creation:
                diff = last_date - adoption
                start_dates.append((adoption - diff).strftime('%Y-%m-%d %H:%M:%S'))
                end_dates.append(last_date.strftime('%Y-%m-%d %H:%M:%S'))
            else:
                diff = adoption - creation
                start_dates.append(creation.strftime('%Y-%m-%d %H:%M:%S'))
                end_dates.append((adoption + diff).strftime('%Y-%m-%d %H:%M:%S'))

        else:
            start_dates.append(np.nan)
            end_dates.append(np.nan)

    df['start_date'] = start_dates
    df['end_date'] = end_dates
    return df


def temp(df):
    # df = set_periods(df)
    df = prepare_data(df)
    # df = df[df['adoption_date'] <= pd.to_datetime('2019-01-01')]
    df = df[~df['adoption_date'].isnull()]
    from matplotlib import dates

    unique_df = df.drop_duplicates(['repository_url'])
    unique_df = unique_df.sort_values('adoption_date')
    fig, gnt = plt.subplots(figsize=(16, 6))

    # gnt.set_ylim(0, len(unique_df['repository_url'].unique()) + 1)
    gnt.set_xlim([datetime.date(2009, 6, 8), datetime.date(2022, 6, 8)])
    gnt.set_xlabel('time')
    gnt.set_ylabel('projects')
    hfmt = dates.DateFormatter('%Y-%m')
    gnt.xaxis.set_major_formatter(hfmt)
    fmt_half_year = dates.MonthLocator(interval=12)
    gnt.xaxis.set_major_locator(fmt_half_year)
    plt.axvline(unique_df['adoption_date'].median(), color='k', linewidth=1.5, label='median')
    plt.axvline(unique_df['adoption_date'].mean(), color='k', linestyle='dashed', linewidth=1.5, label='mean')
    plt.axvline(unique_df['adoption_date'].mean() + unique_df['adoption_date'].std(), color='k', linestyle='dashed',
                linewidth=0.5, label='std')
    plt.axvline(unique_df['adoption_date'].mean() - unique_df['adoption_date'].std(), color='k', linestyle='dashed',
                linewidth=0.5)

    plt.barh(y=unique_df['repository_url'], left=unique_df['start_date'],
             width=unique_df['adoption_date'] - unique_df['start_date'], color='cornflowerblue', label='before CoC')
    plt.barh(y=unique_df['repository_url'], left=unique_df['adoption_date'],
             width=unique_df['end_date'] - unique_df['adoption_date'], color='indianred', label='after CoC')
    plt.legend(loc="upper left")
    plt.savefig('periods.png')
    return df


def run(df, interval='3M'):
    df = temp(df)
    df = prepare_data(df)
    projects = df['repository_url'].unique().tolist()
    projects = {projects[i]: i + 1 for i in range(len(projects))}
    df['project'] = [projects[p] for p in df['repository_url']]
    # test 1
    # print('\n test effect of having CoC in project on project\'s Blau index.\n')
    # test_blau_with_without(df)
    print('\n')
    # test 2
    print('\ntest effect of adding a CoC to project on the ratio of underrepresented developers.')
    test_contributors_slopes(df, projects, interval)
    test_contributions_slopes(df, projects, interval)
    test_contributors_residuals(df, projects, interval)
    test_contributions_residuals(df, projects, interval)


if __name__ == '__main__':
    df = pd.read_csv(data_path + '/All_Pulls.csv')
    run(df)
   
    # data = df[~df['repository_url'].isin(['rails/rails', 'TryGhost/ghost'])]
    # wo = data[data['adoption_date'].isnull()]['repository_url'].unique().tolist()
    # w = data[~data['adoption_date'].isnull()]['repository_url'].unique().tolist()
    # fig = plt.figure(figsize=(12, 8))
    # hist_data = df[~df['repository_url'].isin(['rails/rails', 'TryGhost/ghost'])].groupby(
    #     'repository_url').size().reset_index(name='count').sort_values('count', ascending=False)
    # bar_list = plt.bar(hist_data['repository_url'], hist_data['count'], color='mediumseagreen', width=0.4)
    # wo_indexes = [i for i, p in enumerate(hist_data['repository_url']) if p in wo]
    # plt.xticks(rotation=90)
    # for i in wo_indexes:
    #     bar_list[i].set_color('salmon')
    # colors = {'With CoC': 'mediumseagreen', 'Without CoC': 'salmon'}
    # handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in colors]
    # plt.legend(handles, colors.keys())
    # plt.subplots_adjust(bottom=0.3)
    # plt.xlabel('Projects')
    # plt.ylabel('# of Pull Requests')
    # plt.savefig('Project_Distribution.eps', format='eps')

