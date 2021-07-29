import json
import math
import os
import pandas as pd
import numpy as np
import datetime

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_PATH, 'data')


class PullRequest:
    def __init__(self, adoption_dates):
        self.adoption_dates = adoption_dates

    def separate_prs(self, project):
        adopt_d = datetime.datetime.strptime(self.adoption_dates[project], '%Y-%m-%d %H:%M:%S')
        df = pd.read_csv(os.path.join(data_path, 'prs', '{}_Pull_Information.csv'.format(project)))
        df['pull_creation_date'] = pd.to_datetime(df['pull_creation_date'])
        df = df[~df['user_login'].str.endswith('[bot]')]
        unique_users = df['user_login'].nunique()

        before_df = df[df['pull_creation_date'] <= adopt_d]\
            .groupby(['user_login']).size()\
            .reset_index(name='n_pr_before_coc')
        after_df = df[df['pull_creation_date'] > adopt_d] \
            .groupby(['user_login']).size() \
            .reset_index(name='n_pr_after_coc')

        result_df = pd.merge(before_df, after_df, how='outer', on=['user_login'])
        result_df['n_pr_before_coc'] = result_df['n_pr_before_coc'].fillna(0).astype('int64')
        result_df['n_pr_after_coc'] = result_df['n_pr_after_coc'].fillna(0).astype('int64')

        df = df[['repository_url', 'user_login', 'user_name']].drop_duplicates()
        result_df = pd.merge(df, result_df, on='user_login', validate='one_to_one')
        assert len(result_df) == unique_users

        return result_df

    def set_periods(self, df):
        start_dates, end_dates = [], []
        last_date = datetime.datetime(2021, 6, 8)
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


if __name__ == '__main__':
    with open(os.path.join(data_path, 'coc_adoption.json')) as file:
        dates = json.load(file)
    pr = PullRequest(dates)
    # for project, _ in dates.items():
    #     df = pr.separate_prs(project)
    #     df.to_csv(os.path.join(data_path, 'prs/contributors', '{}_contributors.csv'.format(project)), index=False)
    df = pd.read_csv(os.path.join(data_path, 'All_Pulls.csv'))
    df = pr.set_periods(df)
    df.to_csv(os.path.join(data_path, 'All_Pulls.csv'), index=False)
    print()
