import datetime
import json
import os
from time import sleep

import pandas as pd
import requests

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_PATH, 'data')


class GitMiner:
    def __init__(self):
        self.base_url = 'https://api.github.com'
        with open(os.path.join(BASE_PATH, 'conf', 'auth.conf'), 'r') as file:
            lines = file.readlines()
        self.token = lines[2].split('\n')[0]
        self.session = requests.Session()
        self.headers = {'Authorization': 'token ' + self.token,
                        'content-type': 'application/json'}
        self.popular_repos = []

    def get_popular_repos(self):
        self.headers['accept'] = 'application/vnd.github.cloak-preview'
        api = '/search/repositories?q=stars:>1&sort=stars'
        response = self.session.get(self.base_url + api, headers=self.headers)
        results = json.loads(response.text)
        # results['items'][0]['full_name']
        print()

    def get_projects_with_coc(self):
        self.headers['accept'] = 'application/vnd.github.v3+json'
        # repos that contain a code of conduct in their root directory
        api = '/search/code?q=filename:code_of_conduct+path%3A%2F&sort=indexed&order=asc&per_page=100&page='
        repo, path = [], []
        for i in range(1, 11):     # only first 1000 results
            response = self.session.get(self.base_url+api+str(i), headers=self.headers)
            results = json.loads(response.text)
            while True:
                try:
                    repo += [item['repository']['full_name'] for item in results['items']]
                    path += [item['path'] for item in results['items']]
                    break
                except KeyError:    # rate limit
                    sleep(90)
                    pass
            sleep(30)
        return repo, path

    def get_project_features(self, projects):
        self.headers['accept'] = 'application/vnd.github.v3+json'
        api = '/repos/'
        last_date = datetime.datetime(2021, 6, 8).timestamp()
        stars, age, size = [], [], []
        for p in projects:
            response = self.session.get(self.base_url + api + p, headers=self.headers, verify=False)
            results = json.loads(response.text)
            stars.append(results['stargazers_count'])
            size.append(results['size'])
            created_at = datetime.datetime.strptime(results['created_at'], '%Y-%m-%dT%H:%M:%SZ').timestamp()
            age.append(int(last_date - created_at))

        pd.DataFrame({'repository_url': projects, 'star': stars, 'size': size, 'age': age})\
            .to_csv('repo_info.csv', index=False)


if __name__ == '__main__':
    miner = GitMiner()
    projects = pd.read_csv(os.path.join(data_path, 'All_Pulls.csv'))['repository_url'].unique().tolist()
    miner.get_project_features(projects)
    # repo, path = miner.get_projects_with_coc()
    # df = pd.DataFrame({'repo': repo, 'path': path})
    # df.to_csv(os.path.join(data_path, 'coc_repos.csv'), index=False)
    # from os import listdir
    # from os.path import isfile, join
    # mypath = os.path.join(data_path, 'prs')
    # onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    # print(onlyfiles)
    # pr_size = []
    # for i in onlyfiles:
    #     df = pd.read_csv(os.path.join(mypath, i))
    #     pr_size.append(len(df))
    # onlyfiles = [i.split('_')[0] for i in onlyfiles]
    # df = pd.DataFrame({'project': onlyfiles, 'n_pr': pr_size})
    # df.to_csv(os.path.join(mypath, 'stats.csv'))
