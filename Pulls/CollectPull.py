import os.path
import argparse
from github import Github
import pandas as pd
import logging
from Token import Token
from logging.handlers import RotatingFileHandler
from pathlib import Path
import time


class GithubCollector(object):

    def __init__(self):
        self.token_list = Token.get_token_list()
        self.github = Github(self.token_list[0].token)
        Path("Logs/").mkdir(parents=True, exist_ok=True)
        self.dump_rate = 2
        logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                            level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[
                                RotatingFileHandler(filename='Logs/GithubCollection.log', maxBytes=5 * 1024 * 1024,
                                                    backupCount=5)])

    def dump_data(self, row_list):
        file_name = "{}_Pull_Information.csv".format(self.repo_name)
        if os.path.isfile(file_name):
            df_prev = pd.read_csv(file_name)
        else:
            df_prev = pd.DataFrame(
                columns=["repository_url", "user_login", "user_name", "user_location", "pull_number", "pull_current_state",
                         "pull_creation_date", "pull_closing_date", "pull_merged", "pull_merge_date", "pull_comment_count"])
        df_new = pd.DataFrame(row_list)
        df = pd.concat([df_prev, df_new])
        df.to_csv(file_name, index=False)

    def extract_data(self, pull_list, url):
        data_list = []
        count = 0
        for pull_request in pull_list:
            logging.info("Completed {} %".format(round((count/pull_list.totalCount) * 100, 2)))
            Token.update_token(self.github, token_list=self.token_list)
            pull_request_information = {
                "repository_url": url,
                "user_login": pull_request.user.login,
                "user_name": pull_request.user.name,
                "user_location": pull_request.user.location,
                "pull_number": pull_request.number,
                "pull_current_state": pull_request.state,
                "pull_creation_date": pull_request.created_at,
                "pull_closing_date": pull_request.closed_at,
                "pull_merged": pull_request.merged,
                "pull_merge_date": pull_request.merged_at,
                "pull_comment_count": pull_request.review_comments
            }
            data_list.append(pull_request_information)
            count += 1
            if len(data_list) % self.dump_rate == 0:
                self.dump_data(data_list)
                data_list = []

    def start(self, user_name, repo_name, limit=None):
        self.repo_name = repo_name
        url = "{}/{}".format(user_name, repo_name)
        repository = self.github.get_repo(url)
        repository_open_pulls = repository.get_pulls(state="open", direction="desc")
        self.extract_data(repository_open_pulls, url=url)
        repository_closed_pulls = repository.get_pulls(state="closed", direction="desc")
        self.extract_data(repository_closed_pulls, url=url)
        Token.dump_all_token(self.token_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default=None, type=str, help="")
    parser.add_argument("--project", default=None, type=str, help="")
    args = parser.parse_args()
    github = GithubCollector()
    github.start(user_name=args.user, repo_name=args.project)
