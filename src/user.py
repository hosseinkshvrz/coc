import math
import os
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk import StanfordNERTagger
import ethnicolr
import pandas as pd
import gender_guesser.detector as gender

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_PATH, 'data')


class User:
    def __init__(self, name_file):
        self.df = pd.read_csv(os.path.join(data_path, name_file))
        self.g_detector = gender.Detector(case_sensitive=False)
        self.ner = StanfordNERTagger(
            os.environ['HOME'] + '/nltk_data/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz',
            os.environ['HOME'] + '/nltk_data/stanford-ner/stanford-ner.jar',
            encoding='utf-8')
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.initialize()

    def initialize(self):
        # self.df = self.df.dropna()
        names = self.df['user_name'].tolist()
        genders = self.df['gender'].tolist()
        ethnicities = self.df['ethnicity'].tolist()
        names = [str(n).split() for n in names]
        is_persons = []
        for i in range(len(names)):
            if isinstance(genders[i], float) and isinstance(ethnicities[i], float):
                is_persons.append(False)
                continue
            tags = self.ner.tag(names[i])
            for t in tags:
                if t[1] == 'PERSON':
                    is_persons.append(True)
                    break
            if len(is_persons) == i:
                is_persons.append(False)
            if len(is_persons) % 500 == 0:
                print('{} records passed.'.format(len(is_persons)))
        self.df['is_person'] = is_persons
        self.df.to_csv('All_Pulls.csv')
        # first_name = [n.split()[0] for n in names]
        # last_name = [n.split()[-1] for n in names]
        # self.df['first_name'] = first_name
        # self.df['last_name'] = last_name

    def normalize_ethnicity(self):
        self.df['census'] = [e if e != 'api' else 'asian' for e in self.df['census']]
        self.df['fl_lname'] = [e if e != 'nh_white' else 'white' for e in self.df['fl_lname']]
        self.df['fl_lname'] = [e if e != 'nh_black' else 'black' for e in self.df['fl_lname']]
        self.df['fl_flname'] = [e if e != 'nh_white' else 'white' for e in self.df['fl_flname']]
        self.df['fl_flname'] = [e if e != 'nh_black' else 'black' for e in self.df['fl_flname']]

    def get_ethnicity_agreement(self):
        data = list(zip(self.df['census'], self.df['fl_lname'], self.df['fl_flname']))
        ethnicities = []
        for row in data:
            majority = Counter(row).most_common()[0]
            if majority[1] == 1:
                ethnicities.append(row[2])  # based on full name
            else:
                ethnicities.append(majority[0])
        self.df['ethnicity'] = ethnicities

    def collect_gender(self):
        genders = []
        for fname in self.df['first_name']:
            g = self.g_detector.get_gender(fname)
            if g.startswith('mostly'):
                g = g.split('mostly_')[-1]
            elif g == 'andy':
                g = 'androgynous'
            genders.append(g)
        self.df['gender'] = genders

    def collect_ethnicity(self):
        ethnicolr.pred_census_ln(self.df, 'last_name')  # 2000 and 2010 yield same result
        self.df = self.df.rename(columns={'race': 'census'})

        ethnicolr.pred_fl_reg_ln(self.df, 'last_name')
        self.df = self.df.rename(columns={'race': 'fl_lname'})

        ethnicolr.pred_fl_reg_name(self.df, 'last_name', 'first_name')
        self.df = self.df.rename(columns={'race': 'fl_flname'})

    def store_demographic(self):
        self.collect_gender()
        self.collect_ethnicity()
        self.normalize_ethnicity()
        self.get_ethnicity_agreement()
        self.df.drop(columns=['first_name', 'last_name'])
        self.df[['user_login', 'real_name', 'gender', 'ethnicity']] \
            .to_csv(os.path.join(data_path, 'demographic.csv'), index=False)


if __name__ == '__main__':
    user = User('All_Pulls.csv')
    # user.store_demographic()
    print('finished.')
    # df = pd.read_csv(os.path.join(data_path, 'All_Pulls.csv'))
    print()
