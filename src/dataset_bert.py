from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval
from os import path
import numpy as np
from config import model_name
import importlib
import torch

try:
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
except AttributeError:
    print(f"{model_name} not included!")
    exit()

class BaseDataset(Dataset):
    def __init__(self, behaviors_path, news_path, bert_embedding_dir):
        super(BaseDataset, self).__init__()
        assert all(attribute in [
            'category', 'subcategory', 'title', 'abstract', 'title_entities',
            'abstract_entities', 'content', 'title_bert', 'title_mask_bert',
            'abstract_bert', 'abstract_mask_bert', 'content_bert', 'content_mask_bert'
        ] for attribute in config.dataset_attributes['news'])
        assert all(attribute in ['user', 'clicked_news_length']
                   for attribute in config.dataset_attributes['record'])

        self.behaviors_parsed = pd.read_table(behaviors_path)
        self.news_parsed = pd.read_table(
            news_path,
            index_col='id',
            usecols=['id'] + config.dataset_attributes['news'],
            converters={
                attribute: literal_eval
                for attribute in set(config.dataset_attributes['news']) & set([
                    'title', 'abstract', 'title_entities', 'abstract_entities', 'content',
                    'title_bert', 'title_mask_bert', 'abstract_bert', 'abstract_mask_bert',
                    'content_bert', 'content_mask_bert'
                ])
            })

        self.news_id2int = {x: i for i, x in enumerate(self.news_parsed.index)}
        self.news2dict = self.news_parsed.to_dict('index')
        for key1 in self.news2dict.keys():
            for key2 in self.news2dict[key1].keys():
                self.news2dict[key1][key2] = torch.tensor(
                    self.news2dict[key1][key2])

        padding_all = {
            'category': 0,
            'subcategory': 0,
            'title': [0] * config.num_words_title,
            'abstract': [0] * config.num_words_abstract,
            'title_entities': [0] * config.num_words_title,
            'abstract_entities': [0] * config.num_words_abstract,
            'content' : [0] * config.num_words_content,
            'title_bert': [0] * config.num_words_title,
            'title_mask_bert': [0] * config.num_words_title,
            'abstract_bert': [0] * config.num_words_abstract,
            'abstract_mask_bert': [0] * config.num_words_abstract,
            'content_bert': [0] * config.num_words_content,
            'content_mask_bert' : [0] * config.num_words_content
        }

        for key in padding_all.keys():
            padding_all[key] = torch.tensor(padding_all[key])

        self.padding = {
            k: v
            for k, v in padding_all.items()
            if k in config.dataset_attributes['news']
        }

        if model_name == 'Exp1' and not config.fine_tune:
            if config.bert_level == 'word':
                self.bert_embedding = {
                    k: torch.from_numpy(
                        np.load(
                            path.join(bert_embedding_dir,
                                      f'{k}_last_hidden_state.npy'))).float()
                    for k in set(config.dataset_attributes['news'])
                    & set(['title', 'abstract', 'content'])
                }
                name2length = {
                    'title': config.num_words_title,
                    'abstract': config.num_words_abstract,
                    'content': config.num_words_content
                }
                for k in set(config.dataset_attributes['news']) & set(
                    ['title', 'abstract', 'content']):
                    self.padding[k] = torch.zeros((name2length[k], 768))

            elif config.bert_level == 'sentence':
                self.bert_embedding = {
                    k: torch.from_numpy(
                        np.load(
                            path.join(bert_embedding_dir,
                                      f'{k}_pooler_output.npy'))).float()
                    for k in set(config.dataset_attributes['news'])
                    & set(['title', 'abstract', 'content'])
                }
                for k in set(config.dataset_attributes['news']) & set(
                    ['title', 'abstract', 'content']):
                    self.padding[k] = torch.zeros(768)
    
    def _news2dict(self, id):
        ret = self.news2dict[id]
        if model_name == 'Exp1' and not config.fine_tune:
            for k in set(config.dataset_attributes['news']) & set(
                ['title', 'abstract', 'content']):
                ret[k] = self.bert_embedding[k][self.news_id2int[id]]
        return ret

    def __len__(self):
        return len(self.behaviors_parsed)

    def __getitem__(self, idx):
        item = {}
        row = self.behaviors_parsed.iloc[idx]
        if 'user' in config.dataset_attributes['record']:
            item['user'] = row.user
        item["clicked"] = list(map(int, row.clicked.split()))
        item["candidate_news"] = [
            self._news2dict(x) for x in row.candidate_news.split()
        ]
        item["clicked_news"] = [
            self._news2dict(x)
            for x in row.clicked_news.split()[:config.num_clicked_news_a_user]
        ]
        if 'clicked_news_length' in config.dataset_attributes['record']:
            item['clicked_news_length'] = len(item["clicked_news"])
        repeated_times = config.num_clicked_news_a_user - \
            len(item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"] = [self.padding
                                ] * repeated_times + item["clicked_news"]

        return item
