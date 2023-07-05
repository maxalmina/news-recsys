# Adapted from https://github.com/taoqi98/CAUM/blob/main/Code/Models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.attention.additive import AdditiveAttention
from model.general.attention.multihead_self import MultiHeadSelfAttention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TextEncoder(nn.Module):
    def __init__(self,
                 word_embedding,
                 word_embedding_dim,
                 num_attention_heads,
                 dropout_probability,
                 query_vector_dim):
        super(TextEncoder, self).__init__()

        self.word_embedding = word_embedding

        self.multihead_attention = MultiHeadSelfAttention(word_embedding_dim,
                                                          num_attention_heads)

        self.additive_attention = AdditiveAttention(query_vector_dim,
                                                    word_embedding_dim)

        self.dropout_probability = dropout_probability

    def forward(self, text):
        text_vector = F.dropout(self.word_embedding(text),
                                p=self.dropout_probability,
                                training=self.training)

        # mhsa
        multihead_text_vector = self.multihead_attention(text_vector)

        multihead_text_vector = F.dropout(multihead_text_vector,
                                          p=self.dropout_probability,
                                          training=self.training)
        
        #additive
        final_text_vector = self.additive_attention(multihead_text_vector)

        return final_text_vector

class CategoryEncoder(nn.Module):
    def __init__(self,
                 category_embedding,
                 category_embedding_dim,
                 dropout_probability):
        super(CategoryEncoder, self).__init__()

        self.category_embedding = category_embedding

        self.linear = nn.Linear(category_embedding_dim, category_embedding_dim)

        self.dropout_probability = dropout_probability

    def forward(self, category):
        category_vector = F.dropout(self.category_embedding(category),
                                    p=self.dropout_probability,
                                    training=self.training)
        category_vector = F.relu(self.linear(category_vector))

        return category_vector

class NewsEncoder(torch.nn.Module):
    def __init__(self, config, pretrained_word_embeddings):
        super(NewsEncoder, self).__init__()
        self.config = config
        
        if pretrained_word_embeddings is None:
            word_embedding = nn.Embedding(config.num_words,
                                               config.word_embedding_dim,
                                               padding_idx = 0)
        else:
            word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embeddings, freeze=False, padding_idx = 0)
        
        self.text_encoder = TextEncoder(word_embedding,
                                        config.word_embedding_dim,
                                        config.num_attention_heads,
                                        config.dropout_probability,
                                        config.query_vector_dim)

        category_embedding = nn.Embedding(
                num_embeddings = config.num_categories, 
                embedding_dim = config.category_embedding_dim, 
                padding_idx  = 0)

        self.category_encoder = CategoryEncoder(category_embedding, config.category_embedding_dim, config.dropout_probability)

        self.linear = nn.Linear(
                in_features = config.word_embedding_dim,
                out_features = config.news_vector_dim)

    def forward(self, news):
        category_vector = self.category_encoder(news['category'].to(device))

        text_vector = self.text_encoder(news['title'].to(device))

        all_vectors = torch.cat(
                [text_vector, category_vector],
                dim=-1)

        news_vector = self.linear(all_vectors)

        return news_vector
