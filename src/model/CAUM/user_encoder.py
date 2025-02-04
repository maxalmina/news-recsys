# Adapted from https://github.com/taoqi98/CAUM/blob/main/Code/Models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.general.attention.multihead_self import MultiHeadSelfAttention
from model.general.attention.dense_att import DenseAttention

class UserEncoder(nn.Module):
    def __init__(self, config):
        super(UserEncoder, self).__init__()
        self.config = config

        self.dropout_probability = config.dropout_probability

        self.linear1 = nn.Linear(config.news_vector_dim * 4, config.num_filters)
        self.linear2 = nn.Linear(config.news_vector_dim * 2, config.user_vector_dim)
        self.linear3 = nn.Linear(config.num_filters + config.user_vector_dim, config.user_vector_dim)

        self.dense_att = DenseAttention(
                input_dim=config.user_vector_dim * 2,
                hidden_dim1=config.dense_att_hidden_dim1,
                hidden_dim2=config.dense_att_hidden_dim2
                )

        self.multihead_attention = MultiHeadSelfAttention(config.user_vector_dim, config.num_attention_heads)

    def forward(self, clicked_news_vector, cand_news_vector):
        cand_news_vector = F.dropout(cand_news_vector,
                                     p=self.dropout_probability,
                                     training=self.training)

        clicked_news_vector = F.dropout(clicked_news_vector,
                                     p=self.dropout_probability,
                                     training=self.training)

        repeated_cand_news_vector = cand_news_vector.unsqueeze(dim=1).repeat(1, clicked_news_vector.shape[1], 1)

        # candi-cnn
        clicked_news_left = torch.cat(
                [clicked_news_vector[:, -1:, :], clicked_news_vector[:, :-1, :]],
                dim=-2
                )
        clicked_news_right =  torch.cat(
                [clicked_news_vector[:, 1:, :], clicked_news_vector[:, :1, :]],
                dim=-2
                )
        clicked_news_cnn = torch.cat(
                [
                    clicked_news_left, 
                    clicked_news_vector, 
                    clicked_news_right, 
                    repeated_cand_news_vector
                    ],
                dim=-1
                )

        clicked_news_cnn = self.linear1(clicked_news_cnn)

        # candi-selfatt
        clicked_news = torch.cat(
                [repeated_cand_news_vector, clicked_news_vector],
                dim=-1) 
        clicked_news = self.linear2(clicked_news)
        clicked_news_self = self.multihead_attention(clicked_news)

        clicked_news_all = torch.cat(
                [clicked_news_cnn, clicked_news_self], 
                dim=-1
                )

        clicked_news_all = F.dropout(clicked_news_all,
                                     p=self.dropout_probability,
                                     training=self.training)

        clicked_news_all = self.linear3(clicked_news_all)

        # candi-att
        attention_vector = torch.cat(
                [clicked_news_all, repeated_cand_news_vector],
                dim=-1)

        attention_score = self.dense_att(attention_vector)
        attention_score = attention_score.squeeze(dim=-1)
        attention_score = F.softmax(attention_score, dim=-1)

        user_vector = torch.bmm(attention_score.unsqueeze(dim=1), clicked_news_all).squeeze(dim=1)

        score = torch.bmm(cand_news_vector.unsqueeze(dim=1), user_vector.unsqueeze(dim=-1)).flatten()

        return score
