# Adapted from https://github.com/taoqi98/CAUM/blob/main/Code/Models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.attention.additive import AdditiveAttention
from model.general.attention.multihead_self import MultiHeadSelfAttention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NewsEncoder2(torch.nn.Module):
    def __init__(self, config, pretrained_word_embedding):
        super(NewsEncoder2, self).__init__()
        self.config = config

        # title/abstract/content section
        if pretrained_word_embedding is None:
            self.word_embedding = nn.Embedding(config.num_words,
                                               config.word_embedding_dim,
                                               padding_idx=0)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(pretrained_word_embedding,
                                                               freeze=False,
                                                               padding_idx=0)
        
        self.multihead_self_attention = MultiHeadSelfAttention(config.word_embedding_dim,
                                                               config.num_attention_heads)

        self.additive_attention = AdditiveAttention(config.query_vector_dim,
                                                    config.word_embedding_dim)

        # category section
        self.category_embedding = nn.Embedding(num_embeddings=config.num_categories,
                                               embedding_dim=config.category_embedding_dim,
                                               padding_idx=0)

        self.linear_cat = nn.Linear(config.category_embedding_dim,
                                 config.category_embedding_dim)

        # final section
        self.linear_final = nn.Linear(config.word_embedding_dim+ # dim_title
                                      config.category_embedding_dim+ # dim_category
                                      #config.category_embedding_dim+ # dim_subcategory
                                      config.word_embedding_dim+ #dim_abstract
                                      config.word_embedding_dim, #dim_content
                                      config.news_vector_dim # output_dim
                                      )

    def forward(self, news):
        """
        Args:
            news:
                {
                    "title": batch_size * num_words_title
                }
        Returns:
            (shape) batch_size, word_embedding_dim
        """

        # title
        # batch_size, num_words_title, word_embedding_dim
        news_vector = F.dropout(self.word_embedding(news["title"].to(device)),
                                p=self.config.dropout_probability,
                                training=self.training)

        # batch_size, num_words_title, word_embedding_dim
        multihead_news_vector = self.multihead_self_attention(news_vector)
        multihead_news_vector = F.dropout(multihead_news_vector,
                                          p=self.config.dropout_probability,
                                          training=self.training)

        # batch_size, word_embedding_dim
        additive_news_vector = self.additive_attention(multihead_news_vector)

        # abstract
        # batch_size, num_words_title, word_embedding_dim
        abs_news_vector = F.dropout(self.word_embedding(news["abstract"].to(device)),
                                p=self.config.dropout_probability,
                                training=self.training)

        # batch_size, num_words_title, word_embedding_dim
        abs_multihead_news_vector = self.multihead_self_attention(abs_news_vector)
        abs_multihead_news_vector = F.dropout(abs_multihead_news_vector,
                                          p=self.config.dropout_probability,
                                          training=self.training)

        # batch_size, word_embedding_dim
        abs_additive_news_vector = self.additive_attention(abs_multihead_news_vector)

        # content
        # batch_size, num_words_title, word_embedding_dim
        #cont_news_vector = F.dropout(self.word_embedding(news["content"].to(device)),
        #                             p=self.config.dropout_probability,
        #                             training=self.training)

        ## batch_size, num_words_title, word_embedding_dim
        #cont_multihead_news_vector = self.multihead_self_attention(cont_news_vector)
        #cont_multihead_news_vector = F.dropout(cont_multihead_news_vector,
        #                                       p=self.config.dropout_probability,
        #                                       training=self.training)

        ## batch_size, word_embedding_dim
        #cont_additive_news_vector = self.additive_attention(cont_multihead_news_vector)

        # category
        cat_vector = F.dropout(self.category_embedding(news["category"].to(device)),
                               p=self.config.dropout_probability,
                               training=self.training)
        cat_vector = F.relu(self.linear_cat(cat_vector))

        # subcategory
        #subcat_vector = F.dropout(self.category_embedding(news["subcategory"].to(device)),
        #                          p=self.config.dropout_probability,
        #                          training=self.training)

        #subcat_vector = F.relu(self.linear_cat(subcat_vector))
        
        # final vector
        all_vector = torch.cat([additive_news_vector,
                                cat_vector,
                                #subcat_vector,
                                abs_additive_news_vector,
                                #cont_additive_news_vector
                                ],
                               dim=-1)

        final_news_vector = self.linear_final(all_vector)

        return final_news_vector
