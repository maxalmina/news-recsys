# Adapted from https://github.com/taoqi98/CAUM/blob/main/Code/Models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.attention.additive import AdditiveAttention
from model.general.attention.multihead_self import MultiHeadSelfAttention
from transformers import AutoModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TextEncoder(nn.Module):
    def __init__(self,
                 bert,
                 word_embedding_dim,
                 num_attention_heads,
                 dropout_probability,
                 query_vector_dim,
                 bert_level):
        super(TextEncoder, self).__init__()

        self.bert = bert

        self.bert_level = bert_level

        self.reduce_dim = nn.Linear(768, word_embedding_dim)

        self.multihead_attention = MultiHeadSelfAttention(word_embedding_dim,
                                                          num_attention_heads)

        self.additive_attention = AdditiveAttention(query_vector_dim,
                                                    word_embedding_dim)

        self.dropout_probability = dropout_probability

    def forward(self,
                input_ids=None,
                attention_mask=None,
                last_hidden_state=None,
                pooler_output=None):

        if self.bert is not None:
            embeddings = self.bert(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   return_dict=True)

        if self.bert_level == 'word':
            # batch_size, num_words_text, word_embedding_dim
            text_vector = F.dropout(self.reduce_dim(embeddings['last_hidden_state']
                                                    if self.bert is not None else last_hidden_state),
                                    p=self.dropout_probability,
                                    training=self.training)

            # batch_size, num_words_text, word_embedding_dim
            multihead_text_vector = self.multihead_self_attention(text_vector)
            multihead_text_vector = F.dropout(multihead_text_vector,
                                              p=self.dropout_probability,
                                              training=self.training)
            # batch_size, word_embedding_dim
            return self.additive_attention(multihead_text_vector)

        elif self.bert_level == 'sentence':   
            # batch_size, word_embedding_dim
            return self.reduce_dim(embeddings['pooler_output']
                                   if self.bert is not None else pooler_output)

class CategoryEncoder(nn.Module):
    def __init__(self,
                 category_embedding,
                 category_embedding_dim,
                 #word_embedding_dim,
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
    def __init__(self, config):
        super(NewsEncoder, self).__init__()
        self.config = config
        assert len(config.dataset_attributes['news']) > 0

        #self.text_encoders = nn.ModuleDict()

        if config.fine_tune:
            bert = AutoModel.from_pretrained('indobenchmark/indobert-large-p2')
            if self.training:
                bert.train()
        else:
            bert = None
        
        self.text_encoder = TextEncoder(bert,
                                        config.word_embedding_dim,
                                        config.num_attention_heads,
                                        config.dropout_probability,
                                        config.query_vector_dim,
                                        config.bert_level)

        category_embedding = nn.Embedding(config.num_categories,
                                          config.category_embedding_dim,
                                          padding_idx  = 0)

        self.category_encoder = CategoryEncoder(category_embedding,
                                                config.category_embedding_dim,
                                                config.dropout_probability)

        #element_encoders_candidates = ['category', 'subcategory']
        #self.element_encoders = nn.ModuleDict({
        #    name:
        #    CategoryEncoder(category_embedding, config.category_embedding_dim, config.word_embedding_dim, config.dropout_probability)
        #    for name in (set(config.dataset_attributes['news'])
        #                    & set(element_encoders_candidates))
        #})

        #if len(config.dataset_attributes['news']) > 1:
        #    self.final_attention = AdditiveAttention(config.query_vector_dim,
        #                                                config.word_embedding_dim)

        self.linear = nn.Linear(config.word_embedding_dim+ # dim_title
                                config.category_embedding_dim+ # dim_category
                                config.category_embedding_dim+ # dim_subcategory
                                config.word_embedding_dim, #dim_abstract
                                #config.word_embedding_dim, #dim_content
                                config.news_vector_dim # output_dim
        )

    def forward(self, news):
        #if self.config.fine_tune:
        #    text_vectors = [
        #        encoder(input_ids=news[f"{name}_bert"].to(device),
        #                attention_mask=news[f"{name}_mask_bert"].to(device))
        #        for name, encoder in self.text_encoders.items()
        #    ]
        #elif self.config.bert_level == 'word':
        #    text_vectors = [
        #        encoder(last_hidden_state=news[name].to(device))
        #        for name, encoder in self.text_encoders.items()
        #    ]
        #elif self.config.bert_level == 'sentence':
        #    text_vectors = [
        #        encoder(pooler_output=news[name].to(device))
        #        for name, encoder in self.text_encoders.items()
        #    ]

        text_vector = self.text_encoder(#input_ids=news["title_bert"].to(device),
                                        #attention_mask=news["title_mask_bert"].to(device)
                                        pooler_output=news["title"].to(device)
                                        )

        category_vector = self.category_encoder(news['category'].to(device))
        subcategory_vector = self.category_encoder(news['subcategory'].to(device))

        abstract_vector = self.text_encoder(#input_ids=news["content_bert"].to(device),
                                            #attention_mask=news["content_mask_bert"].to(device)
                                            pooler_output=news["content"].to(device)
                                            )

        all_vectors = torch.cat(
                [text_vector, category_vector, subcategory_vector, abstract_vector],
                dim=-1
                )

        #all_vectors = text_vectors + element_vectors

        #if len(all_vectors) == 1:
        #    final_news_vector = all_vectors[0]
        #else:
        #    final_news_vector = self.final_attention(
        #        torch.stack(all_vectors, dim=1))

        final_news_vector = self.linear(all_vectors)

        return final_news_vector
