import torch

from model.CAUM.news_encoder import NewsEncoder
from model.CAUM.news_encoder2 import NewsEncoder2
from model.CAUM.user_encoder import UserEncoder
from model.general.click_predictor.dot_product import DotProductClickPredictor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CAUM(torch.nn.Module):
    def __init__(self, config, pretrained_word_embedding=None):
        super(CAUM, self).__init__()
        self.config = config
        self.news_encoder = NewsEncoder2(config, pretrained_word_embedding)
        self.user_encoder = UserEncoder(config)
        self.click_predictor = DotProductClickPredictor()
        
    def forward(self, candidate_news, clicked_news):
        # encode user history
        clicked_news_vector = torch.stack(
            [self.news_encoder(x) for x in clicked_news], dim=1)

        # encode candidate news
        candidate_news_vector = torch.stack(
            [self.news_encoder(x) for x in candidate_news], dim=1)
        
        # learn user encoder
        scores = torch.zeros(candidate_news_vector.shape[0], candidate_news_vector.shape[1], device=device)
        scores = scores.transpose(1, 0)

        for i in range(candidate_news_vector.shape[1]):
            cand_score = self.user_encoder(clicked_news_vector, candidate_news_vector[:, i, :])
            scores[i, :] = cand_score

        scores = scores.transpose(1, 0)
    
        return scores

    def get_news_vector(self, news):
        # batch_size, num_filters * 3
        return self.news_encoder(news)

    def get_user_vector(self, clicked_news_vector):
        """
        Args:
            clicked_news_vector: batch_size, num_clicked_news_a_user, num_filters
        Returns:
            (shape) batch_size, num_filters
        """
        # batch_size, num_filters
        return clicked_news_vector

    def get_prediction(self, candidate_news_vector, clicked_news_vector):
        """
        Args:
            news_vector: candidate_size, word_embedding_dim
            user_vector: word_embedding_dim
        Returns:
            click_probability: candidate_size
        """
        # candidate_size

        #score = torch.zeros(candidate_news_vector.shape[0], candidate_news_vector.shape[1], device=device)
        #score = score.transpose(1, 0)

        #for i in range(candidate_news_vector.shape[1]):
        #    cand_score = self.user_encoder(clicked_news_vector, candidate_news_vector[:, i, :])
        #    score[i, :] = cand_score

        #score = score.transpose(1, 0)
        score = self.user_encoder(clicked_news_vector.expand(candidate_news_vector.size(0), -1, -1), candidate_news_vector)

        return score

