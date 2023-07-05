import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseAttention(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim1,
            hidden_dim2
            ):
        super().__init__()

        self.linear = nn.Linear(input_dim, hidden_dim1)
        self.tanh1 = nn.Tanh()
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.tanh2 = nn.Tanh()
        self.linear3 = nn.Linear(hidden_dim2, 1)

    def forward(self, input_vector):
        transformed_vector = self.linear(input_vector)
        transformed_vector = self.tanh1(transformed_vector)
        transformed_vector = self.linear2(transformed_vector)
        transformed_vector = self.tanh2(transformed_vector)
        transformed_vector = self.linear3(transformed_vector)

        return transformed_vector