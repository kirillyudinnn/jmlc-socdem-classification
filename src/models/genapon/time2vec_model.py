import torch
import torch.nn as nn

from typing import Dict, Any


class Time2Vec(nn.Module):
    """
        Time2Vec

    Layer for learning event time vector representation
    Default features:
        1) Day of week
        2) Day of month
        3) Number of week in training time span
        4) Local day time (hour)

    See https://arxiv.org/pdf/1907.05321

    Args:
        Config (Dict): Time2Vec configuration
        
        Keys:
            in_features (int): count of time features 
            emb_dim (int): time vector size
    """
    def __init__(self, T2vConfig: Dict[str, Any]):
        super().__init__()

        self.in_features = T2vConfig["in_features"]
        self.emb_dim = T2vConfig["emb_dim"]
        self.periodic_f = torch.sin

        # Non-periodic weights
        self.W = nn.Parameter(
            data=nn.init.xavier_uniform_(torch.rand(self.in_features, 1))
        )
        self.b = nn.Parameter(
            data=torch.rand(1)
        )

        # Periodic weights
        self.W_p = nn.Parameter(
            data=nn.init.xavier_uniform_(torch.rand(self.in_features, self.emb_dim - 1))
        )
        self.b_p = nn.Parameter(
            data=torch.rand(1, self.emb_dim - 1)
        )


    def forward(self, tau):
        non_periodic_t = tau @ self.W + self.b
        periodic_t = self.periodic_f(tau @ self.W_p) + self.b_p

        return torch.concat([non_periodic_t, periodic_t], dim=2)
