import torch.nn as nn
from typing import Dict, Any


class PhiFunction(nn.Module):
    """
        Phi Function for set of video embeddings

    Args:
        PhiConfig (Dict): PhiFunction configuration
        
        Keys:
            emb_dim (int): video embedding size (must be equal to size from model's video_embeddings layer)
            out_features (int): output video embedding size
    """
    def __init__(self, PhiConfig: Dict[str, Any]):
        super().__init__()
        self.emb_dim = PhiConfig["emb_dim"]
        self.out_features = PhiConfig["out_features"]
        self.dropout_prob = PhiConfig.get("dropout_prob", 0.25)

        self.fc1 = nn.Linear(in_features=self.emb_dim, out_features=self.emb_dim)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.fc2 = nn.Linear(in_features=self.emb_dim, out_features=self.out_features)
        self.relu = nn.ReLU()

    def forward(self, X):
        X = self.fc1(X)
        X = self.relu(X)
        X = self.dropout(X)
        X = self.fc2(X)

        return X


class RhoFunction(nn.Module):
    """
        Rho Function for summarized set of video embeddings

    Args:
        RhoConfig (Dict): RhoFunction configuration

        Keys: 
            in_features (int): video embedding size after phi function applying (must be equal to PhiConfig["out_features"])
            out_features (int): embedding size of result

    """
    def __init__(self, RhoConfig: Dict[str, Any]):
        super().__init__()
        self.in_features = RhoConfig["in_features"]
        self.out_features = RhoConfig["out_features"]
        self.dropout_prob = RhoConfig.get("dropout_prob", 0.25)
        
        self.fc1 = nn.Linear(in_features=self.in_features, out_features=self.out_features)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.fc2 = nn.Linear(in_features=self.out_features, out_features=self.out_features)
        self.relu = nn.ReLU()


    def forward(self, X):
        X = self.fc1(X)
        X = self.relu(X)
        X = self.dropout(X)
        X = self.fc2(X)

        return X


class DeepSetsModel(nn.Module):
    """
    Implementation Deep Sets 
    
    This function is invariant on permutations of the set of video ids. 

    See https://arxiv.org/pdf/1703.06114

    f(X) = р(Σф(x)), where X - set of user's video embeddings
             X э x

    Args:
        PhiConfig (Dict): Config for phi function
        RhoConfig (Dict): Config for rho function
    """
    def __init__(self, PhiConfig: Dict[str, Any], RhoConfig: Dict[str, Any]):
        super().__init__()
        self.phi = PhiFunction(PhiConfig)
        self.rho = RhoFunction(RhoConfig)


    def forward(self, X):
        X_phi = self.phi(X)
        X_phi = X_phi.sum(dim=1, keepdim=True)
        X_rho = self.rho(X_phi)

        return X_rho
