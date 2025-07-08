import torch
import torch.nn as nn

from typing import Dict

from genapon.deepsets_model import DeepSetsModel
from genapon.time2vec_model import Time2Vec 
from genapon.ffn import FeedForwardNetwork


class GENAPON(nn.Module):
    """
        GENAPON – 

    Args:
        RhoConfig (Dict): Rho Configuration for DeepSets layer
        PhiConfig (Dict): Phi Configuration for DeepSets layer
        T2vConfig (Dict): Time2vec Configuration
        ModelConfig (Dict): GENAPON Configuration

        Keys:
            device (str): training device
            v_num_embeddings (int): count of video ids
            v_embedding_dim (int): video embedding size

            time_vector_input_size (int): time embedding size (must be equal to emb_dim from t2v layer)
            time_vector_hidden_size (int): hidden time embedding size
    """
    def __init__(
            self,
            ModelConfig: Dict[str, int],
            RhoConfig: Dict[str, int],
            PhiConfig: Dict[str, int],
            T2vConfig: Dict[str, int],
    ):
        
        super(GENAPON, self).__init__()

        self.deepsets = DeepSetsModel(PhiConfig, RhoConfig)
        self.t2v = Time2Vec(T2vConfig)

        self.device = ModelConfig["device"]

        self.video_embeddings = nn.Embedding(
            num_embeddings=ModelConfig["v_num_embeddings"],
            embedding_dim=ModelConfig["v_embedding_dim"],
            device=self.device
        )

        self.input_size = ModelConfig["time_vector_input_size"]
        self.hidden_size = ModelConfig["time_vector_hidden_size"]
        self.time_activity_lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            device=self.device,
        )

        self.v_ffn = FeedForwardNetwork(
            input_dim=RhoConfig["out_features"],
            hidden_dim=ModelConfig["v_hidden_dim"],
            output_dim=ModelConfig["v_output_dim"],
            dropout_prob=ModelConfig.get("v_dropout_prob", 0.25),
            device=self.device
        )

        self.t_ffn = FeedForwardNetwork(
            input_dim=ModelConfig["time_vector_hidden_size"],
            hidden_dim=ModelConfig["t_hidden_dim"],
            output_dim=ModelConfig["t_output_dim"],
            dropout_prob=ModelConfig.get("t_dropout_prob", 0.25),
            device=self.device
        )

        self.u_ffn = FeedForwardNetwork(
            input_dim=ModelConfig["user_features_dim"],
            hidden_dim=ModelConfig["u_hidden_dim"],
            output_dim=ModelConfig["u_output_dim"],
            dropout_prob=ModelConfig.get("u_dropout_prob", 0.25),
            device=self.device
        )

        self.concatenated_features_dim = ModelConfig["user_features_dim"] + ModelConfig["t_output_dim"] + ModelConfig["v_output_dim"]
        self.classification_tower = FeedForwardNetwork(
            input_dim=self.concatenated_features_dim,
            hidden_dim=ModelConfig["c_hidden_dim"],
            output_dim=ModelConfig["c_output_dim"],
            dropout_prob=ModelConfig.get("c_dropout_prob", 0.25),
            device=self.device
        )


    def forward(self, X):
        v_embeddings = self.video_embeddings(X["video_id_tokens"].to(self.device))
        time_activity = X["time_activity"].to(self.device)
        user_features = X["user_features"].to(self.device)


        video_total_embedding = self.deepsets(v_embeddings)
        video_total_embedding = self.v_ffn(video_total_embedding)


        t2v_activity = self.t2v(time_activity)
        _, (time_hidden_n, _) = self.time_activity_lstm(t2v_activity)
        time_hidden_n = time_hidden_n.permute(1, 0, 2)
        time_hidden_n = self.t_ffn(time_hidden_n)


        user_features = self.u_ffn(user_features)


        user_video_activity_features = torch.concat([user_features, video_total_embedding, time_hidden_n], dim=2)

        logits = self.classification_tower(user_video_activity_features)

        return logits
