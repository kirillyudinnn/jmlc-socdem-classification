import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import pandas as pd

from typing import List

class UserViewsDataset(Dataset):
    def __init__(
            self, 
            events: pd.DataFrame,
            target: pd.Series, 
            user_features: pd.DataFrame,
            viewer_col: str = "viewer_id",
            video_col: str = "video_id",
            time_events_cols: List[str] = ["day_of_week", "local_hour", "week", "day_of_month"],
            video_seq_len: int = 5
    ):
        self.events = events
        self.target = target
        self.user_features = user_features
        self.viewer_col = viewer_col
        self.video_col = video_col
        self.video_seq_len = video_seq_len
        self.time_events_cols = time_events_cols

        self.grouped_events = self.events.groupby(self.viewer_col)
        self.viewer_ids = list(self.grouped_events.groups.keys())

    def __len__(self):
        return len(self.viewer_ids)
    
    def __getitem__(self, idx):
        viewer_id = self.viewer_ids[idx]
        user_events = self.grouped_events.get_group(viewer_id)
        user_events_subsample = user_events.sample(self.video_seq_len)
        user_features = self.user_features.loc[[viewer_id]].values
        user_target = self.target.loc[viewer_id]

        video_ids = user_events_subsample[self.video_col].values

        time_activity = user_events[self.time_events_cols].values

        batch = {
            'video_id_tokens': torch.tensor(video_ids),
            'user_features': torch.tensor(user_features, dtype=torch.float32),
            'time_activity' : torch.tensor(time_activity, dtype=torch.float32),
            'target' : torch.tensor(user_target)
        }

        return batch
    
    @staticmethod
    def collate_fn(batch):
        video_ids = torch.stack([item['video_id_tokens'] for item in batch])
        user_features = torch.stack([item['user_features'] for item in batch])
        target = torch.stack([item["target"] for item in batch])

        time_activity_seqs = [item["time_activity"] for item in batch]
        time_activity = pad_sequence(time_activity_seqs, batch_first=True)


        return {
            "video_id_tokens" : video_ids,
            "user_features" : user_features,
            "time_activity" : time_activity,
            "target" : target
        }
