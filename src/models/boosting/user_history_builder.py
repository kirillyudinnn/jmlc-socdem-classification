import pandas as pd
from typing import Dict, List, Union, Any

from enum import Enum, auto
from boosting.features_config import BoostingFeaturesConfig

class ValueCountsMode(Enum):
    COUNT = auto()      
    NORMALIZE = auto()
    BOTH = auto()


class UserHistoryFeatureGenerator:
    def __init__(self, features_config: BoostingFeaturesConfig):
        self.user_features_ = features_config.get_user_features()
        self.video_features_ = features_config.get_video_features()

        if len(self.user_features_) != 0:
            self.user_agg_config = features_config.user_agg_config
        
        if len(self.video_features_) != 0:
            self.video_agg_config = features_config.video_agg_config


    def fit_transform(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Transform events to users features DataFrame 

        Args:
            events (pd.DataFrame): DataFrame with users views 

        Returns:
            pd.DataFrame: user aggregated features
        """

        grouped_events = events.groupby("viewer_id")

        if len(self.user_features_) != 0:
            user_features = self._calculate_statistics(grouped_events, self.user_agg_config)

        if len(self.video_features_) != 0:
            video_features = self._calculate_statistics(grouped_events, self.video_agg_config)


        result = user_features.join(video_features, on='viewer_id')

        return result
        

    def _calculate_statistics(
            self,
            grouped_events,
            agg_config: Dict[str, Dict[str, Union[Dict[str, Any], List[str]]]]
    ) -> pd.DataFrame: 
        
        list_of_feat_statistics = []
        
        for feature_name in agg_config.keys():
            feature_desc = agg_config[feature_name]
            agg_functions = feature_desc.get("agg", None)
            value_counts = feature_desc.get("value_counts", None)

            if agg_functions is not None:
                feature_agg_statistics = grouped_events[feature_name].agg(agg_functions)
                feature_agg_statistics.rename(
                    columns={
                        agg_name : feature_name + "_" + agg_name for agg_name in agg_functions
                    },
                    inplace=True
                )
                feature_agg_statistics.fillna(0.0, inplace=True)
                
                list_of_feat_statistics.append(feature_agg_statistics)

            if value_counts is not None:
                normalize = value_counts["mode"]
                values = value_counts["values"]

                feature_vc_statistics = grouped_events[feature_name].value_counts()

                if normalize == ValueCountsMode.NORMALIZE:
                    feature_vc_statistics = feature_vc_statistics / grouped_events.size()
                
                elif normalize == ValueCountsMode.BOTH:
                    raise NotImplementedError

                feature_vc_statistics = feature_vc_statistics.unstack().fillna(0.0)
                feature_vc_statistics = feature_vc_statistics.reindex(columns=values, fill_value=0.0).rename_axis(None, axis=1)
                feature_vc_statistics.rename(
                    columns={
                        value : feature_name + "_" + value for value in values
                    },
                    inplace=True
                )

                list_of_feat_statistics.append(feature_vc_statistics)


        viewer_id_feature_stats = pd.concat(list_of_feat_statistics, axis=1)

        return viewer_id_feature_stats
