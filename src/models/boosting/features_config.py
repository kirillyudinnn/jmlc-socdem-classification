from typing import List
from enum import Enum, auto


from boosting.boosting_consts import (
    DAY_OF_WEEK_UNIQUE,
    TIME_OF_DAY_UNIQUE, 
    UNIQUE_UA_OS,
    UNIQUE_UA_CLIENT_NAME,
    UNIQUE_CATEGORY_IDS
)

class ValueCountsMode(Enum):
    COUNT = auto()      
    NORMALIZE = auto()
    BOTH = auto()

class BoostingFeaturesConfig:
    user_agg_config = {
        "total_watchtime" : {"agg" : ["sum", "median", "std", "max", "min"]},

        "video_id" : {"agg" : ["nunique"]},

        "event_date" : {"agg" : ["nunique"]},

        "viewer_id" : {"agg" : ["size"]},

        "ua_client_name" : {
            "value_counts" : {
                "mode" : ValueCountsMode.NORMALIZE,
                "values" : UNIQUE_UA_CLIENT_NAME
            }
        },

        "ua_os" : {
            "value_counts" : {
                "mode" : ValueCountsMode.NORMALIZE,
                "values" : UNIQUE_UA_OS
            }},

        "day_of_week" : {
            "value_counts" : {
                "mode" : ValueCountsMode.NORMALIZE, 
                "values" : DAY_OF_WEEK_UNIQUE
            },
        },
        
        "time_of_day" : {
            "value_counts" : {
                "mode" : ValueCountsMode.NORMALIZE,
                "values" : TIME_OF_DAY_UNIQUE
            }, 
        },
    }

    video_agg_config = {
        "cumulative_male_ratio" : {"agg" : ["sum", "median", "std", "min", "max"]},
        
        "cumulative_female_ratio" : {"agg" : ["sum", "median", "std", "min", "max"]},

        "cumulative_under_18_ratio" : {"agg" : ["sum", "median", "std", "min", "max"]},
        
        "cumulative_18_24_ratio" : {"agg" : ["sum", "median", "std", "min", "max"]},

        "cumulative_25_34_ratio" : {"agg" : ["sum", "median", "std", "min", "max"]},

        "cumulative_35_44_ratio" : {"agg" : ["sum", "median", "std", "min", "max"]},

        "cumulative_45_54_ratio" : {"agg" : ["sum", "median", "std", "min", "max"]},

        "duration" : {"agg" : ["sum", "median", "max", "min", "std"]},

        "category_id" : {
            "value_counts" : {
                "mode" : ValueCountsMode.NORMALIZE,
                "values" : UNIQUE_CATEGORY_IDS
            }
        },
    }

    def get_user_features(self) -> List[str]:
        return list(self.user_agg_config.keys())
    
    def get_video_features(self) -> List[str]:
        return list(self.video_agg_config.keys())

    def get_unique_columns(self) -> List[str]:
        user_features = self.get_user_features()
        video_features = self.get_video_features()

        return user_features + video_features