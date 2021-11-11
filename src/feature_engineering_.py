"""
Feature Engineering Functions
"""

##
## packages
##
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json

##
## config
##
with open("config/data_config.json", "r") as f_json:
        data_config_dict = json.load(f_json)

important_events = data_config_dict["important_events"]

##
## functions
##
def rowsum(row, important_events = important_events):
    """
    Sum of important events by row (used as a lambda function)
    """
    action_sum = 0
    for important_event in important_events:
        action_sum += row[important_event]
    return action_sum

def create_cluster_dataset(df):
    """
    Receives: raw dataset (pandas DataFrame) from Clickhouse `sessions_leaf_base` table
    Returns: clustering ready dataset (pandas DataFrame) 
    """
    df_tmp = df.groupby(["appSessionId"])["timestamp"].agg([("sessionTime", lambda value: (np.max(value) - np.min(value)).seconds)])
    df = pd.merge(df, df_tmp, how = "left", on = "appSessionId")
    df = df[df["sessionTime"] > 0]

    df_count_total = df.loc[:,["propertyId", "foreignUserId", "appSessionId"]].drop_duplicates().groupby(["propertyId", "foreignUserId"]).count().reset_index().rename(
    columns={"appSessionId": "sessionCount"}
    )
    df_count = df.loc[:,["propertyId", "foreignUserId", "action", "appSessionId"]].groupby(["propertyId", "foreignUserId", "action"]).count().reset_index().rename(
    columns={"appSessionId": "countSum"}
    )

    df_count = pd.merge(df_count, df_count_total, how = "left", on = ["propertyId", "foreignUserId"])
    df_count["countMean"] = df_count["countSum"] / df_count["sessionCount"]
    df = df.loc[:,["propertyId", "foreignUserId", "sessionTime"]].groupby(["propertyId", "foreignUserId"]).agg(sessionTime = ("sessionTime", np.mean)).reset_index()
    df_count = pd.pivot_table(df_count, values="countMean", index = "foreignUserId", columns = "action").reset_index().fillna(0)
    df = pd.merge(df_count, df.loc[:,["foreignUserId", "sessionTime"]].drop_duplicates(), how = "left", on = "foreignUserId")
    df["importantEventCount"] = df.apply(lambda row: rowsum(row, important_events), axis = 1)
    return df

