"""
Clickhouse Functions Config & Queries. 
"""

##
## packages
##
import os 
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from clickhouse_driver import Client
from clickhouse_driver.connection import Connection

##
## configs
##
with open("config/clickhouse_config.json", "r") as f_json:
        clickhouse_config_dict = json.load(f_json)

with open("config/data_config.json", "r") as f_json:
        data_config_dict = json.load(f_json)

##
## clickhouse query functions
##
def get_relevant_propertyIds(clickhouse_config = clickhouse_config_dict, data_config = data_config_dict):
        """
        Returns a list of property Ids that have enough data to perform clustering
        """

        clickhouse_host, clickhouse_user, clickhouse_password = clickhouse_config["clickhouse_host"], clickhouse_config["clickhouse_user"], clickhouse_config["clickhouse_password"]
        clickhouse_port, clickhouse_database = clickhouse_config["clickhouse_port"], clickhouse_config["clickhouse_database"]
        days_prev, count_threshold = data_config["days"], data_config["count"]

        curr_date = datetime.now().date()
        param_timedelta = timedelta(days_prev)
        param_date_threshold = curr_date - param_timedelta

        get_all_valid_properties_query = f"""
        SELECT propertyId, count(DISTINCT appSessionId)
        FROM 
        sessions_leaf_base
        WHERE timestamp >= '{param_date_threshold}'
        GROUP BY propertyId
        """

        client = Client(host = clickhouse_host,
                        user = clickhouse_user,
                        password = clickhouse_password,
                        port = clickhouse_port,
                        database = clickhouse_database,
                        )

        property_list = client.execute(get_all_valid_properties_query)
        client.disconnect()
        return [property for property, count in property_list if count > count_threshold]

def get_session_data(propertyId, clickhouse_config = clickhouse_config_dict, data_config = data_config_dict):
        """
        Returns app session data for a specific property Id
        """

        clickhouse_host, clickhouse_user, clickhouse_password = clickhouse_config["clickhouse_host"], clickhouse_config["clickhouse_user"], clickhouse_config["clickhouse_password"]
        clickhouse_port, clickhouse_database = clickhouse_config["clickhouse_port"], clickhouse_config["clickhouse_database"]
        days_prev = data_config["days"]

        curr_date = datetime.now().date()
        param_timedelta = timedelta(days_prev)
        param_date_threshold = curr_date - param_timedelta

        query_get_session_events = f"""
        SELECT 
        propertyId
        , foreignUserId
        , appSessionId
        , action
        , timestamp
        FROM 
        sessions_leaf_base
        WHERE propertyId = '{propertyId}'
        AND timestamp >= '{param_date_threshold}'
        """

        client = Client(host = clickhouse_host,
                        user = clickhouse_user,
                        password = clickhouse_password,
                        port = clickhouse_port,
                        database = clickhouse_database,
                        )

        df_session_events = pd.DataFrame(
                        client.execute(query_get_session_events)
                ).rename(columns = {
                        0:"propertyId", 1:"foreignUserId", 2:"appSessionId", 3:"action", 4:"timestamp"
                })
        client.disconnect()
        return df_session_events
