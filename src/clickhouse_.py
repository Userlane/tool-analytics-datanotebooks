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
## functions
##

def parse_clickhouse_config(clickhouse_config):
        return clickhouse_config["clickhouse_host"], clickhouse_config["clickhouse_user"], clickhouse_config["clickhouse_password"], clickhouse_config["clickhouse_port"], clickhouse_config["clickhouse_database"]

def get_relevant_propertyIds(clickhouse_config = clickhouse_config_dict, data_config = data_config_dict):
        """
        Returns a list of property Ids that have enough data to perform clustering
        """

        clickhouse_host, clickhouse_user, clickhouse_password, clickhouse_port, clickhouse_database = parse_clickhouse_config(clickhouse_config)
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
        try:
                client = Client(host = clickhouse_host,
                                user = clickhouse_user,
                                password = clickhouse_password,
                                port = clickhouse_port,
                                database = clickhouse_database,
                                )

                property_list = client.execute(get_all_valid_properties_query)
                client.disconnect()
                return [property for property, count in property_list if count > count_threshold]
        except Exception as e:
                print(e)
                return []

def get_session_data(propertyId, clickhouse_config = clickhouse_config_dict, data_config = data_config_dict):
        """
        Returns app session data for a specific property Id
        """

        clickhouse_host, clickhouse_user, clickhouse_password, clickhouse_port, clickhouse_database = parse_clickhouse_config(clickhouse_config)
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

def check_if_exists_cluster_center_table(clickhouse_config = clickhouse_config_dict):
        """
        Checks if the table: user_cluster_centers exsts in Clickhouse
        If yes -> return True, if not -> it creates it, returns True -> if it can't create it -> returns False
        """

        clickhouse_host, clickhouse_user, clickhouse_password, clickhouse_port, clickhouse_database = parse_clickhouse_config(clickhouse_config)
        
        list_tables_query = """
        SHOW TABLES
        """
        try:
                client = Client(host = clickhouse_host,
                                user = clickhouse_user,
                                password = clickhouse_password,
                                port = clickhouse_port,
                                database = clickhouse_database,
                                )
                table_list = [el[0] for el in client.execute(list_tables_query)]
        except Exception as e:
                print(e)
                return False

        if "user_cluster_centers" in table_list:
                client.disconnect()
                return True
        else:
                try:
                        create_user_cluster_centers_table_query = """
                        CREATE TABLE user_cluster_centers 
                                (
                                `dateId` Date,
                                `propertyId` String,
                                `clusterLabel` String,
                                `sessionTimeCenter` Float64,
                                `eventCountCenter` Float64
                                )
                                ENGINE = MergeTree ORDER BY dateId
                        """
                        client.execute(create_user_cluster_centers_table_query)
                        client.disconnect()
                        return True
                except Exception as e:
                        print(e)
                        return False

def check_if_exists_cluster_lables_table(clickhouse_config = clickhouse_config_dict):
        """
        Checks if the table: user_cluster_labels exsts in Clickhouse
        If yes -> return True, if not -> it creates it, returns True -> if it can't create it -> returns False
        """

        clickhouse_host, clickhouse_user, clickhouse_password, clickhouse_port, clickhouse_database = parse_clickhouse_config(clickhouse_config)
        
        list_tables_query = """
        SHOW TABLES
        """
        try:
                client = Client(host = clickhouse_host,
                                user = clickhouse_user,
                                password = clickhouse_password,
                                port = clickhouse_port,
                                database = clickhouse_database,
                                )
                table_list = [el[0] for el in client.execute(list_tables_query)]
        except Exception as e:
                print(e)
                return False
                
        if "user_cluster_labels" in table_list:
                client.disconnect()
                return True
        else:
                try:
                        create_user_cluster_centers_table_query = """
                        CREATE TABLE user_cluster_labels 
                                (
                                `dateId` Date,
                                `propertyId` String,
                                `foreignuserId` String,
                                `clusterLabel` String,
                                `sessionTimeAvg` Float64,
                                `eventCountAvg` Float64
                                )
                                ENGINE = MergeTree ORDER BY dateId
                        """
                        client.execute(create_user_cluster_centers_table_query)
                        client.disconnect()
                        return True
                except Exception as e:
                        print(e)
                        return False       

def write_cluster_centers(cluster_centers, propertyId, clickhouse_config = clickhouse_config_dict):
        """
        Write cluster centers to Clickhouse
        """

        clickhouse_host, clickhouse_user, clickhouse_password, clickhouse_port, clickhouse_database = parse_clickhouse_config(clickhouse_config)

        date_today = datetime.now().date()
        cluster_centers_list = list(cluster_centers.to_records(index = False))
        insert_query = """
        INSERT INTO user_cluster_centers 
        (dateId, propertyId, clusterLabel, sessionTimeCenter, eventCountCenter)
        VALUES
        """
        try:
                client = Client(host = clickhouse_host,
                                user = clickhouse_user,
                                password = clickhouse_password,
                                port = clickhouse_port,
                                database = clickhouse_database,
                                )
                client.execute(insert_query,
                                ((date_today, propertyId, str(clusterLabel), float(sessionTimeCenter), float(eventCountCenter)) 
                                for clusterLabel,sessionTimeCenter,eventCountCenter in cluster_centers_list)
                                )
                client.disconnect()
                return True

        except Exception as e:
                print(e)
                return False
        
def write_cluster_labels(cluster_labels, propertyId, clickhouse_config = clickhouse_config_dict):
        """
        Write cluster labels to Clickhouse
        """

        clickhouse_host, clickhouse_user, clickhouse_password, clickhouse_port, clickhouse_database = parse_clickhouse_config(clickhouse_config)

        date_today = datetime.now().date()
        cluster_label_list = list(cluster_labels.to_records(index = False))
        insert_query = """
        INSERT INTO user_cluster_labels 
        (dateId, propertyId, foreignuserId, clusterLabel, sessionTimeAvg, eventCountAvg)
        VALUES
        """
        try:
                client = Client(host = clickhouse_host,
                                user = clickhouse_user,
                                password = clickhouse_password,
                                port = clickhouse_port,
                                database = clickhouse_database,
                                )
                client.execute(insert_query,
                                ((date_today, propertyId, str(foreignUserId), str(clusterLabel), float(sessionTimeAvg), float(eventCountAvg)) 
                                for foreignUserId, clusterLabel, sessionTimeAvg, eventCountAvg in cluster_label_list)
                                )
                client.disconnect()
                return True

        except Exception as e:
                print(e)
                return False
        