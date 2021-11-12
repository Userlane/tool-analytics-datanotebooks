"""
- Import & Transform data from Clickhouse (with Clickhouse client library : TCP connection ->  DEV with Clickhouse port-forwarded to localhost)
- Feature Engineering to create a train dataset for clustering (individual property Id level)
- Feature Normalization & Clustering with Kmeans algorithm (scikit-learn)
- Parallelization
"""

##
## packages
##
import os 
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, Normalizer
import logging

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
#import matplotlib.pyplot as plt
from clickhouse_driver import Client
from clickhouse_driver.connection import Connection 
from multiprocessing import Pool, cpu_count

from src.clickhouse_ import *
from src.feature_engineering_ import *
from src.clustering_ import *

logging.basicConfig(level=logging.INFO, format='%(message)s')
cpu_num = int(cpu_count()/2)

##
## main function 
##
def main_parallel(propertyId):
    df_raw = get_session_data(propertyId)
    df_feature_eng = create_cluster_dataset(df_raw)

    df_cluster_centers, df_cluster_labels = cluster_centers_and_plot_Kmeans(df_feature_eng)

    result_cc = write_cluster_centers(df_cluster_centers, propertyId)
    logging.info(f"Result for property id (user_cluster_centers): {propertyId} | {result_cc}")

    result_cl = write_cluster_labels(df_cluster_labels, propertyId)
    logging.info(f"Result for property id (user_cluster_lables): {propertyId} | {result_cl}")


##
## run
##
if __name__ == "__main__":
    if check_if_exists_cluster_center_table() and check_if_exists_cluster_lables_table():
        logging.info("Process started")

        property_list = get_relevant_propertyIds()
        logging.info(f"Properties: {property_list}")
        
        with Pool(cpu_num) as p:
            p.map_async(main_parallel, property_list)




