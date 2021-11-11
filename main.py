"""
- Import & Transform data from Clickhouse (with Clickhouse client library : TCP connection ->  DEV with Clickhouse port-forwarded to localhost)
- Feature Engineering to create a train dataset for clustering (individual property Id level)
- Feature Normalization & Clustering with Kmeans algorithm (scikit-learn)
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

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
#import matplotlib.pyplot as plt
from clickhouse_driver import Client
from clickhouse_driver.connection import Connection 

from src.clickhouse_ import *
from src.feature_engineering_ import *
from src.clustering_ import *

property_list = get_relevant_propertyIds()
print(check_if_exists_cluster_center_table())
print(check_if_exists_cluster_lables_table())

for propertyId in property_list:
    df_raw = get_session_data(propertyId)
    df_fe = create_cluster_dataset(df_raw)

    df_cc, df_cl = cluster_centers_and_plot_Kmeans(df_fe, 5)
    print(write_cluster_centers(df_cc, propertyId))
    print(write_cluster_labels(df_cl, propertyId))

