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
import matplotlib.pyplot as plt
from clickhouse_driver import Client
from clickhouse_driver.connection import Connection

from src.clickhouse_ import *
from src.feature_engineering_ import *
from src.clustering_ import *

property_list = get_relevant_propertyIds()
for propertyId in property_list:
  df_raw = get_session_data(propertyId)
  df_fe = create_cluster_dataset(df_raw)

  cluster_centers_and_plot_Kmeans(df_fe, 5)





# client = Client(host = "localhost",
#                 user = "admin",
#                 password = "adUslnPass245",
#                 port = "9000",
#                 database = "UnderlyingAppAnalytics"
#                 )

# # time component
# curr_date = datetime.now().date()
# month_timedelta = timedelta(30)
# date_threshold = curr_date - month_timedelta

# # propertyId
# propertyId = '31610'


# query_get_session_events = f"""
# SELECT 
#   propertyId
# , foreignUserId
# , appSessionId
# , action
# , timestamp
# FROM 
# sessions_leaf_base
# WHERE propertyId = '{propertyId}'
# AND timestamp >= '{date_threshold}'
# """

# df_session_events = pd.DataFrame(
#     client.execute(query_get_session_events)
# ).rename(columns = {
#     0:"propertyId", 1:"foreignUserId", 2:"appSessionId", 3:"action", 4:"timestamp"
# })

# df_tmp = df_session_events.groupby(["appSessionId"])["timestamp"].agg([("sessionTime", lambda value: (np.max(value) - np.min(value)).seconds)])
# df_session_events = pd.merge(df_session_events, df_tmp, how = "left", on = "appSessionId")
# #df_session_events = df_session_events[df_session_events["sessionTime"] > 0]

# df_session_count_total = df_session_events.loc[:,["propertyId", "foreignUserId", "appSessionId"]].drop_duplicates().groupby(["propertyId", "foreignUserId"]).count().reset_index().rename(
#   columns={"appSessionId": "sessionCount"}
# )

# df_session_events_count = df_session_events.loc[:,["propertyId", "foreignUserId", "action", "appSessionId"]].groupby(["propertyId", "foreignUserId", "action"]).count().reset_index().rename(
#   columns={"appSessionId": "countSum"}
# )

# df_session_events_count = pd.merge(df_session_events_count, df_session_count_total, how = "left", on = ["propertyId", "foreignUserId"])
# df_session_events_count["countMean"] = df_session_events_count["countSum"] / df_session_events_count["sessionCount"]

# df_session_events = df_session_events.loc[:,["propertyId", "foreignUserId", "sessionTime"]].groupby(["propertyId", "foreignUserId"]).agg(sessionTime = ("sessionTime", np.mean)).reset_index()
# df_session_events_count = pd.pivot_table(df_session_events_count, values="countMean", index = "foreignUserId", columns = "action").reset_index().fillna(0)
# df_session_events = pd.merge(df_session_events_count, df_session_events.loc[:,["foreignUserId", "sessionTime"]].drop_duplicates(), how = "left", on = "foreignUserId")

# important_events = ["changeUrl", "click", "rightClick"]
# def rowsum(row, important_events = important_events):
#   action_sum = 0
#   for important_event in important_events:
#     action_sum += row[important_event]
#   return action_sum

# df_session_events["importantEventCount"] = df_session_events.apply(lambda row: rowsum(row, important_events), axis = 1)
# df_num = df_session_events.loc[:,["sessionTime", "importantEventCount"]]

# def cluster_centers_and_plot_Kmeans(df, clu_num, scale = True, normalize = "l2"):
#     """
#     Receive Numeric Dataset -> Kmeans Clustering -> Cluster Center Results
#     """

#     if scale:
#         scalar = StandardScaler()
#         scalar.fit(df_num)
#         df_scale = pd.DataFrame(scalar.transform(df_num))

#         kmeans = KMeans(init="random", n_clusters=5, n_init=4, random_state=0)
#         kmeans.fit(df_scale)

#         sessionTimeMean = np.mean(df_num.sessionTime)
#         sessionTimeStd = np.std(df_num.sessionTime)
#         importantEventCountMean = np.mean(df_num.importantEventCount)
#         importantEventCountStd = np.std(df_num.importantEventCount)

#         sessionTime = (df_scale[0] * sessionTimeStd + sessionTimeMean)
#         importantEventCount = (df_scale[1] * importantEventCountStd + importantEventCountMean)
        
#         cluster_centers = pd.DataFrame({
#         "SessionTimeCenter": kmeans.cluster_centers_[:,0] * sessionTimeStd + sessionTimeMean,
#         "EventCountCenter" : kmeans.cluster_centers_[:,1] * importantEventCountStd + importantEventCountMean
#         })

#     else:
#         normalizer = Normalizer(norm = normalize)
#         normalizer.fit(df_num.to_numpy().transpose())
#         df_norm = pd.DataFrame(normalizer.transform(df_num.to_numpy().transpose()).transpose())
    
#         if normalize == "l2":
#             sessionTimeNorm = np.sqrt(np.sum(abs(df_num.sessionTime)))
#             importantEventCountNorm = np.sqrt(np.sum(abs(df_num.importantEventCount)))
            
#             sessionTime = (df_norm[0] * sessionTimeNorm)
#             importantEventCount = (df_norm[1] * importantEventCountNorm)
        
#         elif normalize == "l1":
#             sessionTimeNorm = np.linalg.norm(df_num.sessionTime, ord = 1)
#             importantEventCountNorm = np.linalg.norm(df_num.importantEventCount, ord = 1)
            
#             sessionTime = (df_norm[0] * sessionTimeNorm)
#             importantEventCount = (df_norm[1] * importantEventCountNorm)
        
#         elif normalize == "max":
#             sessionTimeNorm = np.linalg.norm(df_num.sessionTime, ord = "inf")
#             importantEventCountNorm = np.linalg.norm(df_num.importantEventCount, ord = "inf")
            
#             sessionTime = (df_norm[0] * sessionTimeNorm)
#             importantEventCount = (df_norm[1] * importantEventCountNorm)
#         else:
#             raise "Norm not defined!"
        
#         kmeans = KMeans(init="random", n_clusters=clu_num, n_init=4, random_state=0)
#         kmeans.fit(df_norm)

#         cluster_centers = pd.DataFrame({
#         "SessionTimeCenter": kmeans.cluster_centers_[:,0] * sessionTimeNorm,
#         "EventCountCenter" : kmeans.cluster_centers_[:,1] * importantEventCountNorm
#         })

        
#     df_final = pd.DataFrame({
#     "SessionTime": sessionTime,
#     "EventCount": importantEventCount,
#     "ClusterLabel": kmeans.labels_
#     })
    


#     # plot
#     plt.scatter(
#       df_final["SessionTime"], 
#       df_final["EventCount"], 
#       c = df_final["ClusterLabel"],
#            )
#     plt.scatter(
#       cluster_centers["SessionTimeCenter"],
#       cluster_centers["EventCountCenter"],
#       c = "red"
#     )

#         # logic to create lines between clusters
#     #cluster_centers = cluster_centers.sort_values(["EventCountCenter"])
#     # session_time_centers = np.array(cluster_centers.SessionTimeCenter)
#     # event_count_centers = np.array(cluster_centers.EventCountCenter)

#     # for j in range(4):
#     #   p1 = session_time_centers[j]
#     #   q1 = event_count_centers[j]
#     #   p2 = session_time_centers[j + 1]
#     #   q2 = event_count_centers[j + 1] 

#     #   k = (q2 - q1) / (p2 - p1)
#     #   #k = -1/k
#     #   x = np.linspace(0,10000, 10000) 
#     #   y = k*x + p1 - q1
#     #   print(x)
#     #   print(y)
#     #   x_new = []
#     #   y_new = []
#     #   for i,j in zip(x,y):
#     #     if j >= 0 and j < 500:
#     #       x_new.append(i)
#     #       y_new.append(j) 
#     #   plt.plot(
#     #         x_new,
#     #         y_new,
#     #         '-',
#     #         c = "red"
#     #       )

#     plt.xlabel("SessionTime")
#     plt.ylabel("EventCount")
#     plt.show()
    
#     return cluster_centers


# cluster_centers = cluster_centers_and_plot_Kmeans(df_num, 5)
# print(cluster_centers)