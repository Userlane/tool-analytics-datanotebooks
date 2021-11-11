"""
Clustering Functions
"""

# packages
import os 
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, Normalizer

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
#import matplotlib.pyplot as plt

##
## functions
##
def cluster_centers_and_plot_Kmeans(df, clu_num, scale = True, normalize = "l2"):
    """
    Receive Dataset -> Kmeans Clustering -> Cluster Center Results + Cluster Labels
    """
    df_num = df.loc[:,["sessionTime", "importantEventCount"]]

    if scale:
        scalar = StandardScaler()
        scalar.fit(df_num)
        df_scale = pd.DataFrame(scalar.transform(df_num))

        sessionTimeMean = np.mean(df_num.sessionTime)
        sessionTimeStd = np.std(df_num.sessionTime)
        importantEventCountMean = np.mean(df_num.importantEventCount)
        importantEventCountStd = np.std(df_num.importantEventCount)

        sessionTime = (df_scale[0] * sessionTimeStd + sessionTimeMean)
        importantEventCount = (df_scale[1] * importantEventCountStd + importantEventCountMean)

        kmeans = KMeans(init="random", n_clusters=5, n_init=4, random_state=0)
        kmeans.fit(df_scale)

        cluster_centers = pd.DataFrame({
            "ClusterLabel": range(0,5),
            "SessionTimeCenter": kmeans.cluster_centers_[:,0] * sessionTimeStd + sessionTimeMean,
            "EventCountCenter" : kmeans.cluster_centers_[:,1] * importantEventCountStd + importantEventCountMean
        })

    else:
        normalizer = Normalizer(norm = normalize)
        normalizer.fit(df_num.to_numpy().transpose())
        df_norm = pd.DataFrame(normalizer.transform(df_num.to_numpy().transpose()).transpose())
    
        if normalize == "l2":
            sessionTimeNorm = np.sqrt(np.sum(abs(df_num.sessionTime)))
            importantEventCountNorm = np.sqrt(np.sum(abs(df_num.importantEventCount)))
            
            sessionTime = (df_norm[0] * sessionTimeNorm)
            importantEventCount = (df_norm[1] * importantEventCountNorm)
        
        elif normalize == "l1":
            sessionTimeNorm = np.linalg.norm(df_num.sessionTime, ord = 1)
            importantEventCountNorm = np.linalg.norm(df_num.importantEventCount, ord = 1)
            
            sessionTime = (df_norm[0] * sessionTimeNorm)
            importantEventCount = (df_norm[1] * importantEventCountNorm)
        
        elif normalize == "max":
            sessionTimeNorm = np.linalg.norm(df_num.sessionTime, ord = "inf")
            importantEventCountNorm = np.linalg.norm(df_num.importantEventCount, ord = "inf")
            
            sessionTime = (df_norm[0] * sessionTimeNorm)
            importantEventCount = (df_norm[1] * importantEventCountNorm)
        else:
            raise "Norm not defined!"
        
        kmeans = KMeans(init="random", n_clusters=clu_num, n_init=4, random_state=0)
        kmeans.fit(df_norm)

        cluster_centers = pd.DataFrame({
            "ClusterLabel": range(0,5),
            "SessionTimeCenter": kmeans.cluster_centers_[:,0] * sessionTimeNorm,
            "EventCountCenter" : kmeans.cluster_centers_[:,1] * importantEventCountNorm
        })

    df_user_labels = pd.DataFrame({
        "ForeignUserId": df.foreignUserId,
        "ClusterLabel": kmeans.labels_,
        "SessionTime": sessionTime,
        "EventCount": importantEventCount
        
    })

    # plot
    # plt.scatter(
    #   df_user_labels["SessionTime"], 
    #   df_user_labels["EventCount"], 
    #   c = df_user_labels["ClusterLabel"],
    # )

    # plt.scatter(
    #   cluster_centers["SessionTimeCenter"],
    #   cluster_centers["EventCountCenter"],
    #   c = "red"
    # )

        # logic to create lines between clusters
    #cluster_centers = cluster_centers.sort_values(["EventCountCenter"])
    # session_time_centers = np.array(cluster_centers.SessionTimeCenter)
    # event_count_centers = np.array(cluster_centers.EventCountCenter)

    # for j in range(4):
    #   p1 = session_time_centers[j]
    #   q1 = event_count_centers[j]
    #   p2 = session_time_centers[j + 1]
    #   q2 = event_count_centers[j + 1] 

    #   k = (q2 - q1) / (p2 - p1)
    #   #k = -1/k
    #   x = np.linspace(0,10000, 10000) 
    #   y = k*x + p1 - q1
    #   print(x)
    #   print(y)
    #   x_new = []
    #   y_new = []
    #   for i,j in zip(x,y):
    #     if j >= 0 and j < 500:
    #       x_new.append(i)
    #       y_new.append(j) 
    #   plt.plot(
    #         x_new,
    #         y_new,
    #         '-',
    #         c = "red"
    #       )

    # plt.xlabel("SessionTime")
    # plt.ylabel("EventCount")
    # plt.show()
    
    return cluster_centers, df_user_labels