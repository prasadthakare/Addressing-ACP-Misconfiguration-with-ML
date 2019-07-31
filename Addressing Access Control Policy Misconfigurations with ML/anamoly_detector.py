# -*- coding: utf-8 -*-
"""
Created on Fri May 31 05:49:52 2019

@author:prasad.thakare
"""
#%%
#Import function and modules.


import pandas as pd
import numpy as np
import datetime
import random

from utility.funcs import hourly_work_cal, allocate_memory, randomtimes,\
                    data_preparation, calculate_avg_usage, calculate_max_usage, calculate_quantile75_usage
                    

#%%                    
#Read csv file.
data = pd.read_csv("Labels.csv", sep=';', delimiter=';')
#input_dataframe = data[:1000].copy()
#data = data[:5000].copy()

#input_dataframe = data.copy()
data = data.copy()

##every entity, assign a memory between 100 to 1000 bytes randomly.

##sort dataframe by user_key and working hour


#######################################################################################################################################

#Change data type
data[['ENT', 'HIGH_RISK', 'IS_ASSIGNED']] = data[['ENT', 'HIGH_RISK', 'IS_ASSIGNED']].astype('category')
#encoding categories into numbers
data[['ENT', 'HIGH_RISK', 'IS_ASSIGNED']] = data[['ENT', 'HIGH_RISK', 'IS_ASSIGNED']].apply(lambda x: x.cat.codes)

########Not to be done by user############
#Creating a new "memory column"
no_of_days = 30
data = allocate_memory(data)

#%%
#Data Preparation using time of usage and taking 36% of data from whole data each time.
#no_times = 1 #for max
no_times = 3 #for avg
combined_data = data_preparation(data, no_of_days, no_times)
######### To be done###########################
#caluculating avg usage by 
£user_behav_df =  calculate_avg_usage(combined_data, no_of_days)

##caluculating max usage by 
#user_behav_df = calculate_max_usage(combined_data)
##calculating .75 quantile usage 
user_behav_df = calculate_quantile75_usage(combined_data)
user_behav_df.index = list(range(len(user_behav_df)))
###adding data usage column and timestamp.


#%%
##K-means to determine no. of clusters;
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
 #for avg
#user_behav_data = user_behav_df[["avg_resource_use","avg_data_use"]]
#for max
user_behav_data = user_behav_df[["max_resource_use","max_data_use"]] 
scaled_data = scaler.fit_transform(user_behav_data)
#X = np.array(list(zip(scaled_data.avg_resource_use, scaled_data.avg_data_use))).reshape(len(user_behav_df), 2)    
X = scaled_data
#X = np.array(list(zip(user_behav_df.resource_use, user_behav_df.data_use/200))).reshape(len(user_behav_df), 2)   

for total_clusters in [5, 10, 15, 20]:
    plt.plot()
    distortions = []
    K = range(1,total_clusters+1)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    
    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.savefig("plot {}".format(str(total_clusters)))
    plt.show()



#%%        
##Based upon graphs taking no. of clusters = 4.
num_of_clusters = 4
user_behav_df["cluster"]  = None  
kmeanModel = KMeans(n_clusters=num_of_clusters).fit(X)
kmeanModel.fit(X)  ##Check 
user_behav_df["cluster"] = kmeanModel.fit_predict(X)
#user_behav_df["cluster"] = kmeanModel.fit_predict(list(zip(user_behav_df.avg_resource_use, user_behav_df.avg_data_use)))



#%%
##Creating threshold of each cluster.
cluster_thresholds_df = pd.DataFrame(columns = ["cluster", "max_resource_use", "max_data_use"]) 
for clustr in range(num_of_clusters):
    clustr_df = user_behav_df[user_behav_df["cluster"]==clustr]
    #for avg
    #max_resource_use = max(clustr_df.avg_resource_use)
    #max_data_use = max(clustr_df.avg_data_use)
    #for max
    max_resource_use = max(clustr_df.max_resource_use)
    max_data_use = max(clustr_df.max_data_use)
    thData_dict = {"cluster":clustr, "max_resource_use":max_resource_use, "max_data_use":max_data_use}
    cluster_thresholds_df= cluster_thresholds_df.append([thData_dict],  'sort=True')    
#set index to cluster number 
cluster_thresholds_df["clusters"] = cluster_thresholds_df["cluster"].copy()
cluster_thresholds_df.set_index("cluster", inplace = True)




#%%
#Get input dataframe
#input_dataframe[['ENT', 'HIGH_RISK', 'IS_ASSIGNED']] = input_dataframe[\
#               ['ENT', 'HIGH_RISK', 'IS_ASSIGNED']].astype('category')
##encoding categories into numbers
#input_dataframe[['ENT', 'HIGH_RISK', 'IS_ASSIGNED']] = input_dataframe[\
#               ['ENT', 'HIGH_RISK', 'IS_ASSIGNED']].apply(lambda x: x.cat.codes)

#%%
########Not to be done by user############
##Creating a new "memory column"
#input_dataframe = allocate_memory(input_dataframe)
##Data Preparation using time of usage and taking 36% of data from whole data each time.
##in_combined_data = data_preparation(input_dataframe)
#
#stime = "20-01-2019 09:00:00"
#etime = "20-01-2019 18:00:00"
#n = len(input_dataframe)
#input_dataframe["access_time"] = randomtimes(stime, etime, n, nth_day = 0)

#%%

input_dataframe = pd.read_csv("test_data.csv")
######### To be done by user###########################
#calculating avg usage by a day.

in_user_behav_df = hourly_work_cal(input_dataframe)
#in_user_behav_df =  calculate_avg_usage(in_combined_data, no_of_days = 1)
in_user_behav_df.index = list(range(len(in_user_behav_df)))
###adding data usage column and timestamp.

#%%
in_user_behav_df["anomaly"] = None
in_user_behav_df["cluster"] = None

for i,key in enumerate(in_user_behav_df["USR_KEY"]):
    in_user_behav_df.at[i, "cluster"] = user_behav_df.loc[user_behav_df["USR_KEY"] == key , "cluster"].iloc[0]
#func - function to check anomaly or not
func = lambda x, cl_th_df: (x.sum_resource_use >= cl_th_df.at[x.cluster, "max_resource_use"])\
 or (x.sum_data_use >= cl_th_df.at[x.cluster, "max_data_use"]) 
 
in_user_behav_df["anomaly"] = in_user_behav_df.apply(func, cl_th_df= cluster_thresholds_df, axis = 1)






