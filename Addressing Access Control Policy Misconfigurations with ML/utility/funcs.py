
import pandas as pd
import numpy as np
import datetime
import random
##every entity, assign a memory between 100 to 1000 bytes randomly.
def allocate_memory(data):
    """Random memory allocation function
    Args:
        data - dataframe
    Return:
        data - df with DATA_USAGE column"""
    
    ent_list = data["ENT"].unique()
    rand_val_list = np.random.randint(100, 1000, size = 1000)
    ent_data_dict = {key: val for key,val in zip(ent_list, rand_val_list)}
    data["DATA_USAGE"] = data["ENT"].apply(lambda x: ent_data_dict[x])    
    return data


#Random date generator
def randomtimes(stime, etime, n, nth_day):
    """Random time generator function
    Args:
        stime - starttime with date
        etime - endtime with date
        n -  no. of random timestamps to generate
        nth_day - nth day after start date.
    Return:
        data - list of random timestamp for given time interval"""
    
    frmt = '%d-%m-%Y %H:%M:%S'
    stime = datetime.datetime.strptime(stime, frmt) + datetime.timedelta(days=nth_day)
    etime = datetime.datetime.strptime(etime, frmt) + datetime.timedelta(days=nth_day)
    td = etime - stime
    return [random.random() * td + stime for _ in range(n)]




##Data preparation 
def data_preparation(data, no_of_days, no_times):
    """Creates a 'access_time' column in dataframe.
        Args:
        data - dataframe 
        no_of_days - no. of days for whic data to create.
    Return:
        combined_df - dataframe with combined data of given no. of days"""
    #After random shuffling the data. Taking 50,000(36.06%)from 138633 each time. 
    from sklearn.model_selection import ShuffleSplit
    #Appending data of each day
    combined_df = pd.DataFrame()
    #Create 3 times random data of 30 days.
    for t in range(no_times):
        rs = ShuffleSplit(n_splits = no_of_days, test_size=.3606, random_state=None)
    #Random shuffled and sliced dataframe list.
        sliced_data = [data.iloc[rndm_data_indxs,:] for _, rndm_data_indxs in rs.split(data)]
    #Random date column addition 
        final_sliced_data = []
        for n_day, data_dff in enumerate(sliced_data):
            stime = "20-01-2019 09:00:00"
            etime = "20-01-2019 18:00:00"
            nth_day = n_day + 1
            n = len(data_dff)
            data_dff["access_time"] = randomtimes(stime, etime, n, nth_day)
            final_sliced_data.append(data_dff)
            

        for data_df in final_sliced_data:
            combined_df = combined_df.append(data_df, ignore_index = True)
        
    return combined_df
        



def calculate_avg_usage(combined_df, no_of_days): 
    """ Creates dataframe having avg of no. of access and data used for each id of 30x9
    Args:
        combined_df - dataframe with date and data columns.
        no_of_days - no. of days for which data has been created.
    Return:
        user_behav_df - dataframe having avg of resource and data used for no_of_days*9"""
        
    combined_df = combined_df.copy()
    ##Averaging hourly usage of data by each user 
    #Getting hours of each time.
    combined_df["access_hour"] = pd.to_datetime(combined_df["access_time"]).dt.hour
    #Creating x df having no. of access and sum of data usage by grouping with "USR_KEY" 
    #and dividing by no_of_days*9.
    combined_ser = combined_df.groupby(["USR_KEY"]).apply(lambda x:[len(x["ENT"])\
                                      /(no_of_days*9),(x["DATA_USAGE"].sum())/(no_of_days*9)])
    user_behav_df = combined_ser.to_frame().reset_index()
    #converting series of list into two columns
    user_behav_df[["avg_resource_use", "avg_data_use"]] = pd.DataFrame(\
                 user_behav_df[0].tolist(), index= user_behav_df.index)
    return user_behav_df



    
def calculate_max_usage(combined_df):
    """Creates dataframe having max of no. of access and data used for each id for 30x9
    Args:
        combined_df - dataframe with date and data columns.
    Return:
        user_behav_df - dataframe having max of resource and data used for no_of_days*9"""
    
    combined_df = combined_df.copy() 
    #Getting hours of each time.
    combined_df["access_hour"] = pd.to_datetime(combined_df["access_time"]).dt.hour
    combined_df =  combined_df.sort_values(by = ["USR_KEY", "access_hour"])
    ##Now using hours calculation will be made
    combined_df["access_date"] = pd.to_datetime(combined_df["access_time"]).dt.date
    
    #Creating x df having no. of access and sum of data usage by grouping with "USR_KEY", "access_date", "access_hour".
    x = combined_df.groupby(["USR_KEY", "access_date", "access_hour"]).apply(\
                             lambda x: [len(x["ENT"]), x["DATA_USAGE"].sum()]).to_frame()
    x.reset_index(inplace = True)
    #converting series of list into two columns
    x[["sum_resource_use", "sum_data_use"]] = pd.DataFrame(x[0].tolist(), index= x.index)
    #taking max  of sum_resource_use and sum_data_use by grouping with USR_KEY.
    mx_df = x.groupby(["USR_KEY"]).apply(lambda x: [x.sum_resource_use.max(),\
                              x.sum_data_use.max()]).to_frame().reset_index()
    #converting series of list into two columns
    mx_df[["max_resource_use", "max_data_use"]] = pd.DataFrame(\
                 mx_df[0].tolist(), index= mx_df.index)
    user_behav_df = mx_df.copy()
    return user_behav_df
     

def calculate_quantile75_usage(combined_df):
    """Creates dataframe having max of no. of access and data used for each id for 30x9
    Args:
        combined_df - dataframe with date and data columns.
    Return:
        user_behav_df - dataframe having max of resource and data used for no_of_days*9"""
    
    combined_df = combined_df.copy() 
    #Getting hours of each time.
    combined_df["access_hour"] = pd.to_datetime(combined_df["access_time"]).dt.hour
    combined_df =  combined_df.sort_values(by = ["USR_KEY", "access_hour"])
    ##Now using hours calculation will be made
    combined_df["access_date"] = pd.to_datetime(combined_df["access_time"]).dt.date
    
    #Creating x df having no. of access and sum of data usage by grouping with "USR_KEY", "access_date", "access_hour".
    x = combined_df.groupby(["USR_KEY", "access_date", "access_hour"]).apply(\
                             lambda x: [len(x["ENT"]), x["DATA_USAGE"].sum()]).to_frame()
    x.reset_index(inplace = True)
    #converting series of list into two columns
    x[["sum_resource_use", "sum_data_use"]] = pd.DataFrame(x[0].tolist(), index= x.index)
    #taking max  of sum_resource_use and sum_data_use by grouping with USR_KEY.
    mx_df = x.groupby(["USR_KEY"]).apply(lambda x: [x.sum_resource_use.quantile(0.75),\
                              x.sum_data_use.quantile(0.75)]).to_frame().reset_index()
    #converting series of list into two columns
    mx_df[["max_resource_use", "max_data_use"]] = pd.DataFrame(\
                 mx_df[0].tolist(), index= mx_df.index)
    user_behav_df = mx_df.copy()
    return user_behav_df
     



def hourly_work_cal(input_dataframe):
    """Calculates resource access and sum of data usage on hour basis
    Args:
        input_dataframe - dataframe with access_time, DATA_USAGE and ENT
    Return:
        in_user_behav_df - dataframe with hourly calculation for each id and hour for a day."""    
    
    input_dataframe = input_dataframe.copy()
    
    input_dataframe["access_hour"] = pd.to_datetime(input_dataframe["access_time"]).dt.hour
    all_hours = sorted(list(input_dataframe["access_hour"].unique()))  
    #Creating x df having no. of access and sum of data usage by grouping with "USR_KEY", "access_hour".
    x = input_dataframe.groupby(["USR_KEY", "access_hour"]).apply(\
                             lambda x: [len(x["ENT"]), x["DATA_USAGE"].sum()]).to_frame()
    x.reset_index(inplace = True)
    #converting series of list into two columns
    x[["sum_resource_use", "sum_data_use"]] = pd.DataFrame(x[0].tolist(), index= x.index)
    in_user_behav_df = x.copy()

    return in_user_behav_df    

