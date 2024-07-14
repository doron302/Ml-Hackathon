import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression

import plots
from model import *


def preproccess_train(train_df):

    # Drop rows where passengers_up is null
    cleaned_df = train_df.dropna(subset=['passengers_up'])

    # Step 3: Drop the station_name column
    cleaned_df = cleaned_df.drop(columns=['station_name'])

    # Step 4: Create one-hot vectors from the column alternatives
    cleaned_df = pd.get_dummies(cleaned_df, columns=['alternative'])

    # Step 5: Change the cluster column to numerical values
    cleaned_df['cluster'] = cleaned_df['cluster'].astype('category').cat.codes

    cleaned_df['part'] = cleaned_df['part'].astype('category').cat.codes

    cleaned_df["trip_id_unique"] =\
        cleaned_df["trip_id"].astype(str) + cleaned_df["part"].astype(str)

    cleaned_df["trip_id_unique"] = cleaned_df["trip_id_unique"].astype('int64')

    cleaned_df['trip_id_unique_station'] = \
        cleaned_df['trip_id_unique'].astype(str) + cleaned_df['station_index'].astype(str)

    cleaned_df['trip_id_unique_station'] = cleaned_df['trip_id_unique_station'].astype('int64')

    # cleaned_df['arrival_time'] = pd.to_datetime(cleaned_df['arrival_time'], format='%H:%M:%S')
    #
    # # Extract hour, minute, and second, convert to numeric value (e.g., HHMMSS format)
    # cleaned_df['arrival_time'] = (
    #         cleaned_df['arrival_time'].dt.hour * 3600 + cleaned_df['arrival_time'].dt.minute
    #         * 60 + cleaned_df['arrival_time'].dt.second)
    #
    # cleaned_df['door_closing_time'] = pd.to_datetime(cleaned_df['door_closing_time'], format='%H:%M:%S')
    #
    # # Extract hour, minute, and second, convert to numeric value (e.g., HHMMSS format)
    # cleaned_df['door_closing_time'] = (
    #         cleaned_df['door_closing_time'].dt.hour * 3600 + cleaned_df['door_closing_time'].dt.minute
    #         * 60 + cleaned_df['door_closing_time'].dt.second)
    #
    # cleaned_df['door_opening_duration'] = (cleaned_df['door_closing_time'] - cleaned_df['arrival_time']).fillna(0)
    #
    # cleaned_df = cleaned_df.drop(columns=['door_closing_time'])



    # Display the cleaned DataFramedsf
    return cleaned_df



def preprocess_test(test_df):

    cleaned_df = test_df.drop(columns=['station_name'])

    cleaned_df = pd.get_dummies(cleaned_df, columns=['alternative'])

    cleaned_df['cluster'] = cleaned_df['cluster'].astype('category').cat.codes

    cleaned_df['part'] = cleaned_df['part'].astype('category').cat.codes

    cleaned_df["trip_id_unique"] = \
        cleaned_df["trip_id"].astype(str) + cleaned_df["part"].astype(str)

    cleaned_df["trip_id_unique"] = cleaned_df["trip_id_unique"].astype('int64')

    cleaned_df['trip_id_unique_station'] = \
        cleaned_df['trip_id_unique'].astype(str) + cleaned_df['station_index'].astype(str)

    cleaned_df['trip_id_unique_station'] = cleaned_df['trip_id_unique_station'].astype('int64')

    # cleaned_df['arrival_time'] = pd.to_datetime(cleaned_df['arrival_time'], format='%H:%M:%S')
    #
    # # Extract hour, minute, and second, convert to numeric value (e.g., HHMMSS format)
    # cleaned_df['arrival_time'] = (
    #         cleaned_df['arrival_time'].dt.hour * 3600 + cleaned_df['arrival_time'].dt.minute
    #         * 60 + cleaned_df['arrival_time'].dt.second)
    #
    # cleaned_df['door_closing_time'] = pd.to_datetime(cleaned_df['door_closing_time'], format='%H:%M:%S')
    #
    # # Extract hour, minute, and second, convert to numeric value (e.g., HHMMSS format)
    # cleaned_df['door_closing_time'] = (
    #         cleaned_df['door_closing_time'].dt.hour * 3600 + cleaned_df['door_closing_time'].dt.minute
    #         * 60 + cleaned_df['door_closing_time'].dt.second)
    #
    # cleaned_df['door_opening_duration'] = (cleaned_df['door_closing_time'] - cleaned_df['arrival_time']).fillna(0)
    #
    # cleaned_df = cleaned_df.drop(columns=['door_closing_time'])

    return cleaned_df

def pearson_coor(cleaned_df, column_name):
    # Select only numeric columns
    numeric_data = cleaned_df.select_dtypes(include=['float64', 'int64', 'bool'])

    # Calculate the Pearson correlation for each numeric column with 'passengers_up'
    correlation = numeric_data.corr()[column_name]

    # Print the correlation values
    print(correlation)
    
def preprocess_and_aggregate(cleaned_df):
    cleaned_df = trip_duration_preprocess(cleaned_df)


    # Aggregate data by trip_id_unique
    agg_df = cleaned_df.groupby('trip_id_unique').agg({
        'line_id': 'first',
        'direction': 'first',
        'cluster': 'first',
        'latitude': 'mean',
        'longitude': 'mean',
        'passengers_continue': 'mean',
        'mekadem_nipuach_luz': 'first',
        'passengers_continue_menupach': 'mean',
        'station_index': 'max',
        # 'door_opening_duration': 'sum',
        'trip_duration_in_minutes': 'first'
    }).reset_index()
    
    agg_df.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in agg_df.columns]

   #  agg_df = agg_df.drop(columns=['arrival_time_max'])

    agg_df.to_csv('train_set/agg_train.csv', index=False)
    
    return agg_df



def plots_on_raw_data(train_bus_df):
    # plots.region_vs_passengers_up(train_bus_df)
    # plots.heatmap_correlation(train_bus_df)
    pass

def plots_on_cleaned_data(cleaned_df):
    # plots.heatmap_correlation(cleaned_df)
    # plots.distance_vs_passengers_up(cleaned_df)
    pass

def trip_duration_preprocess(data):
    # Ensure 'arrival_time' is in datetime format
    data['arrival_time'] = pd.to_datetime(data['arrival_time'], format='%H:%M:%S')

    # Calculate the first arrival time for each trip (identified by 'trip_id' and 'line_id')
    first_arrivals = data[data['station_index'] == 1].copy()
    first_arrivals.rename(columns={'arrival_time': 'first_arrival_time'}, inplace=True)
    first_arrivals = first_arrivals[['trip_id', 'line_id', 'first_arrival_time']]

    # Identify the last station for each trip
    last_arrivals = data.groupby(['trip_id', 'line_id']).apply(
        lambda x: x.loc[x['station_index'].idxmax()]).reset_index(drop=True)
    last_arrivals.rename(columns={'arrival_time': 'last_arrival_time'}, inplace=True)
    last_arrivals = last_arrivals[['trip_id', 'line_id', 'last_arrival_time']]

    # Merge the first and last arrivals to calculate the trip duration
    merged_data = pd.merge(first_arrivals, last_arrivals, on=['trip_id', 'line_id'])

    # Calculate trip duration in minutes
    merged_data['trip_duration_in_minutes'] = (merged_data['last_arrival_time'] - merged_data[
        'first_arrival_time']).dt.total_seconds() / 60.0

    # Adjust for trips that span across midnight
    merged_data['trip_duration_in_minutes'] = merged_data['trip_duration_in_minutes'].apply(
        lambda x: x if x >= 0 else x + 24 * 60)

    # Merge the trip duration back into the original data
    data = pd.merge(data, merged_data[['trip_id', 'line_id', 'trip_duration_in_minutes']], on=['trip_id', 'line_id'],
                    how='left')

    data.to_csv('kfdv.csv')
    return data

def task_1():
    #
    train_bus_df = pd.read_csv('train_set/train_bus_schedule.csv', encoding="ISO-8859-8")

    plots_on_raw_data(train_bus_df)

    clean_df = preproccess_train(train_bus_df)

    plots_on_cleaned_data(clean_df)

    train_data, temp_data = train_test_split(clean_df, test_size=0.3, random_state=42)

    y_train = train_data['passengers_up']

    X_train = train_data.drop(columns=['passengers_up'])

    dev_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    y_dev = dev_data['passengers_up']
    X_dev = dev_data.drop(columns=['passengers_up'])
    
    y_test = test_data['passengers_up']
    X_test = test_data.drop(columns=['passengers_up'])   
        
        
    best_model = find_best_XGBoost(X_train, y_train,X_test,y_test,X_dev,y_dev)

    X_test = pd.read_csv('tests_set/X_passengers_up.csv', encoding="ISO-8859-8")
    # Preprocess the test data
    X_test_pre = preprocess_test(X_test)
    X_test_pre = X_test_pre.reindex(columns=X_train.columns, fill_value=0)
    y_pred = best_model.predict(X_test_pre)
    y_pred = np.round(y_pred)
    output_df = pd.DataFrame()
    output_df['trip_id_unique_station'] = X_test['trip_id_unique_station']
    output_df['passengers_up'] = y_pred

    output_df.to_csv('passengers_up_predictions.csv', index=False)

def task_2():
    
    le = pd.read_csv("/Users/doron302/menupah/task_number/code/train_bus_schedule.csv", encoding="ISO-8859-8")

    train_bus_df = pd.read_csv('train_set/train_bus_schedule.csv', encoding="ISO-8859-8")

    clean_df = preproccess_train(train_bus_df)
    aggdf = preprocess_and_aggregate(clean_df)

    train_data, temp_data = train_test_split(aggdf, test_size=0.3, random_state=42)

    y_train = train_data['trip_duration_in_minutes']

    X_train = train_data.drop(columns=['trip_duration_in_minutes'])

    dev_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    y_dev = dev_data['trip_duration_in_minutes']

    X_dev = dev_data.drop(columns=['trip_duration_in_minutes'])

    X_fake_test = test_data.drop(columns=['trip_duration_in_minutes'])
    y_fake_test = test_data['trip_duration_in_minutes']

    model = find_best_XGBoost(X_train, y_train,X_fake_test,y_fake_test,X_dev,y_dev)
    
    
    
    


    print((mse(y_dev, model.predict(X_dev))))





if __name__ == '__main__':
    np.random.seed(42)
    #task_1()
    task_2()


