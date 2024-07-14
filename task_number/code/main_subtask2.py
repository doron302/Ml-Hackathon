from argparse import ArgumentParser
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from hackathon_code.model import find_best_XGBoost

"""
usage:
    python code/main.py --training_set PATH --test_set PATH --out PATH

for example:
    python code/main.py --training_set /cs/usr/gililior/training.csv --test_set /cs/usr/gililior/test.csv --out predictions/trip_duration_predictions.csv 

"""

# implement here your load,preprocess,train,predict,save functions (or any other design you choose)

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

    return cleaned_df


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
        'trip_duration_in_minutes': 'first'
    }).reset_index()

    agg_df.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in agg_df.columns]

    #  agg_df = agg_df.drop(columns=['arrival_time_max'])


    return agg_df


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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True,
                        help="path to the training set")
    parser.add_argument('--test_set', type=str, required=True,
                        help="path to the test set")
    parser.add_argument('--out', type=str, required=True,
                        help="path of the output file as required in the task description")
    args = parser.parse_args()

    
    # 1. load the training set (args.training_set)
    train_bus_df = pd.read_csv(args.training_set, encoding="ISO-8859-8")

    # 2. preprocess the training set
    logging.info("preprocessing train...")
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

    # 3. train a model
    logging.info("training...")
    best_model = find_best_XGBoost(X_train, y_train,X_fake_test,y_fake_test,X_dev,y_dev)

    # 4. load the test set (args.test_set)
    print(args.test_set)
    X_test = pd.read_csv(args.test_set, encoding="ISO-8859-8")

    # 5. preprocess the test set
    logging.info("preprocessing test...")
    original_trip_ids = X_test[['trip_id_unique']].copy()
    X_test_pre = preprocess_test(X_test)
    X_test_pre = preprocess_and_aggregate(X_test_pre)
    X_test_pre = X_test_pre.drop(columns=['trip_duration_in_minutes'])
    X_test_pre = X_test_pre.reindex(columns=X_train.columns, fill_value=0)
    # 6. predict the test set using the trained model
    logging.info("predicting...")
    y_pred = best_model.predict(X_test_pre)
    y_pred = np.round(y_pred)

    # 7. save the predictions to args.out
    output_df = pd.DataFrame()
    logging.info("predictions saved to {}".format(args.out))
    output_df['trip_id_unique'] = original_trip_ids['trip_id_unique'].unique()
    
    output_df['trip_duration_in_minutes'] = y_pred

    output_df.to_csv(args.out, index=False)





