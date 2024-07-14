from argparse import ArgumentParser
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from hackathon_code.model import *


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

    cleaned_df['arrival_time'] = pd.to_datetime(cleaned_df['arrival_time'], format='%H:%M:%S')

    # Extract hour, minute, and second, convert to numeric value (e.g., HHMMSS format)
    cleaned_df['arrival_time'] = (
            cleaned_df['arrival_time'].dt.hour * 3600 + cleaned_df['arrival_time'].dt.minute
            * 60 + cleaned_df['arrival_time'].dt.second)

    cleaned_df['door_closing_time'] = pd.to_datetime(cleaned_df['door_closing_time'], format='%H:%M:%S')

    # Extract hour, minute, and second, convert to numeric value (e.g., HHMMSS format)
    cleaned_df['door_closing_time'] = (
            cleaned_df['door_closing_time'].dt.hour * 3600 + cleaned_df['door_closing_time'].dt.minute
            * 60 + cleaned_df['door_closing_time'].dt.second)

    cleaned_df['door_opening_duration'] = (cleaned_df['door_closing_time'] - cleaned_df['arrival_time']).fillna(0)

    cleaned_df = cleaned_df.drop(columns=['door_closing_time'])

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

    cleaned_df['arrival_time'] = pd.to_datetime(cleaned_df['arrival_time'], format='%H:%M:%S')

    # Extract hour, minute, and second, convert to numeric value (e.g., HHMMSS format)
    cleaned_df['arrival_time'] = (
            cleaned_df['arrival_time'].dt.hour * 3600 + cleaned_df['arrival_time'].dt.minute
            * 60 + cleaned_df['arrival_time'].dt.second)

    cleaned_df['door_closing_time'] = pd.to_datetime(cleaned_df['door_closing_time'], format='%H:%M:%S')

    # Extract hour, minute, and second, convert to numeric value (e.g., HHMMSS format)
    cleaned_df['door_closing_time'] = (
            cleaned_df['door_closing_time'].dt.hour * 3600 + cleaned_df['door_closing_time'].dt.minute
            * 60 + cleaned_df['door_closing_time'].dt.second)

    cleaned_df['door_opening_duration'] = (cleaned_df['door_closing_time'] - cleaned_df['arrival_time']).fillna(0)

    cleaned_df = cleaned_df.drop(columns=['door_closing_time'])

    return cleaned_df

"""
usage:
    python code/main.py --training_set PATH --test_set PATH --out PATH

for example:
    python code/main.py --training_set /cs/usr/gililior/training.csv --test_set /cs/usr/gililior/test.csv --out predictions/trip_duration_predictions.csv 

"""

# implement here your load,preprocess,train,predict,save functions (or any other design you choose)


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
    # 2. preprocess the training set
    logging.info("preprocessing train...")

    train_bus_df = pd.read_csv(args.training_set, encoding="ISO-8859-8")
    clean_df = preproccess_train(train_bus_df)
    train_data, temp_data = train_test_split(clean_df, test_size=0.3, random_state=42)
    y_train = train_data['passengers_up']
    X_train = train_data.drop(columns=['passengers_up'])

    dev_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    y_dev = dev_data['passengers_up']
    X_dev = dev_data.drop(columns=['passengers_up'])

    y_test = test_data['passengers_up']
    X_test = test_data.drop(columns=['passengers_up'])

    # 3. train a model
    logging.info("training...")
    best_model = find_best_XGBoost(X_train, y_train, X_test, y_test, X_dev, y_dev)

    # 4. load the test set (args.test_set)
    # 5. preprocess the test set
    logging.info("preprocessing test...")
    X_test_pre = preprocess_test(X_test)
    X_test_pre = X_test_pre.reindex(columns=X_train.columns, fill_value=0)

    # 6. predict the test set using the trained model
    logging.info("predicting...")
    y_pred = best_model.predict(X_test_pre)
    y_pred = np.round(y_pred)

    # 7. save the predictions to args.out
    logging.info("predictions saved to {}".format(args.out))
    output_df = pd.DataFrame()
    output_df['trip_id_unique_station'] = X_test['trip_id_unique_station']
    output_df['passengers_up'] = y_pred

    output_df.to_csv(args.out, index=False)




