import pandas as pd
import pandas as pd
import sklearn as sk
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression , Ridge, Lasso, ElasticNet, LassoLars, BayesianRidge, HuberRegressor, PassiveAggressiveRegressor, RANSACRegressor, SGDRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as ms
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor


def is_one_to_one(first_col, second_col, df):
    train_bus_df = pd.read_csv('train_set/train_bus_schedule.csv', encoding="ISO-8859-8")
    return (df.groupby(first_col)[second_col].nunique() == 1).all()


def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def preproccess_train(train_df):
    # Step 1: Drop rows where passengers_up is null or outliers
    # Define a function to identify and remove outliers using IQR method

    # Drop rows where passengers_up is null
    cleaned_df = train_df.dropna(subset=['passengers_up'])

    # Remove outliers from passengers_up
    # cleaned_df = remove_outliers(cleaned_df, 'passengers_up')

    # Step 2: Drop general outliers for other numerical columns
    # numerical_columns = ['passengers_continue', 'mekadem_nipuach_luz', 'passengers_continue_menupach']
    # for column in numerical_columns:
    #     cleaned_df = remove_outliers(cleaned_df, column)

    # Step 3: Drop the station_name column
    cleaned_df = cleaned_df.drop(columns=['station_name'])

    # Step 4: Create one-hot vectors from the column alternatives
    cleaned_df = pd.get_dummies(cleaned_df, columns=['alternative'])

    # Step 5: Change the cluster column to numerical values
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
            cleaned_df['arrival_time'].dt.hour * 10000 + cleaned_df['arrival_time'].dt.minute * 100 + cleaned_df[
        'arrival_time'].dt.second)

    cleaned_df['door_closing_time'] = pd.to_datetime(cleaned_df['door_closing_time'], format='%H:%M:%S')

    # Extract hour, minute, and second, convert to numeric value (e.g., HHMMSS format)
    cleaned_df['door_closing_time'] = (
            cleaned_df['door_closing_time'].dt.hour * 3600 + cleaned_df['door_closing_time'].dt.minute
            * 60 + cleaned_df['door_closing_time'].dt.second)

    cleaned_df['door_opening_duration'] = (cleaned_df['door_closing_time'] - cleaned_df['arrival_time']).fillna(0)

    cleaned_df = cleaned_df.drop(columns=['door_closing_time'])

    # Display the cleaned DataFramedsf
    return cleaned_df


def pearson_coor(cleaned_df):
    # Select only numeric columns
    numeric_data = cleaned_df.select_dtypes(include=['float64', 'int64', 'bool'])

    # Calculate the Pearson correlation for each numeric column with 'passengers_up'
    correlation = numeric_data.corr()['passengers_up']

    # Print the correlation values
    print(correlation)



def trainXGBboost(train_data, dev_data, reg_lambda,
                  n_estimators=50,
                  random_state=42,
                  max_depth=10,
                  learning_rate=0.1):
    model = XGBRegressor(
        objective='reg:squarederror',
        reg_lambda=reg_lambda,
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=max_depth,
        learning_rate=learning_rate,

    )
    y_train = train_data['passengers_up']

    X_train = train_data.drop(columns=['passengers_up'])

    y_dev = dev_data['passengers_up']
    X_dev = dev_data.drop(columns=['passengers_up'])

    # training the data
    model.fit(X_train, y_train)


    y_true_dev = y_dev
    y_pred_dev = model.predict(X_dev)
    y_pred_train = model.predict(X_train)
    dev_loss = (ms(y_true_dev, y_pred_dev))
    train_loss = (ms(y_train, y_pred_train))

    return dev_loss, train_loss

    # print(f"dev:error{ms(y_true_dev, y_pred_dev)}")
    # print(f"train:error{ms(y_train, y_pred_train)}")





def train_and_evaluate_random_forest(train_data, dev_data, n_estimators, random_state, max_depth):
    # Initialize the RandomForestRegressor with the specified parameters
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

    # Extract the target variable and features for training data
    y_train = train_data['passengers_up']
    X_train = train_data.drop(columns=['passengers_up'])

    # Extract the target variable and features for development data
    y_dev = dev_data['passengers_up']
    X_dev = dev_data.drop(columns=['passengers_up'])

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Predict on the development data
    y_pred_dev = model.predict(X_dev)
    y_pred_train = model.predict(X_train)

    # Calculate the mean squared error for both training and development data
    dev_loss = ms(y_dev, y_pred_dev)
    train_loss = ms(y_train, y_pred_train)

    return dev_loss, train_loss




def trainXGBboost(X_train, y_train, reg_lambda, n_estimators, max_depth):
    model = XGBRegressor(
        objective='reg:squarederror',
        reg_lambda=reg_lambda,
        n_estimators=n_estimators,
        random_state=42,
        max_depth=max_depth
    )
    # training the data
    model.fit(X_train, y_train)
    return model


def evalXGBboost(model, X_train, y_train, X_test, y_test, X_dev, y_dev):

    y_pred_dev = model.predict(X_dev)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_loss = (ms(y_train, y_pred_train))
    dev_loss = (ms(y_dev, y_pred_dev))
    test_loss = (ms(y_test, y_pred_test))

    return train_loss, test_loss, dev_loss


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as ms

def find_best_XGBoost(train_data, test_data, dev_data):
    y_train = train_data['passengers_up']
    X_train = train_data.drop(columns=['passengers_up'])

    y_test = test_data['passengers_up']
    X_test = test_data.drop(columns=['passengers_up'])

    y_dev = dev_data['passengers_up']
    X_dev = dev_data.drop(columns=['passengers_up'])

    train_losses = []
    test_losses = []
    dev_losses = []
    params_list = []

    # Track the best model
    best_model = None
    best_dev_loss = float('inf')
    best_params = None

    # Define the hyperparameters grid
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 10],
        'reg_lambda': [0.1, 0.5, 1, 5, 10]}

    models={}

    for n_estimators in param_grid['n_estimators']:
        for max_depth in param_grid['max_depth']:
            for lamb in param_grid['reg_lambda']:
                model = trainXGBboost(X_train, y_train, lamb, n_estimators, max_depth)
                train_loss, test_loss, dev_loss = evalXGBboost(model, X_train, y_train, X_test, y_test, X_dev, y_dev)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                dev_losses.append(dev_loss)
                params_list.append((n_estimators, max_depth, lamb))

                print(f"n_estimators: {n_estimators}, max_depth: {max_depth}, reg_lambda: {lamb}",
                      "Train Loss: ", train_loss, "Test Loss: ", test_loss, "Dev Loss: ", dev_loss)

                models[(n_estimators, max_depth, lamb)]=(train_loss, test_loss, dev_loss)

                # Check if this model is the best so far
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    best_model = model
                    best_params = (n_estimators, max_depth, lamb)

    plot_results(train_losses, test_losses, dev_losses, params_list)
    print("Best parameters found: ", best_params)
    plot_lambda_vs_loss(models, best_params[0], best_params[1])

    return best_model

def plot_lambda_vs_loss(models, best_n_estimator, best_max_depth ):

    filtered_models = {lamb: losses for (n_estimators, max_depth, lamb), losses in models.items()
                       if n_estimators == best_n_estimator and max_depth == best_max_depth}

    # Step 2: Sort the lambda values
    sorted_lambdas = sorted(filtered_models.keys())

    # Extract losses
    train_losses = [filtered_models[lamb][0] for lamb in sorted_lambdas]
    test_losses = [filtered_models[lamb][1] for lamb in sorted_lambdas]
    dev_losses = [filtered_models[lamb][2] for lamb in sorted_lambdas]

    # Step 3: Plot the losses
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_lambdas, train_losses, marker='o', label='Train Loss')
    plt.plot(sorted_lambdas, test_losses, marker='o', label='Test Loss')
    plt.plot(sorted_lambdas, dev_losses, marker='o', label='Dev Loss')
    plt.xlabel('Lambda')
    plt.ylabel('Loss')
    plt.title(f'Losses as a Function of Lambda\n(n_estimators={best_n_estimator}, max_depth={best_max_depth})')
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_results(train_losses, test_losses, dev_losses, params_list):
    # Convert the results to a DataFrame for easier plotting
    results_df = pd.DataFrame(params_list,
                              columns=['n_estimators', 'max_depth', 'reg_lambda'])
    results_df['train_loss'] = train_losses
    results_df['val_loss'] = dev_losses
    results_df['test_loss'] = test_losses

    # Plotting the results
    plt.figure(figsize=(14, 7))
    plt.plot(np.arange(len(train_losses)), train_losses, label='Train Loss', color='blue')
    plt.plot(np.arange(len(dev_losses)), dev_losses, label='Validation Loss', color='orange')
    plt.plot(np.arange(len(test_losses)), test_losses, label='Test Loss', color='green')
    plt.xlabel('Hyperparameter Combination Index')
    plt.ylabel('Mean Squared Error')
    plt.title('Train, Test, and Validation Losses')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    np.random.seed(42)

    train_bus_df = pd.read_csv('train_set/train_bus_schedule.csv', encoding="ISO-8859-8")

    clean_df = preproccess_train(train_bus_df)

    train_data, temp_data = train_test_split(clean_df, test_size=0.3, random_state=42)

    dev_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    pass_up_mean = train_data['passengers_up'].mean()

    models = [LinearRegression(), Ridge(), Lasso(), ElasticNet(), LassoLars(), BayesianRidge(), HuberRegressor(),
              PassiveAggressiveRegressor(), RANSACRegressor(), SGDRegressor()]

    model_names = [
        'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'LassoLars',
        'BayesianRidge', 'HuberRegressor', 'PassiveAggressiveRegressor',
        'RANSACRegressor', 'SGDRegressor'
    ]



    xgb_model=find_best_XGBoost(train_data, test_data, dev_data)



    # lambdas = [0.1, 0.5, 1, 5, 10]
    # for lamb in lambdas:
    #     dev_loss, train_loss = trainXGBboost(train_data, dev_data,
    #                                          reg_lambda=lamb,
    #                                          n_estimators=1000,
    #                                          random_state=42,
    #                                          max_depth=3,
    #                                          learning_rate=1)
    #     print(f"dev:error{dev_loss}")
    #     print(f"train:error{train_loss}")
    #     dev_losses.append(dev_loss)
    #     train_losses.append(train_loss)
    #
    # print(dev_losses)
    # print(train_losses)
