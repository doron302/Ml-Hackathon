import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as ms
from xgboost import XGBRegressor


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



def find_best_XGBoost(X_train, y_train,X_test,y_test,X_dev,y_dev):

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

    # plot_results(train_losses, test_losses, dev_losses, params_list)
    # print("Best parameters found: ", best_params)
    # plot_lambda_vs_loss(models, best_params[0], best_params[1])

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