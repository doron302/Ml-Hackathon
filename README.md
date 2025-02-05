# ML Hackathon Project: Bus Line System Analysis

## Overview
This project predicts aspects of a bus line system, including the number of passengers boarding at each stop, the total duration of bus trips, and additional predictions.

## Tasks

### Task 1: Predicting Number of Passengers Boarding

**Data Preprocessing**
- Dropped rows with missing labels
- Removed redundant columns
- Converted categorical features using one-hot encoding
- Created a feature for the duration the bus door was open

**Model Development**
- **Baseline Model**: Linear Regression (MSE: 3.02)
- **Advanced Model**: XGBoost (MSE: 1.99)

### Task 2: Predicting Trip Duration

**Data Preprocessing**
- Grouped stops by trip ID
- Calculated trip duration, adjusting for midnight crossings

**Feature Engineering**
- Selected relevant features
- Applied aggregation methods (mean, max, sum)

**Model Development**
- **Baseline Model**: Linear Regression (MSE: 166.64)
- **Advanced Model**: XGBoost (MSE: 143.57)

### Task 3: Additional Predictions

**Data Preprocessing**
- Handled special cases and anomalies
- Created new features for better predictive power

**Model Development**
- Applied advanced ML techniques

## Results
- Improved prediction accuracy using XGBoost
- Insights into special cases and anomalies

## Conclusion
- Most crowded times: 7-9 AM and 4-7 PM
- Suggest adding more bus lines during peak hours
- Adjust bus frequency and routes in high congestion areas

## How to Run

1. **Preprocessing**: Ensure required libraries are installed (e.g., pandas, numpy, scikit-learn, xgboost). Run preprocessing scripts.
2. **Model Training**: Use provided scripts to train models. Adjust hyperparameters as needed.
3. **Evaluation**: Evaluate models on the test set and analyze performance metrics.

## Authors

- **Roy Mainfeld**
- **Doron Shwartz**
- **Karin Lein**
- **Dvir Gil**

## Acknowledgments

- Hackathon organizers and mentors
- Team members for their contributions and collaboration
