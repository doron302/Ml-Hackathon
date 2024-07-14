import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import geodesic



def heatmap_correlation(data):
    # Select numeric columns
    numeric_data = data.select_dtypes(include=['float64', 'int64', 'bool'])
    target_column = 'passengers_up'

    # Calculate the correlation coefficients with the target variable
    correlation_series = numeric_data.corr()[target_column].sort_values(ascending=False)

    # Convert the correlation series to a DataFrame for heatmap
    correlation_df = pd.DataFrame(correlation_series).transpose()

    # Create a heatmap of the correlation coefficients
    plt.figure(figsize=(12, 2))
    sns.heatmap(correlation_df, annot=True, cmap='coolwarm', fmt='.2f', cbar=False)
    plt.title(f'Correlation with {target_column}')
    plt.show()

def distance_vs_passengers_up(data):

    # Ensure datetime format
    data['arrival_time'] = pd.to_datetime(data['arrival_time'])

    # Sort data
    data = data.sort_values(by=['trip_id', 'arrival_time'])

    # Calculate distances between consecutive stations
    data[['latitude_next', 'longitude_next']] = data.groupby('trip_id')[['latitude', 'longitude']].shift(-1)

    def calculate_distance(row):
        coords_1 = (row['latitude'], row['longitude'])
        coords_2 = (row['latitude_next'], row['longitude_next'])
        if pd.notnull(coords_2[0]) and pd.notnull(coords_2[1]):
            return geodesic(coords_1, coords_2).kilometers
        return np.nan

    data['distance_next'] = data.apply(calculate_distance, axis=1)

    # Remove rows with NaN values in distance_next
    data = data.dropna(subset=['distance_next'])

    # Scatter plot to visualize correlation
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='distance_next', y='passengers_up', data=data)
    plt.title('Correlation between Distance between Stations and Passengers Going Up')
    plt.xlabel('Distance between Stations (km)')
    plt.ylabel('Number of Passengers Going Up')
    plt.show()

    # Calculate and print the correlation coefficient
    correlation_coefficient = data['distance_next'].corr(data['passengers_up'])
    print(f'Correlation Coefficient: {correlation_coefficient:.2f}')


def passenger_boarding_map(data):
    # Map of Passenger Boardings
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='longitude', y='latitude', size='passengers_up', hue='passengers_up', data=data,
                    palette='viridis', sizes=(20, 200))
    plt.title('Map of Passenger Boardings')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()



def mean_passenger_per_station(data):
    # Calculate the mean number of passengers going up in each station
    mean_passengers_per_station = data.groupby('station_id')['passengers_up'].mean().reset_index()

    # Plot the mean number of passengers going up in each station
    plt.figure(figsize=(14, 8))
    sns.barplot(data=mean_passengers_per_station, x='station_id', y='passengers_up', palette='viridis')
    plt.title('Mean Number of Passengers Going Up in Each Station')
    plt.xlabel('Station ID')
    plt.ylabel('Mean Number of Passengers Going Up')
    plt.xticks(rotation=90)
    plt.show()

def peak_hours(data):
    # Ensure datetime format
    data['arrival_time'] = pd.to_datetime(data['arrival_time'])

    # Extract hour from arrival_time
    data['hour'] = data['arrival_time'].dt.hour

    # Group by hour and calculate mean passengers_up
    hourly_passenger_volume = data.groupby('hour')['passengers_up'].mean().reset_index()

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=hourly_passenger_volume, x='hour', y='passengers_up', marker='o')
    plt.title('Average Passenger Volume by Hour')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Number of Passengers Going Up')
    plt.grid(True)
    plt.savefig('passenger_volume_by_hour.png')
    plt.show()

def region_vs_passengers_up(data):
    # Group by region and calculate mean passengers_up

    # Drop rows where 'cluster' is null
    data = data.dropna(subset=['cluster'])
    regional_passenger_volume = data.groupby('cluster')['passengers_up'].mean().reset_index()

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(data=regional_passenger_volume, x='cluster', y='passengers_up')
    plt.title('Average Passenger Volume by Region')
    plt.xlabel('Region')
    plt.ylabel('Average Number of Passengers Going Up')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig('passenger_volume_by_region.png')
    plt.show()

def bus_line_vs_passengers_up(data):
    # Group by line_id and calculate mean passengers_up
    line_passenger_volume = data.groupby('line_id')['passengers_up'].mean().reset_index()

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(data=line_passenger_volume, x='line_id', y='passengers_up')
    plt.title('Average Passenger Volume by Bus Line')
    plt.xlabel('Bus Line')
    plt.ylabel('Average Number of Passengers Going Up')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig('passenger_volume_by_line.png')
    plt.show()


# after we will have a model that predicts the trip duration:
def passenger_boarding_vs_trip_duration(data):
    # Scatter Plot of Passenger Boardings vs. Trip Duration
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='trip_duration_in_minutes', y='passengers_up', data=data)
    plt.title('Passenger Boardings vs. Trip Duration')
    plt.xlabel('Trip Duration (minutes)')
    plt.ylabel('Number of Passengers')
    plt.show()
