import pandas as pd


def bus_line_frequency_with_clusters(file_path):
    # Load the dataset with 'ISO-8859-8' encoding
    bus_data = pd.read_csv(file_path, encoding='ISO-8859-8')

    # Count the occurrences of each bus line
    line_frequency = bus_data['line_id'].value_counts().reset_index()
    line_frequency.columns = ['line_id', 'frequency']

    # Merge with the cluster information
    cluster_info = bus_data[['line_id', 'cluster']].drop_duplicates()
    line_frequency_with_clusters = line_frequency.merge(cluster_info, on='line_id', how='left')

    # Sort by frequency in descending order
    line_frequency_with_clusters = line_frequency_with_clusters.sort_values(by='frequency', ascending=False)

    return line_frequency_with_clusters

file_path = 'train_set/train_bus_schedule.csv'
line_frequency_with_clusters = bus_line_frequency_with_clusters(file_path)

print("Bus Line Frequency with Clusters:")
print(line_frequency_with_clusters)

# For top 15 most frequent bus lines
top_15_lines_with_clusters = line_frequency_with_clusters.head(15)
print("\nTop 15 Bus Lines with Clusters:")
print(top_15_lines_with_clusters)


def bus_line_frequency(file_path):
    # Load the dataset with 'ISO-8859-8' encoding
    bus_data = pd.read_csv(file_path, encoding='ISO-8859-8')

    # Count the occurrences of each bus line
    line_frequency = bus_data['line_id'].value_counts().reset_index()
    line_frequency.columns = ['line_id', 'frequency']

    # Sort by frequency in descending order
    line_frequency = line_frequency.sort_values(by='frequency', ascending=False)
    top_15_lines = line_frequency.head(15)
    return top_15_lines


# Example usage
file_path = 'train_set/train_bus_schedule.csv'
line_frequency = bus_line_frequency(file_path)

print("Bus Line Frequency:")
print(line_frequency)



def cluster_peak_hours(file_path):
    # Load the dataset with 'ISO-8859-8' encoding
    bus_data = pd.read_csv(file_path, encoding='ISO-8859-8')

    # Convert 'arrival_time' to datetime
    bus_data['arrival_time'] = pd.to_datetime(bus_data['arrival_time'], format='%H:%M:%S')

    # Extract hour from 'arrival_time'
    bus_data['hour'] = bus_data['arrival_time'].dt.hour

    # Group by cluster and hour, then sum the passengers_up
    passengers_by_cluster = bus_data.groupby(['cluster', 'hour'])['passengers_up'].sum().reset_index()

    # Calculate the total number of passengers per cluster for the entire day
    total_passengers_per_cluster = bus_data.groupby('cluster')['passengers_up'].sum().reset_index()

    # Identify peak hours for each cluster by finding the hour with the maximum passengers
    peak_hours_by_cluster = passengers_by_cluster.loc[
        passengers_by_cluster.groupby('cluster')['passengers_up'].idxmax()]

    return total_passengers_per_cluster, peak_hours_by_cluster


# Example usage
file_path = 'train_set/train_bus_schedule.csv'
total_passengers_per_cluster, peak_hours_by_cluster = cluster_peak_hours(file_path)

print("Total Passengers Per Cluster:")
print(total_passengers_per_cluster)

print("\nPeak Hours By Cluster:")
print(peak_hours_by_cluster)
