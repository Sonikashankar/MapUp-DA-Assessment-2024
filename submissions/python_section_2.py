import pandas as pd
import numpy as np
from datetime import datetime,time, timedelta

def calculate_distance_matrix(dataset_path)->pd.DataFrame():
        # Write your logic here

    # Load the dataset into a DataFrame
    df = pd.read_csv('E:\downloads\MAPUP_ASSIGNMENT_2024\dataset-2.csv')

    distance_matrix = df.pivot(index='id_start', columns='id_end', values='distance').fillna(0)
    distance_matrix = distance_matrix + distance_matrix.transpose()
    distance_matrix.values[[range(len(distance_matrix))]*2] = 0

    # Calculate cumulative distances along known routes
    for col in distance_matrix.columns:
        for row in distance_matrix.index:
            if distance_matrix.at[row, col] == 0 and row != col:
                # Calculate cumulative distance for unknown routes
                known_routes = distance_matrix.loc[row, distance_matrix.loc[row] != 0]
                distance_matrix.at[row, col] = known_routes.sum()

    return distance_matrix

def unroll_distance_matrix(distance_matrix)->pd.DataFrame():
       # Write your logic here
    # Get the lower triangular part of the distance matrix (excluding the diagonal)
    lower_triangle = distance_matrix.where(np.tril(np.ones(distance_matrix.shape), k=-1).astype(bool))

    unrolled_series = lower_triangle.stack()
    unrolled_df = unrolled_series.reset_index()

    # Rename columns
    unrolled_df.columns = ['id_start', 'id_end', 'distance']

    return unrolled_df

def find_ids_within_ten_percentage_threshold(result_unrolled_df, reference_value)->pd.DataFrame():
   
    # Write your logic here
    # Filter rows based on the reference value
    reference_rows = result_unrolled_df[result_unrolled_df['id_start'] == reference_value]
    average_distance = reference_rows['distance'].mean()

    lower_threshold = average_distance * 0.9
    upper_threshold = average_distance * 1.1


    within_threshold_rows = result_unrolled_df[(result_unrolled_df['distance'] >= lower_threshold) & (result_unrolled_df['distance'] <= upper_threshold)]

    result_list = sorted(within_threshold_rows['id_start'].unique())

    return result_list

def calculate_toll_rate(df)->pd.DataFrame():
    # Wrie your logic here

    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient

    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    # Define time ranges and discount factors
    time_ranges_weekdays = [
        {'start_time': datetime.strptime('00:00:00', '%H:%M:%S').time(), 'end_time': datetime.strptime('10:00:00', '%H:%M:%S').time(), 'discount_factor': 0.8},
        {'start_time': datetime.strptime('10:00:00', '%H:%M:%S').time(), 'end_time': datetime.strptime('18:00:00', '%H:%M:%S').time(), 'discount_factor': 1.2},
        {'start_time': datetime.strptime('18:00:00', '%H:%M:%S').time(), 'end_time': datetime.strptime('23:59:59', '%H:%M:%S').time(), 'discount_factor': 0.8}
    ]

    time_ranges_weekends = [
        {'start_time': datetime.strptime('00:00:00', '%H:%M:%S').time(), 'end_time': datetime.strptime('23:59:59', '%H:%M:%S').time(), 'discount_factor': 0.7}
    ]

# Function to calculate discount factor based on time
def calculate_discount_factor(row):
    print("temp \n", row)
    current_time = row['start_time']

    # Check if it's a weekday or weekend
    if row['start_day'] in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        time_ranges = time_ranges_weekdays
    else:
        time_ranges = time_ranges_weekends

    for time_range in time_ranges:
        if time_range['start_time'] <= current_time <= time_range['end_time']:
            return time_range['discount_factor']

    df['discount_factor'] = df.apply(calculate_discount_factor, axis=1)

    vehicle_columns = ['moto', 'car', 'rv', 'bus', 'truck']
    for vehicle_column in vehicle_columns:
        df[vehicle_column] = df[vehicle_column] * df['discount_factor']

    df = df.drop(columns=['discount_factor'])

    return df


