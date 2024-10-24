from typing import Dict, List,Any
import itertools
import re
import pandas as pd
import polyline
from typing import Tuple
from geopy.distance import geodesic


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    result = []
    length = len(lst)
    
    for i in range(0, length, n):
        group = lst[i:i+n]
        reversed_group = []
        for j in range(len(group) - 1, -1, -1):
            reversed_group.append(group[j])
        result.extend(reversed_group)
    
    return result


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    result = {}
    for string in lst:
        length = len(string)
        if length not in result:
            result[length] = []
        result[length].append(string)
    
    sorted_result = {k: result[k] for k in sorted(result)}
    
    return sorted_result


def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
    def _flatten(obj: Any, parent_key: str = '') -> Dict:
        items = {}
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(_flatten(v, new_key))
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    items.update(_flatten({f"{new_key}[{i}]": item}, ''))
            else:
                items[new_key] = v
        return items
    
    return _flatten(nested_dict)
    


def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    # Generate permutations using itertools
    permutations = list(itertools.permutations(nums))
    
    # Use set to remove duplicates and convert back to list
    unique_perms = list(set(permutations))
    
    # Convert each tuple to list
    unique_perms = [list(perm) for perm in unique_perms]
    
    return unique_perms



def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
     # Define regex patterns for different date formats
    date_patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  # dd-mm-yyyy
        r'\b\d{2}/\d{2}/\d{4}\b',  # mm/dd/yyyy
        r'\b\d{4}\.\d{2}\.\d{2}\b'  # yyyy.mm.dd
    ]
    
    # Combine the patterns into a single regex
    combined_pattern = '|'.join(date_patterns)
    
    # Find all matches in the text
    matches = re.findall(combined_pattern, text)
    
    return matches



def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """

    decoded_coords = polyline.decode(polyline_str)
    df = pd.DataFrame(decoded_coords, columns=['latitude', 'longitude'])
    df['distance'] = 0.0
    
    for i in range(1, len(df)):
        coord1 = (df.loc[i-1, 'latitude'], df.loc[i-1, 'longitude'])
        coord2 = (df.loc[i, 'latitude'], df.loc[i, 'longitude'])
        df.loc[i, 'distance'] = geodesic(coord1, coord2).meters
    
    return df



def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here n = len(matrix)
    n = len(matrix)
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    
    result = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            result[i][j] = row_sum + col_sum
    
    return result




def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    df['startTime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['endTime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    grouped = df.groupby(['id', 'id_2'])
    
    def check_completeness(group):
        total_time = (group['endTime'] - group['startTime']).sum()
        return total_time == pd.Timedelta(days=7)

    return grouped.apply(check_completeness)
