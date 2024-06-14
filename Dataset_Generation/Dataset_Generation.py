import pandas as pd
import numpy as np
import random

# Set constants for number of users and timestamps per user
users_num = 10000

# Generate column names
columns = ['user_id']
columns.extend([f'_{i}_{metric}' for i in range(1, 7) for metric in [
    'cognitive_flexibility',
    'commission_errors',
    'correct_responses',
    'error_in_answers',
    'inattentiveness',
    'incorrect_clicks',
    'motor_control',
    'ommission_errors',
    'processing_speed',
    'puzzle_duration',
    'reaction_time',
    'response_inhibition',
    'response_speed',
    'sustained_attention',
    'task_switching',
    'time_taken',
    'working_memory']])
columns.extend([f'_{i}_distracted' for i in range(1, 11)])
columns.append('has_adhd')

# Create an empty DataFrame
df = pd.DataFrame(columns=columns)

had_adhd = None
# Generate data and populate DataFrame
for user_id in range(1, users_num + 1):
    has_adhd = 1 if user_id % 5 == 0 else 0
    
    # Dictionary to hold the data for the current row
    row_data = {'user_id': user_id}
    
    # Generate random data for other columns (except user_id and has_adhd)
    for col in columns[1:-11]:
        row_data[col] = round(random.uniform(0, 1), 1)

    for col in columns[-11:-1]:
        if has_adhd == 1:
            row_data[col] = random.choice([0,1])
        else:
            row_data[col] = 1 if (random.choice([0,1]) == 1 and random.choice([0,1]) == 1) else 0
    
    # Add data for 'has_adhd' column, assuming binary classification
    row_data['has_adhd'] = has_adhd
    
    # Append the row to the DataFrame
    df = df._append(row_data, ignore_index=True)

# Save the DataFrame to a CSV file for further analysis, without the index column
updated_csv_path = "dataset.csv"
df.to_csv(updated_csv_path, index=False)

# Return the path to the saved CSV and display the first few rows of the DataFrame for verification
updated_csv_path, df.head()
