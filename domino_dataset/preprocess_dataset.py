import os
import pandas as pd
from tqdm import tqdm
import re

# Define the path to the DOMINO folder
domino_folder = 'DOMINO'


# Function to generate a list of timestamps from ts_start to ts_end (in milliseconds)
def generate_time_range(ts_start, ts_end):
    # Create a range of timestamps from ts_start to ts_end with 1 millisecond intervals
    time_range = list(range(int(ts_start), int(ts_end) + 1, 1))  # Increment by 1 millisecond
    return time_range


merged_files = []


# Iterate through all user folders in the DOMINO folder
for folder in tqdm(os.listdir(domino_folder)):
    folder_path = os.path.join(domino_folder, folder)

    if os.path.isdir(folder_path) and folder.startswith('user-'):
        # Extract the numeric part of the user folder name
        user_id = re.search(r'\d+', folder).group()

        activity_labels_path = os.path.join(folder_path, 'activity_labels.csv')
        acc_path = os.path.join(folder_path, 'smartwatch_acc.csv')
        gyro_path = os.path.join(folder_path, 'smartwatch_gyr.csv')

        # Step 1: Process activity_labels.csv
        if os.path.exists(activity_labels_path):
            activity_labels = pd.read_csv(activity_labels_path)

            # Expand the time ranges between ts_start and ts_end, and keep the "label" column
            expanded_data = []
            for _, row in activity_labels.iterrows():
                time_range = generate_time_range(row['ts_start'], row['ts_end'])

                expanded_df = pd.DataFrame({
                    'ts': time_range,
                    'label': row['label']  # Keep the label column constant for each expanded range
                })
                expanded_data.append(expanded_df)

            activity_labels_ts = pd.concat(expanded_data, ignore_index=True)

            # Save the expanded data as activity_labels_ts.csv
            activity_labels_ts_path = os.path.join(folder_path, 'activity_labels_ts.csv')
            activity_labels_ts.to_csv(activity_labels_ts_path, index=False)

            # Step 2: Load smartwatch_acc.csv and smartwatch_gyro.csv
            if os.path.exists(acc_path) and os.path.exists(gyro_path):
                smartwatch_acc = pd.read_csv(acc_path)
                smartwatch_gyro = pd.read_csv(gyro_path)

                smartwatch_acc.rename(columns={'x': 'acc_x', 'y': 'acc_y', 'z': 'acc_z'}, inplace=True)
                smartwatch_gyro.rename(columns={'x': 'gyro_x', 'y': 'gyro_y', 'z': 'gyro_z'}, inplace=True)

                merged_data = activity_labels_ts.merge(smartwatch_acc, on='ts', how='inner')
                merged_data = merged_data.merge(smartwatch_gyro, on='ts', how='inner')

                merged_data['user_id'] = user_id
                merged_data.rename(columns={'ts': 'timestamp'}, inplace=True)

                # Step 3: Save the merged data for this user
                merged_file_path = os.path.join(domino_folder, f'merged_user_{user_id}.csv')
                merged_data.to_csv(merged_file_path, index=False)
                print(f"Saved merged data for user {user_id} to {merged_file_path}")

                merged_files.append(merged_file_path)

# Step 4: Combine all individual merged files into one final CSV file
if merged_files:
    all_data = []

    for file in merged_files:
        df = pd.read_csv(file)
        all_data.append(df)

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_file_path = os.path.join(domino_folder, 'data.csv')
        combined_df.to_csv(combined_file_path, index=False)
        print(f"Saved final combined data to {combined_file_path}")

    else:
        print("No dataframes found to concatenate.")

else:
    print("No merged files found.")
