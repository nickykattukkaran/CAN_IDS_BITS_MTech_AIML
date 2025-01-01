import os
import pandas as pd
from datetime import datetime

# Define the input directory containing multiple OTIDS dataset files
input_dir = r"OTIDS_Dataset"  # Directory where input files are stored

# Dynamically create a folder to store CSV files
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # Current timestamp
output_dir = f"CSV_Converted_{timestamp}"  # Folder name with timestamp
os.makedirs(output_dir, exist_ok=True)  # Create the folder if it doesn't exist

print("Processing the data...")

# Iterate through all files in the input directory
for file_name in os.listdir(input_dir):
    # Construct full file path
    input_file = os.path.join(input_dir, file_name)

    # Process only files with .txt extension
    if os.path.isfile(input_file) and file_name.endswith(".txt"):
        print(f"Processing file: {file_name}")

        # Open and parse the raw data
        data = []
        with open(input_file, 'r') as file:
            for line in file:
                try:
                    # Extract data from each line
                    parts = line.split()
                    timestamp = parts[1]  # Extract timestamp
                    message_id = parts[3]  # Extract CAN ID
                    remote_frame = parts[4]  # Extract Remote Frame
                    dlc = parts[6]  # Extract DLC
                    payload = " ".join(parts[7:])  # Extract Payload

                    # Append to data list
                    data.append([timestamp, message_id, remote_frame, dlc, payload])
                except IndexError:
                    print(f"Skipping malformed line in {file_name}: {line.strip()}")

        # Convert to DataFrame
        columns = ['Timestamp', 'ID', 'RemoteFrame', 'DLC', 'Payload']
        df = pd.DataFrame(data, columns=columns)

        # Add the index column explicitly
        df.reset_index(inplace=True)  # Resets the index and adds it as a column
        df.rename(columns={'index': 'Index'}, inplace=True)  # Rename the index column

        # Define output CSV file path
        output_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.csv")

        # Save as CSV
        df.to_csv(output_file, index=False)  # Exclude default index
        print(f"Converted and saved: {output_file}")

print(f"All files have been processed and saved in {output_dir}")
