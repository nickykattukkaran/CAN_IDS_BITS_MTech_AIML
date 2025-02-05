import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os

# # Load the datasets
# attack_free_file = r"CSV_Converted/Attack_free_dataset.csv"
# dos_attack_file = r"CSV_Converted/DoS_attack_dataset.csv"
# fuzzy_attack_file = r"CSV_Converted/Fuzzy_attack_dataset.csv"
# impersonation_attack_file = r"CSV_Converted/Impersonation_attack_dataset.csv"

# Load CSV files
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    
# attack_free = load_data(attack_free_file)
# dos_attack = load_data(dos_attack_file)
# fuzzy_attack = load_data(fuzzy_attack_file)
# impersonation_attack = load_data(impersonation_attack_file)

# Check for NaN values in each column 
def NaN_Check(df):
    nan_counts = df.isnull().sum() 
    print(nan_counts)

# Function to check and update the payload 
def update_payload_RF(row): 
    if row['RemoteFrame'] == 1 and pd.isna(row['Payload']): 
        # check for remote frame and NaN payload 
        return 'ff ff ff ff ff ff ff ff' 
    return row['Payload'] 

# Define a function to pad the Payload
def pad_payload(row):
    if row['RemoteFrame'] == 0 and row['DLC'] < 8:
        payload = row['Payload'].split()
        # Pad with 'ff' up to 8 bytes
        payload.extend(['ff'] * (8 - len(payload)))
        return ' '.join(payload)
    return row['Payload']

def hex_to_binary(payload):
    """
    Convert a hexadecimal payload string to a binary string.

    Args:
        payload (str): A space-separated string of hexadecimal values.

    Returns:
        str: A binary representation of the payload, or None if invalid input.
    """
    try:
        # Split the hex payload into bytes
        hex_bytes = payload.split()
        # Convert each byte to binary and zero-pad to 8 bits
        binary_bytes = [format(int(byte, 16), '08b') for byte in hex_bytes]
        # Join the binary bytes with no spaces
        return ''.join(binary_bytes)
    except ValueError as e:
        print(f"Invalid payload: {payload}. Error: {e}")
        return None  # Return None for invalid input

def Process(df, name):
    print(f"preprocessing function of {name} dataframe")
    #print(df.head())
    # Calculate the time interval between consecutive rows 
    df['TimeInterval'] = df['Timestamp'].diff().fillna(0)

    count = (df['TimeInterval'] > 0.008).sum() 
    print("count > 0.008 is :", count)

    # Filter rows where TimeInterval <= 0.008
    #df = df[df['TimeInterval'].astype(float) <= 0.008]
    
    # Ensure df is an independent copy after filtering
    df = df[df['TimeInterval'].astype(float) <= 0.008].copy()
    # print(df)
    # df.to_csv('demo.csv', index=False) 
    
    # Check for NaN values in each column 
    NaN_Check(df)

    # Convert 'RemoteFrame' column to 0 for '000' and 1 for '100' 
    df['RemoteFrame'] = df['RemoteFrame'].replace({100: 1, 000: 0})

    # Print the RemoteFrame column alone 
    #print(df['RemoteFrame'])

    # Remove the first character from the 'ID' column values 
    df['ID'] = df['ID'].apply(lambda x: x[1:] if len(x) > 1 else x)
    #Time difference between successive messages with the same ID
    df['InterArrival'] = df.groupby('ID')['Timestamp'].diff().fillna(0)
    
    # Drop the 'Timestamp' column 
    df = df.drop(columns=['Timestamp'])
    #print(df)

    # Convert the ID column to binary encoding 
    df['ID'] = df['ID'].apply(lambda x: format(int(x, 16), '012b'))

    # Apply the padding function
    df['Payload'] = df.apply(pad_payload, axis=1)
    
    # Apply the function to the RemoteFrame 
    df['Payload'] = df.apply(update_payload_RF, axis=1)
    #print(df)

    NaN_Check(df)

    #Export DataFrame to CSV 
    # df.to_csv('output1.csv', index=False) 
    # print("DataFrame exported successfully to output1.csv")

    # Apply the conversion function to the Payload column
    df['Payload'] = df['Payload'].apply(hex_to_binary)

    # Convert TimeInterval and InterArrival to microseconds
    df["TimeInterval"] = (df["TimeInterval"] * 1_000_000).astype(int)
    df["InterArrival"] = (df["InterArrival"] * 1_000).astype(int) #ms
    print(f"The Max value of Time Interval in {name}: ", df['TimeInterval'].max())
    max_index = df['TimeInterval'].idxmax() 
    print("max_index : ", max_index)

    # Format TimeInterval as strings with leading zeros to always have 3 digits
    #df["TimeInterval"] = df["TimeInterval"].apply(lambda x: f"{x:04d}")
    #df["InterArrival"] = df["InterArrival"].apply(lambda x: f"{x:04d}")

    #Export DataFrame to CSV 
    # df.to_csv('output2.csv', index=False) 
    # print("DataFrame exported successfully to output2.csv")

    # Drop the 'InterArrival' column 
    df = df.drop(columns=['InterArrival'])

    # Convert DLC to binary
    df['DLC'] = df['DLC'].apply(lambda x: format(x, '04b'))  # Pad to 8 bits

    #df['TimeInterval'] = df['TimeInterval'].astype(int)  # Convert to integers
    print(f"The Max value of Time Interval in {name}: ", df['TimeInterval'].max())
    df['TimeInterval'] = df['TimeInterval'].apply(lambda x: format(x, '013b'))  # Pad to 16 bits

    NaN_Check(df)

    # Export DataFrame to CSV 
    df.to_csv(f'PreprocessedData_CSV/{name.replace(" ", "")}_output.csv', index=False) 
    print(f"DataFrame exported successfully to PreprocessedData_CSV/{name}_output.csv")

    return df


output_folder = "PreprocessedData_CSV"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process(dos_attack, "DoS Attack")
# Process(attack_free, "Attack Free")
# Process(fuzzy_attack, "Fuzzy Attack")
# Process(impersonation_attack, "Impersonation Attack")


    