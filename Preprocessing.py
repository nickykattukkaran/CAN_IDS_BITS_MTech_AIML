import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load the datasets
attack_free_file = r"CSV_Converted\Attack_free_dataset.csv"
dos_attack_file = r"CSV_Converted\DoS_attack_dataset.csv"
fuzzy_attack_file = r"CSV_Converted\Fuzzy_attack_dataset.csv"
impersonation_attack_file = r"CSV_Converted\Impersonation_attack_dataset.csv"

# Load CSV files
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    
attack_free = load_data(attack_free_file)
dos_attack = load_data(dos_attack_file)
fuzzy_attack = load_data(fuzzy_attack_file)
impersonation_attack = load_data(impersonation_attack_file)

# Check for NaN values in each column 
def NaN_Check(df):
    nan_counts = df.isnull().sum() 
    print(nan_counts)


def Process(df, name):
    print(f"preprocessing function of {name} dataframe")
    print(df.head())
    # Calculate the time interval between consecutive rows 
    df['TimeInterval'] = df['Timestamp'].diff().fillna(0)
    print(df)
    # Check for NaN values in each column 
    NaN_Check(df)

    # Convert 'RemoteFrame' column to 0 for '000' and 1 for '100' 
    df['RemoteFrame'] = df['RemoteFrame'].replace({100: 1, 000: 0})
    # Print the RemoteFrame column alone 
    print(df['RemoteFrame'])

    # Remove the first character from the 'ID' column values 
    df['ID'] = df['ID'].apply(lambda x: x[1:] if len(x) > 1 else x)
    #Time difference between successive messages with the same ID
    df['InterArrival'] = df.groupby('ID')['Timestamp'].diff().fillna(0)
    
    # Drop the 'Timestamp' column 
    df = df.drop(columns=['Timestamp'])
    print(df)

    # Export DataFrame to CSV 
    df.to_csv('output1.csv', index=False) 
    print("DataFrame exported successfully to output.csv")

    # Perform One-Hot Encoding on the ID column 
    df_one_hot_encoded = pd.get_dummies(df, columns=['ID'], prefix='ID') 
    # Display the DataFrame 
    print(df_one_hot_encoded)

        # Export DataFrame to CSV 
    df_one_hot_encoded.to_csv('output2.csv', index=False) 
    print("DataFrame exported successfully to output.csv")

Process(dos_attack, "DoS Attack")
