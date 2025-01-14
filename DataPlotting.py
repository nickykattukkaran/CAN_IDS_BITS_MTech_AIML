import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Summary statistics
def dataset_summary(df, name):
    print(f"\nSummary for {name} dataset:")
    print(f"Number of rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(df.describe())

def can_id_distribution(df, name):
    # Plotting CAN ID vs Index using Seaborn with sharp colors 
    plt.figure(figsize=(12, 6)) 
    sns.scatterplot(data=df, x='Index', y='ID', hue='ID', palette='Set2', s=100) 
    plt.title(f'Scatter Plot of CAN IDs vs. Index in {name}') 
    plt.xlabel('Index') 
    plt.ylabel('CAN ID') 
    plt.legend(title='CAN IDs') 
    plt.show()

# Analyze ID distribution
def analyze_id_distribution(df, name):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='ID', data=df, order=df['ID'].value_counts().index[:10])
    plt.title(f"Top 10 Most Frequent IDs in {name} Dataset")
    plt.xlabel("ID")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analyze_timestamps(df, name):
    df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
    df['Time_Diff'] = df['Timestamp'].diff()
    plt.figure(figsize=(12, 6))
    plt.plot(df['Time_Diff'], label="Time Difference Between Messages")
    plt.title(f"Message Timing Patterns in {name} Dataset")
    plt.xlabel("Index")
    plt.ylabel("Time Difference (seconds)")
    plt.legend()
    plt.show()
    print("\nSummary of Time Differences:")
    print(df['Time_Diff'].describe())

# Analyze DLC distribution
def analyze_dlc_distribution(df, name):
    plt.figure(figsize=(8, 5))
    sns.countplot(x='DLC', data=df)
    plt.title(f"DLC Distribution in {name} Dataset")
    plt.xlabel("DLC")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# Analyze time intervals
def analyze_time_intervals(df, name):
    if 'Timestamp' in df.columns:
        df['Time_Diff'] = df['Timestamp'].diff().fillna(0)
        plt.figure(figsize=(10, 6))
        plt.hist(df['Time_Diff'], bins=50, color='blue', alpha=0.7)
        plt.title(f"Time Interval Distribution in {name} Dataset")
        plt.xlabel("Time Interval (s)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()
    else:
        print(f"Timestamp column not found in {name} dataset.")

# Analyze Payload Uniqueness
def analyze_payload(df, name):
    unique_payloads = df['Payload'].nunique()
    print(f"\nNumber of unique payloads in {name} Dataset: {unique_payloads}")
    payload_counts = df['Payload'].value_counts()
    print(f"\nTop 5 most frequent payloads in {name} Dataset:")
    print(payload_counts.head(5))

# Analyze Remote Frames
def analyze_remote_frames(df, name):
    if 'RemoteFrame' in df.columns:
        remote_frame_counts = df['RemoteFrame'].value_counts()
        print(f"\nRemote Frame Analysis in {name} Dataset:")
        print(remote_frame_counts)
        plt.figure(figsize=(6, 6))
        remote_frame_counts.plot.pie(autopct='%1.1f%%', labels=['Standard', 'Remote'])
        plt.title(f"Proportion of Remote Frames {name} Dataset")
        plt.show()
    else:
        print(f"\nRemote Frame column not found in the {name} dataset.")



def CanId_Between_RemoteFrames(df, name):

    # Identify remote frames (DLC == 0)
    remote_frames = df[(df["Timestamp"] >= 259) & (df["Timestamp"] <= 260)]
    #print(remote_frames)
    df = remote_frames

    # Filter the remote frames
    remote_frames = df[df['RemoteFrame'] == 100]

    # Initialize the count
    count_between_frames = []

    # Initialize a dictionary to store results
    can_id_counts_between_frames = {}

    # Count CAN IDs between consecutive remote frames
    for i in range(len(remote_frames) - 1):
        start_index = remote_frames.index[i]
        end_index = remote_frames.index[i + 1]
        count = len(df.loc[start_index + 1:end_index - 1])
        count_between_frames.append(count)
        
        # Get unique CAN IDs 
        unique_ids = df.loc[start_index + 1:end_index - 1, 'ID'].unique() 
        print(f"Unique CAN IDs between frames {start_index} and {end_index}: {unique_ids}")

         # Get the interval data
        interval_data = df.loc[start_index + 1:end_index]

        # Count occurrences of each CAN ID
        can_id_counts = interval_data['ID'].value_counts().to_dict()

        # Store the results in the dictionary
        can_id_counts_between_frames[(start_index, end_index)] = can_id_counts

        # Print details for this interval
        #print(f"CAN ID counts between frames {start_index} and {end_index}: {can_id_counts}")

    print(f"CAN IDs between consecutive remote frames in {name}:", count_between_frames)
    print(f"CAN IDs counts between two remote frames in {name}:", can_id_counts_between_frames)
    return can_id_counts_between_frames


def plot_can_id_counts(can_id_counts_between_frames, name):
    # Convert the dictionary to a DataFrame for easier plotting
    import pandas as pd
    df_counts = pd.DataFrame.from_dict(can_id_counts_between_frames, orient='index')
    df_counts = df_counts.fillna(0)  # Fill NaN values with 0

    # Plot a stacked bar chart
    ax = df_counts.plot(kind='bar', stacked=True, figsize=(15, 8), colormap='viridis')

    # Add the annotations
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy() 
        if height > 0:
            ax.text(x + width / 2, y + height / 2, int(height), ha='center', va='center', fontsize=8, color='white')

    plt.title(f'Stacked Bar Chart of CAN IDs Between Remote Frames in {name}')
    plt.xlabel('Remote Frame Intervals')
    plt.ylabel('Count of CAN IDs')
    plt.legend(title='CAN IDs')
    plt.xticks(rotation=45)
    
    # Adjust layout to fit everything within the figure
    plt.tight_layout(pad=1.0)
    plt.show()

# Perform analysis on all datasets
def analyze_dataset(df, name):
    if df is not None:
        can_id_distribution(df, name)
        dataset_summary(df, name)
        analyze_id_distribution(df, name)
        analyze_timestamps(df, name)
        analyze_dlc_distribution(df, name)
        analyze_time_intervals(df, name)
        analyze_payload(df, name)
        analyze_remote_frames(df, name)
        plot_can_id_counts(CanId_Between_RemoteFrames(df, name), name)

# Run analysis
#analyze_dataset(attack_free, "Attack Free")
analyze_dataset(dos_attack, "DoS Attack")
#analyze_dataset(fuzzy_attack, "Fuzzy Attack")
#analyze_dataset(impersonation_attack, "Impersonation Attack")
