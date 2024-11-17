import pandas as pd
import os
from sklearn.model_selection import train_test_split


def load_subject_data(data_dir, subject_folder):
    """
    Load all sensor files from a specific subject's folder.

    Args:
        data_dir (str): The base directory containing the raw data.
        subject_folder (str): The folder name for a specific subject (e.g., "Subject_1").

    Returns:
        pd.DataFrame: Combined data for the subject with a 'source' column.
    """
    subject_path = os.path.join(data_dir, subject_folder)
    sensor_files = [f for f in os.listdir(subject_path) if f.endswith('.csv')]
    subject_data = []

    for file in sensor_files:
        file_path = os.path.join(subject_path, file)
        sensor_name = os.path.splitext(file)[0]  # Extract the sensor name (e.g., chest)
        df = pd.read_csv(file_path)
        df['source'] = sensor_name  # Add source column for sensor type
        df['subject'] = subject_folder  # Add subject column for identification
        subject_data.append(df)

    return pd.concat(subject_data, ignore_index=True)


def combine_subjects(data_dir):
    """
    Combine all subject data dynamically from the raw directory.

    Args:
        data_dir (str): The base directory containing the raw data.

    Returns:
        pd.DataFrame: Combined data from all subjects.
    """
    subject_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    all_subject_data = []

    for subject in subject_folders:
        print(f"Loading data for {subject}...")
        subject_data = load_subject_data(data_dir, subject)
        all_subject_data.append(subject_data)

    return pd.concat(all_subject_data, ignore_index=True)


def split_and_save_data(data, output_dir, train_filename="combined_acc_walking.csv",
                        test_filename="test_combined_acc_walking.csv", test_size=0.25):
    """
    Split the combined data into training and testing datasets and save them.

    Args:
        data (pd.DataFrame): The combined dataset to split.
        output_dir (str): Directory to save the processed files.
        train_filename (str): Name of the training data file.
        test_filename (str): Name of the testing data file.
        test_size (float): Proportion of data to reserve for testing.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Split the data into training and testing
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42, shuffle=True)

    # Save the split data
    train_path = os.path.join(output_dir, train_filename)
    test_path = os.path.join(output_dir, test_filename)

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    print(f"Training data saved to {train_path}")
    print(f"Testing data saved to {test_path}")


# Main script
if __name__ == "__main__":
    # Define paths
    data_dir = "../data/raw/"  # Raw data directory
    output_dir = "../data/processed/"  # Directory to save processed files

    # Combine data from all subjects
    print("Combining data from all subjects...")
    combined_data = combine_subjects(data_dir)

    # Split and save the combined data
    print("Splitting and saving data...")
    split_and_save_data(combined_data, output_dir)

