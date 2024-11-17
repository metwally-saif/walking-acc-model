import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks


def extract_features(data, window_size=100):
    """
    Extracts features from accelerometer data.
    Args:
        data (pd.DataFrame): Combined data with 'attr_x', 'attr_y', 'attr_z', and 'source'.
        window_size (int): Number of samples per window for feature extraction.
    Returns:
        pd.DataFrame: Extracted features.
    """
    features = []

    for sensor in data['source'].unique():
        sensor_data = data[data['source'] == sensor]

        # Sliding window
        for start in range(0, len(sensor_data) - window_size, window_size):
            window = sensor_data.iloc[start:start + window_size]

            feature = {
                "source": sensor,
                "mean_x": window['attr_x'].mean(),
                "mean_y": window['attr_y'].mean(),
                "mean_z": window['attr_z'].mean(),
                "std_x": window['attr_x'].std(),
                "std_y": window['attr_y'].std(),
                "std_z": window['attr_z'].std(),
                "skew_x": skew(window['attr_x']),
                "skew_y": skew(window['attr_y']),
                "skew_z": skew(window['attr_z']),
                "kurt_x": kurtosis(window['attr_x']),
                "kurt_y": kurtosis(window['attr_y']),
                "kurt_z": kurtosis(window['attr_z']),
                "magnitude_mean": np.sqrt((window[['attr_x', 'attr_y', 'attr_z']] ** 2).sum(axis=1)).mean(),
                "magnitude_std": np.sqrt((window[['attr_x', 'attr_y', 'attr_z']] ** 2).sum(axis=1)).std(),
                "zero_crossings_x": len(np.where(np.diff(np.sign(window['attr_x'])))[0]),
                "zero_crossings_y": len(np.where(np.diff(np.sign(window['attr_y'])))[0]),
                "zero_crossings_z": len(np.where(np.diff(np.sign(window['attr_z'])))[0]),
            }
            features.append(feature)

    return pd.DataFrame(features)


# Example usage
if __name__ == "__main__":
    # feature data
    combined_data = pd.read_csv("../data/processed/combined_acc_walking.csv")
    feature_data = extract_features(combined_data, window_size=100)
    feature_data.to_csv("../data/processed/feature_data.csv", index=False)
    print("Feature data saved to ../data/processed/feature_data.csv")
    # test feature_data
    test_combined_data = pd.read_csv("../data/processed/test_combined_acc_walking.csv")
    feature_data = extract_features(combined_data, window_size=100)
    feature_data.to_csv("../data/processed/test_feature_data.csv", index=False)
    print("Feature data saved to ../data/processed/test_feature_data.csv")
