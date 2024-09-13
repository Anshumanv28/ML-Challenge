import pandas as pd
import argparse
import re
import os
import constants
from utils import parse_string


# Function to validate the file format (must be CSV)
def check_file(filename):
    if not filename.lower().endswith('.csv'):
        raise ValueError("Only CSV files are allowed.")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Filepath: {filename} invalid or not found.")


# The main sanity check function to compare test and output files
def sanity_check(test_filename, output_filename):
    # Ensure both files are valid
    check_file(test_filename)
    check_file(output_filename)

    # Read the test and output CSV files
    try:
        test_df = pd.read_csv(test_filename)
        output_df = pd.read_csv(output_filename)
    except Exception as e:
        raise ValueError(f"Error reading the CSV files: {e}")

    # Check if 'index' column exists in the test file
    if 'index' not in test_df.columns:
        raise ValueError("Test CSV file must contain the 'index' column.")

    # Check if 'index' and 'prediction' columns exist in the output file
    if 'index' not in output_df.columns or 'prediction' not in output_df.columns:
        raise ValueError(
            "Output CSV file must contain 'index' and 'prediction' columns.")

    # Check for missing indices (present in the test file but missing in the output file)
    missing_index = set(test_df['index']).difference(set(output_df['index']))
    if len(missing_index) != 0:
        print("Missing index in test file: {}".format(missing_index))

    # Check for extra indices (present in the output file but not in the test file)
    extra_index = set(output_df['index']).difference(set(test_df['index']))
    if len(extra_index) != 0:
        print("Extra index in output file: {}".format(extra_index))

    # Apply the parse_string function to check the format of the predictions
    output_df.apply(lambda x: parse_string(x['prediction']), axis=1)
    print("Parsing successful for file: {}".format(output_filename))


# Main function to handle command-line arguments
if __name__ == "__main__":
    # Usage example: python sanity.py --test_filename sample_test.csv --output_filename sample_test_out.csv

    parser = argparse.ArgumentParser(
        description="Run sanity check on a CSV file.")
    parser.add_argument("--test_filename",
                        type=str,
                        required=True,
                        help="The test CSV file name.")
    parser.add_argument("--output_filename",
                        type=str,
                        required=True,
                        help="The output CSV file name to check.")

    args = parser.parse_args()

    try:
        sanity_check(args.test_filename, args.output_filename)
    except Exception as e:
        print('Error:', e)
