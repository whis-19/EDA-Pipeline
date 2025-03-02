import os
import json
import glob
import pandas as pd
import streamlit as st
from paths import *

def convert_json_to_csv(input_folder, output_folder):
    messages = []
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        messages.append(f"Created output folder: {output_folder}")
    
    json_files = glob.glob(os.path.join(input_folder, "*.json"))
    
    if not json_files:
        messages.append("No JSON files found in the folder.")
        return messages
    
    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            records = data.get("response", {}).get("data", [])
            df = pd.json_normalize(records)
            
            base_name = os.path.basename(json_file)
            file_name = os.path.splitext(base_name)[0]
            output_file = os.path.join(output_folder, file_name + ".csv")
            
            df.to_csv(output_file, index=False)
            messages.append(f"Converted {json_file} to {output_file}")
        except Exception as e:
            messages.append(f"Error processing {json_file}: {e}")
    
    return messages

def merge_csv_files(folder1, folder2, output_file):
    messages = []
    csv_files1 = glob.glob(os.path.join(folder1, "*.csv"))
    csv_files2 = glob.glob(os.path.join(folder2, "*.csv"))
    
    if not csv_files1 or not csv_files2:
        messages.append("No CSV files found in one or both folders.")
        return messages
    
    dfs1 = [pd.read_csv(file) for file in csv_files1]
    dfs2 = [pd.read_csv(file) for file in csv_files2]
    
    if not dfs1 or not dfs2:
        messages.append("No data to merge from one or both folders.")
        return messages
    
    df1 = pd.concat(dfs1, ignore_index=True, sort=False)
    df2 = pd.concat(dfs2, ignore_index=True, sort=False)
    merged_df = pd.concat([df1, df2], axis=1)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        messages.append(f"Created output directory: {output_dir}")
    
    try:
        merged_df.to_csv(output_file, index=False)
        messages.append(f"Merged CSV saved as {output_file}")
    except Exception as e:
        messages.append(f"Error saving merged CSV: {e}")
    
    return messages
def merger():
    st.write("Converting JSON to CSV...")
    convert_json_to_csv(json_input_folder, csv_output_folder)

    st.write("Process completed!")

    st.write("Merging CSV files...")
    merge_csv_files(csv_folder1, csv_folder2, merged_csv_output)

    st.write("Process completed!")
