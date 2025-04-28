import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import json

# Read the data
df1 = pd.read_csv('data_MIMIC/CDI_EMR.csv', parse_dates=['date'])
df2 = pd.read_csv('data_MIMIC/CDIx_EMR.csv', parse_dates=['date'])

# Drop unnecessary columns and concatenate dataframes
df1.drop(['cdate'], axis=1, inplace=True)
df = pd.concat([df1, df2], axis=0)
df['date'] = pd.to_datetime(df['date'])
df['timestamp'] = (df['date'] - pd.to_datetime('2128-01-01')).dt.days

# Prepare patient features
df['user_id'] = df['pid']
df['gender'] = df['gender'].astype(int)
df['prev_visit'] = df['prev_visit'].astype(int)
pat_feat = df[['user_id', 'timestamp', 'los', 'age', 'gender', 'prev_visit', 'insurance', 'marital_status']]
pat_feat = pd.get_dummies(pat_feat, columns=['insurance', 'marital_status'], drop_first=True)
pat_feat = pat_feat.drop_duplicates(subset=['user_id', 'timestamp'])

# Read interactions data
interactions = pd.read_csv('data_MIMIC/overall.csv')
interactions['state_label'] = [0 for i in range(len(interactions))]
interactions = interactions[['user_id', 'item_id', 'timestamp', 'state_label', 'label']]

# Merge interactions with patient features
whole = pd.merge(interactions, pat_feat, on=['timestamp', 'user_id'], how='inner')
whole.dropna(inplace=True)

# Create a dictionary for item_id to integer hash value mapping
unique_item_ids = whole['item_id'].unique()
item_id_map = {item_id: idx for idx, item_id in enumerate(unique_item_ids)}

# Apply the mapping to the item_id column
whole['item_id'] = whole['item_id'].map(item_id_map)

# Save the mapping dictionary to a JSON file
with open('item_id_map_mimic.json', 'w') as f:
    json.dump(item_id_map, f)

# Save the modified dataframe to a CSV file
whole.to_csv('MIMIC_interactions.csv', index=False)

print("Mapping dictionary saved to item_id_map.json")
print("Modified dataframe saved to MIMIC_interactions.csv")
