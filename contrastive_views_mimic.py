import numpy as np
import random
import pandas as pd
from tqdm import tqdm
def generate_interaction(file,typed,train_time):
    interaction=pd.read_csv(file)
    #interaction.dropna(inplace=True)
    print(len(interaction))
    doctor_graph=pd.read_csv('data_MIMIC/doctor_cluster.csv')
    medication_graph=pd.read_csv('data_MIMIC/med_cluster.csv')
    #medication_graph.drop_duplicates(subset='medication',inplace=True)
    df_with_na = interaction[interaction.isna().any(axis=1)]
    interaction.dropna(inplace=True)
    medication_graph.drop_duplicates(subset='medication',inplace=True)
    #room_graph=pd.read_csv('domain/room_graph.csv')
    #room_graph[room_graph['cost']==0.8]
    df_d=interaction[interaction['label']=='D']
    df_m=interaction[interaction['label']=='M']
    df_r=interaction[interaction['label']=='R']
    print(len(df_d))
    print(len(df_m))
    medication_graph.dropna(inplace=True)
    T=train_time
    df_other=interaction[interaction['timestamp']>=T]
    if typed=='similar':
        new_interactions=df_r
        for t in tqdm(range(T)):
            df_d_t=df_d[df_d['timestamp']==t]
            df_m_t=df_m[df_m['timestamp']==t]
            #df_r_t=df_r[df_r['timestamp']==t]
            # doctors
            merged_df = pd.merge(df_d_t, doctor_graph, left_on='item_id', right_on='order_provider_id', how='left')
            grouped = merged_df.groupby(['cluster_number', 'user_id'])
            rows_to_drop = []
            for _, group in grouped:
                if len(group['item_id'].unique()) > 1:
                    rows_to_drop.extend(group.index)
            final_df_d_t = merged_df.drop(rows_to_drop)
            final_df_d_t.drop(columns=['order_provider_id','cluster_number'],inplace=True)
            # medications
            merged_df = pd.merge(df_m_t, medication_graph, left_on='item_id', right_on='medication', how='left')
            grouped = merged_df.groupby(['cluster_number', 'user_id'])
            rows_to_drop = []
            for _, group in grouped:
                if len(group['item_id'].unique()) > 1:
                    rows_to_drop.extend(group.index)
            final_df_m_t = merged_df.drop(rows_to_drop)
            final_df_m_t.drop(columns=['medication','cluster_number'],inplace=True)
            new_interactions=pd.concat([new_interactions,final_df_d_t,final_df_m_t],axis=0)
        new_interactions=pd.concat([new_interactions,df_other,df_with_na])
        new_interactions.sort_values(by=['timestamp','user_id'],inplace=True)
        return new_interactions
    if typed=='dissimilar':
        new_interactions=df_r
        for t in tqdm(range(T)):
            df_d_t=df_d[df_d['timestamp']==t]
            df_m_t=df_m[df_m['timestamp']==t]
            #df_r_t=df_r[df_r['timestamp']==t]
            # doctors
            merged_df = pd.merge(df_d_t, doctor_graph, left_on='item_id', right_on='order_provider_id', how='left')
            user_sid_count = merged_df.groupby('user_id')['cluster_number'].nunique()
            # Find pairs of item_ids with different sids
            pairs_to_drop = set()
            for user_id, group in merged_df.groupby('user_id'):
                sids = group['cluster_number'].unique()
                if len(sids) > 1:
                    items = group['item_id'].unique()
                    for i in range(len(items)):
                        for j in range(i + 1, len(items)):
                            if doctor_graph.loc[doctor_graph['order_provider_id'] == items[i], 'cluster_number'].iloc[0] != doctor_graph.loc[doctor_graph['order_provider_id'] == items[j], 'cluster_number'].iloc[0]:
                                pairs_to_drop.add((user_id, items[i]))
                                pairs_to_drop.add((user_id, items[j]))

            # Drop rows from df_d_t based on pairs_to_drop
            rows_to_drop = merged_df[
                merged_df.apply(lambda row: (row['user_id'], row['item_id']) in pairs_to_drop, axis=1)
            ].index
            final_df_d_t = merged_df.drop(rows_to_drop)
            final_df_d_t.drop(columns=['order_provider_id','cluster_number'],inplace=True)
            # medications
            merged_df = pd.merge(df_m_t, medication_graph, left_on='item_id', right_on='medication', how='left')
            user_sid_count = merged_df.groupby('user_id')['cluster_number'].nunique()
            # Find pairs of item_ids with different sids
            pairs_to_drop = set()
            for user_id, group in merged_df.groupby('user_id'):
                sids = group['cluster_number'].unique()
                if len(sids) > 1:
                    items = group['item_id'].unique()
                    for i in range(len(items)):
                        for j in range(i + 1, len(items)):
                            #import pdb;pdb.set_trace()
                            try:
                                if medication_graph.loc[medication_graph['medication'] == items[i], 'cluster_number'].iloc[0] != medication_graph.loc[medication_graph['medication'] == items[j], 'cluster_number'].iloc[0]:
                                    pairs_to_drop.add((user_id, items[i],items[j]))
                                    #pairs_to_drop.add((user_id, items[j]))
    
                            except IndexError: # catch the error
                                continue
            consecutive_pairs=list(pairs_to_drop)
            num_samples=min(len(pairs_to_drop),int(0.1*len(df_m_t)))
            sampled_pairs = random.sample(consecutive_pairs, num_samples)
            pairs_to_drop1=[]
            for p in sampled_pairs:
                pairs_to_drop1.append((p[0],p[1]))
                pairs_to_drop1.append((p[0],p[2]))
            # Drop rows from df_d_t based on pairs_to_drop
            rows_to_drop = merged_df[
                merged_df.apply(lambda row: (row['user_id'], row['item_id']) in pairs_to_drop1, axis=1)
            ].index
            final_df_m_t = merged_df.drop(rows_to_drop)
            final_df_m_t.drop(columns=['medication','cluster_number'],inplace=True)
            new_interactions=pd.concat([new_interactions,final_df_d_t,final_df_m_t],axis=0)
        new_interactions=pd.concat([new_interactions,df_other,df_with_na])
        new_interactions.sort_values(by=['timestamp','user_id'],inplace=True)
        return new_interactions
path='data_MIMIC/overall.csv' #data_UIHC/interactions2010-01-01.csv
new_interactions=generate_interaction(path,'similar',50) #51 38
print(len(new_interactions))
print(len(new_interactions[new_interactions['label']=='D']))
print(len(new_interactions[new_interactions['label']=='M']))
#new_interactions['timestamp']=new_interactions['timestamp']+2
new_interactions.to_csv('hygmimic1/interactions1.csv',index=False)
new_interactions1=generate_interaction(path,'dissimilar',50)
print(len(new_interactions1))
print(len(new_interactions1[new_interactions1['label']=='D']))
print(len(new_interactions1[new_interactions1['label']=='M']))
#new_interactions1['timestamp']=new_interactions1['timestamp']+2
new_interactions1.to_csv('hygmimic2/interactions1.csv',index=False)