import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dhg
from tqdm import tqdm
import torch
import csv
def create_incidence_matrix(hyperedges, p_nodes):
    incidence_matrix = []
    for hyperedges_per_timestamp in hyperedges:
        (rows, columns) = (len(p_nodes), len(hyperedges_per_timestamp))
        incidence_matrix_per_timestamp = [[0] * columns for i1 in range(rows)]
        for i in range(rows):
            for j in range(columns):
                if i in hyperedges_per_timestamp[j]:
                    incidence_matrix_per_timestamp[i][j] = 1
        
        np_incidence_matrix_per_timestamp = np.array(incidence_matrix_per_timestamp) 
        incidence_matrix.append(np_incidence_matrix_per_timestamp)
        print(np_incidence_matrix_per_timestamp.shape)
        #print(incidence_matrix_per_timestamp)

    return incidence_matrix




            

import csv
def main():
    df=pd.read_csv('data_MIMIC/overall.csv') #data_UIHC/4_day_ahead.csv  hyg2/interactions1.csv
    unique_d=df[df['label']=='D']['item_id'].unique().tolist()
    unique_m=df[df['label']=='M']['item_id'].unique().tolist()
    unique_r=df[df['label']=='R']['item_id'].unique().tolist()
    unique_all=df['item_id'].unique().tolist()
    unique_p=df['user_id'].unique().tolist()
    i=0
    D2id={}
    M2id={}
    R2id={}
    P2id={}
    A2id={}
    for d in unique_d:
        D2id[d]=i
        i+=1
    i=0
    for d in unique_m:
        M2id[d]=i
        i+=1
    i=0
    for d in unique_r:
        R2id[d]=i
        i+=1
    #i=0
    #import pdb;pdb.set_trace()
    #for d in unique_p:
    #    P2id[d]=i
    #    i+=1 
    i=0
    with open('mappings/P2id_mimic.csv', 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pid = int(row['pid'])
            index = int(row['index'])
            P2id[pid] = index
    i=0
    for d in unique_all:
        A2id[d]=i
        i+=1
    A2id
    i=0
    hex=[]
    #print(list(D2id.keys()))
    #A1={k: [D2id.get(k), M2id.get(k)] for k in set(D2id.keys() + M2id.keys())}
    #A2id={k: [A1.get(k), R2id.get(k)] for k in set(A1.keys() + R2id.keys())}
    hyperedges=[]
    while (i<=93):
        print(i)
        dat=df[df['timestamp']==i]
        unique_pats=dat['user_id'].unique().tolist()
        hea={}
        for pa in unique_pats:
            df_o=dat[dat['user_id']==pa]['item_id'].unique().tolist()
            int_ids=[]
            for d in df_o:
                int_p=dat[dat['item_id']==d]['user_id'].unique().tolist()
                
                for p in int_p:
                    if p == pa:
                        pass
                    else:
                        int_ids.append(P2id[p])
            hea[P2id[pa]]=int_ids
        
        hex.append(hea)
        #import pdb;pdb.set_trace()
        df_d=dat[dat['label']=='D']['item_id'].unique().tolist()
        he=[]
        hed=[]
        for d in df_d:
            int_p=dat[dat['item_id']==d]['user_id'].unique().tolist()
            int_ids=[]
            for p in int_p:
                int_ids.append(P2id[p])
            if len(int_ids) > 1:    
                hed.append(int_ids)
        he.append(hed)
        df_m=dat[dat['label']=='M']['item_id'].unique().tolist()
        hem=[]
        for d in df_m:
            int_p=dat[dat['item_id']==d]['user_id'].unique().tolist()
            int_ids=[]
            for p in int_p:
                int_ids.append(P2id[p])
            if len(int_ids) > 1:    
                hem.append(int_ids)
        
        he.append(hem)
        df_r=dat[dat['label']=='R']['item_id'].unique().tolist()
        her=[]
        for d in df_r:
            int_p=dat[dat['item_id']==d]['user_id'].unique().tolist()
            int_ids=[]
            for p in int_p:
                int_ids.append(P2id[p])
            if len(int_ids) > 1:
                her.append(int_ids)
        
        he.append(her)

        hyperedges.append(he)
        i+=1
    he_d=[]
    he_m=[]
    he_r=[]
    he_a=[]
    for i in range(len(hyperedges)):
        print(len(hyperedges[i]))
        #he_a.append(hyperedges[i][0])
        he_d.append(hyperedges[i][0])
        he_m.append(hyperedges[i][1])
        he_r.append(hyperedges[i][2])
        file_path1 = 'hygmimic/d'+str(i)+'.csv'
        file_path2 = 'hygmimic/m'+str(i)+'.csv'
        file_path3 = 'hygmimic/r'+str(i)+'.csv'
        # Writing the list of lists to a CSV file
        with open(file_path1, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(he_d[i])
        with open(file_path2, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(he_m[i])
        with open(file_path3, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(he_r[i])
    csv_file_path = 'P2id.csv'

    # Define the header
    header = ['pid', 'index']
    # csv_file_path='mappings/P2id_mimic.csv'
    # # Write the dictionary to the CSV file
    # with open(csv_file_path, 'w', newline='') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=header)
    #     writer.writeheader()
    #     for pid, index in P2id.items():
    #         writer.writerow({'pid': pid, 'index': index})

    print("CSV file saved successfully.")
    #import pdb;pdb.set_trace()
    """ def find_neighbors(lst, target):
        neighbors = []
        for sub_list in lst:
            if target in sub_list:
                eighbors.extend([item for item in sub_list if item != target])
        return neighbors
    hyx=[]
    for p in P2id.keys():
        neighbors = find_neighbors(hex, p)
        if len(neighbors)>0:
            neighbors.append(p)
            hyx.append(neighbors)
        else:
            neighbors.append(p)
            hyx.append(neighbors)
    
    for i in range(len(hex)):
        #import pdb;pdb.set_trace()
        dfd=pd.DataFrame.from_dict(hex[i],orient='index').transpose()
        dfd = dfd.astype(int, errors='ignore')
        path='neighbours/ts_'+str(i)+'.csv'
        dfd.to_csv(path,index=False) """
    #import pickle
    # Specify the file path
    #import csv
    """ csv_file_path = 'mappings/P2id_mimic.csv'

    # Write the dictionary to the CSV file
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['pid', 'index']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Write data rows
        for pid, index in P2id.items():
            writer.writerow({'pid': pid, 'index': index})
    #with open("matrices/positives.pkl", "wb") as f:
        #wr = csv.writer(f)
        #pickle.dump(hyx,f)
    #viz(he_d, P2id, "D")
    #viz(he_m, P2id, "M")
    #viz(he_r, P2id, "R")

    incidence_matrix_d = create_incidence_matrix(he_d, unique_p)
    incidence_matrix_m = create_incidence_matrix(he_m, unique_p)
    incidence_matrix_r = create_incidence_matrix(he_r, unique_p)
    incidence_matrix_a = create_incidence_matrix(he_a, unique_p)
    return incidence_matrix_m, incidence_matrix_d, incidence_matrix_r,incidence_matrix_a """
    #return 1,2,3

main()

"""for i in range(len(x)):
    print(x[i].shape)
    xp=torch.from_numpy(x[i])
    print(xp.size())
    torch.save(xp, 'matrices/d_incidence_matrix_'+str(i)+'.pt')
for i in range(len(y)):
    print(y[i].shape)
    xp=torch.from_numpy(y[i])
    print(xp.size())
    torch.save(xp, 'matrices/m_incidence_matrix_'+str(i)+'.pt')
for i in range(len(z)):
    print(z[i].shape)
    xp=torch.from_numpy(z[i])
    print(xp.size())
    torch.save(xp, 'matrices/r_incidence_matrix_'+str(i)+'.pt')

for i in range(len(a)):
    print(a[i].shape)
    xp=torch.from_numpy(a[i])
    print(xp.size())
    #torch.save(xp, 'matrices/a_incidence_matrix_'+str(i)+'.pt')
"""