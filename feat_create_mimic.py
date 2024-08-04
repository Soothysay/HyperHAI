import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
df1=pd.read_csv('data_MIMIC/CDI_EMR.csv',parse_dates=['date'])
df2=pd.read_csv('data_MIMIC/CDIx_EMR.csv',parse_dates=['date'])
#df=df[df['timestamp']>=4]
#print(df.columns)
df1.drop(['cdate'],axis=1,inplace=True)
df=pd.concat([df1,df2],axis=0)
print(df.columns)
df['date'] = pd.to_datetime(df['date'])
df['timestamp']=(df['date'] - pd.to_datetime('2128-01-01')).dt.days
# Addition
#df['timestamp']=df['timestamp']+2
df['user_id']=df['pid']
df['gender'] = df['gender'].astype(int)
df['prev_visit'] = df['prev_visit'].astype(int)
pat_feat=df[['user_id','timestamp','los','age','gender','prev_visit','insurance','marital_status']]
# Create dummies for 'insurance' and 'marital_status' columns
pat_feat = pd.get_dummies(pat_feat, columns=['insurance', 'marital_status'], drop_first=True)
#pat_feat.rename({'pid':'user_id'},inplace=True)
print(pat_feat.columns)
pat_feat=pat_feat.drop_duplicates(subset=['user_id','timestamp'])
pat_feat.to_csv('data_MIMIC/pat_feat.csv',index=False)

feat=pd.read_csv('data_MIMIC/pat_feat.csv')
labels=pd.read_csv('data_MIMIC/MICU1.csv')
mapping=pd.read_csv('mappings/P2id_mimic.csv')
mapping=mapping[['index','pid']]
id2P=mapping.set_index('index').T.to_dict('list')
for i in id2P.keys():
    id2P[i]=id2P[i][0]
P2id={}
for i in id2P.keys():
    P2id[id2P[i]]=i
mapping['user_id']=mapping['pid']
mapping=mapping.drop(['pid'],axis=1)

for t in range(94):
    pf_t=feat[feat['timestamp']==t]
    pf_t=pf_t.drop(['timestamp'],axis=1)
    ordered_feats=mapping.merge(pf_t,on='user_id',how='left')
    
    ordered_feats=ordered_feats.drop(['user_id','index'],axis=1)
    ordered_feats=ordered_feats.fillna(0)
    #print(ordered_feats.head(5))
    feat_t=torch.tensor(ordered_feats.values)
    path='feat_mimic/feat_tensors'+str(t)+'.pt'
    print(feat_t.size())
    torch.save(feat_t,path)
