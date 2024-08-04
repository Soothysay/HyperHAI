import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import torch.nn as nn
from tqdm import tqdm
#from models import spatial_agg
#from models import temporal_agg
#from models import Decoder1,Decoder2
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from utils.function import *
import time
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import dhg
from dhg.nn import HGNNConv
from dhg import Hypergraph
from tqdm import tqdm
from pathlib import Path
import pickle
from argparse import ArgumentParser
import random
from dhg.nn import JHConv
from dhg.models import DHCF
from sklearn.metrics import average_precision_score
parser = ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--label_filepath', type=str, default='data_MIMIC/CDI1.csv', help='path to labels.csv')
parser.add_argument('--data_name', type=str, default='mimic', help='name of dataset')
parser.add_argument('--model_name', type=str, default='HGCN', help='model name')
parser.add_argument('--window_size', type=int, default=4, help='window size') #1
parser.add_argument('--alpha', type=float, default=0.003, help='CL balancing hyperparameter')
parser.add_argument('--beta', type=float, default=0.1, help='Temporal Consistency hyperparameter')
parser.add_argument('--p2id_filepath', type=str, default='mappings/P2id_mimic.csv', help='path to p2id.csv')
parser.add_argument('--feat_dir', type=str, default='feat_mimic', help='path to feat_tensors')
parser.add_argument('--in_dim', type=int, default=9, help='input dimension')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--graph_savedir', type=str, default=None, help='path to hypergraph files')
parser.add_argument('--hypergraph_dir1', type=str, default='hygmimic1', help='path to hypergraph files for first view')
parser.add_argument('--hypergraph_dir2', type=str, default='hygmimic2', help='path to hypergraph files for second view')
parser.add_argument('--total_timesteps', type=int, default=94, help='total timesteps')
parser.add_argument('--split_timestep_val', type=int, default=50, help='split timestep for validation')#50 37
parser.add_argument('--split_timestep_test', type=int, default=70, help='split timestep for test')#70 50
args = parser.parse_args()
# set seeds for all random number generators
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
dhg.random.set_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

savedir = (
        args.data_name + "~" + args.model_name + '~seed' + str(seed)  
        + '~data_' + args.label_filepath.split('/')[-1].split('.')[0] 
        + '~ws' + str(args.window_size)
        + '~' + time.strftime('%Y%m%d-%H%M%S')
)

Path(savedir).mkdir(parents=True, exist_ok=True)
print(f'Saving to {savedir}', flush=True)


class HGCN(torch.nn.Module):
    def __init__(self,inlen,hiddenlen):
        super().__init__()
        # self.conv1 = GCNConv(inlen,hiddenlen)
        # self.conv2 = GCNConv(hiddenlen,hiddenlen)
        # self.hgnn = dhg.HGNN(inlen, hiddenlen, num_classes=2, use_bn=False, drop_rate=0.0)
        self.conv1 = HGNNConv(inlen, hiddenlen, use_bn=False, drop_rate=0.0)
        self.conv2 = HGNNConv(hiddenlen, hiddenlen, use_bn=False, drop_rate=0.0)

    def forward(self, feat, hg):
        # feat: N x inlen
        # hg: hg (dhg.Hypergraph) – The hypergraph structure that contains vertices.
        x = self.conv1(feat, hg) # already has ReLU
        #x = self.conv2(x, hg)
        return x

class JHGCN(torch.nn.Module):
    def __init__(self,inlen,hiddenlen):
        super().__init__()
        self.input_transform = nn.Linear(inlen, hiddenlen)
        self.conv1 = JHConv(hiddenlen, hiddenlen, use_bn=False, drop_rate=0.0)
        self.conv2 = JHConv(hiddenlen, hiddenlen, use_bn=False, drop_rate=0.0)

    def forward(self, feat, hg):
        # feat: N x inlen
        # hg: hg (dhg.Hypergraph) – The hypergraph structure that contains vertices.
        feat = self.input_transform(feat)
        feat = F.leaky_relu(feat, negative_slope=0.2)
        x = self.conv1(feat, hg)
        x = self.conv2(feat, hg)
        return x
 
class GCN(torch.nn.Module):
    def __init__(self,inlen,hiddenlen):
        super().__init__()
        self.conv1 = GCNConv(inlen,hiddenlen)
        self.conv2 = GCNConv(hiddenlen,hiddenlen)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        return x


class Decoder(nn.Module):
    def __init__(self,input_size,embed_size):
        super(Decoder, self).__init__()
        #self.device=device
        self.i_s=input_size
        self.e_s=embed_size
        self.relu=nn.ReLU()
        self.l1=nn.Linear(self.i_s,self.i_s//2)
        self.l2=nn.Linear(self.i_s//2,self.e_s)
        self.l3=nn.Linear(self.e_s,1)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        x=self.l1(x)
        x=self.relu(x)
        x=self.l2(x)
        x=self.relu(x)
        x=F.normalize(x)
        x=self.l3(x)
        x=self.sigmoid(x)
        return x



def create_graphs(feats, filepath, P2id, savedir='GCN'):
    #start = time.time()
    interactions1=pd.read_csv(filepath)
    interactions=pd.merge(interactions1,P2id[['user_id','index']],on='user_id')
    for t in range(TOTAL_T):
        #print(t)
        x=feats[t]
        x=x.type(torch.FloatTensor)
        timestampted=interactions[interactions['timestamp']==t]
        unique_pat=timestampted['index'].unique().tolist()
        overall=[]
        for p in unique_pat:
            #pin=P2id[P2id['user_id']==p]['index'].tolist()[0]
            items=timestampted[timestampted['index']==p]['item_id'].unique().tolist()
            for i in items:
                otheruser=timestampted[timestampted['item_id']==i]['index'].unique().tolist()
                for u in otheruser:
                    if p!=u:
                        edge=[]
                        edge.append(p)
                        edge.append(u)
                        overall.append(edge)
        edge_index=torch.tensor(overall,dtype=torch.long)
        data = Data(x=x, edge_index=edge_index.t().contiguous())
        path=str(Path(savedir)/'Graph_'+str(t)+'.pt')
        torch.save(data,path)

def load_hyperedges(loaddir='hyg'):
    hyper_edges = {'d': [], 'm': [], 'r': []}
    for t in range(TOTAL_T):
        hyper_edges_d_t = []
        with open(str(Path(loaddir)/f'd{t}.csv'), 'r') as f:
            for line in f:
                hyper_edges_d_t.append([int(x) for x in line.strip().split(',')])
        hyper_edges['d'].append(hyper_edges_d_t)
        #
        hyper_edges_m_t = []
        with open(str(Path(loaddir)/f'm{t}.csv'), 'r') as f:
            for line in f:
                hyper_edges_m_t.append([int(x) for x in line.strip().split(',')])
        hyper_edges['m'].append(hyper_edges_m_t)
        #
        hyper_edges_r_t = []
        with open(str(Path(loaddir)/f'r{t}.csv'), 'r') as f:
            for line in f:
                hyper_edges_r_t.append([int(x) for x in line.strip().split(',')])
        hyper_edges['r'].append(hyper_edges_r_t)

    return hyper_edges


device = torch.device("cuda:5")
label_filepath = args.label_filepath
p2id_filepath = args.p2id_filepath
feat_dir = args.feat_dir
graph_savedir = args.graph_savedir
hypergraph_dir1 = args.hypergraph_dir1
hypergraph_dir2 = args.hypergraph_dir2
P2id=pd.read_csv(p2id_filepath)
labels=pd.read_csv(label_filepath)
#labels['timestamp']=labels['timestamp']-2
roc_val=0.0
roc_test=0.0
prauc_val=0.0
prauc_test=0.0

NUM_OBJS= P2id['pid'].nunique() #6466

if 'mimic' in p2id_filepath:
    assert NUM_OBJS == 1155
else:
    assert NUM_OBJS == 6466
    
TOTAL_T = args.total_timesteps #90
SPLIT_T_val = args.split_timestep_val #60
SPLIT_T_test = args.split_timestep_test #60
WINDOW_SIZE = args.window_size #4

if args.model_name == 'GCN':
    assert WINDOW_SIZE == 1


pats = P2id['pid'].tolist()
P2id['user_id'] = P2id['pid']
train_labels=[]
train_take=[]
for t in tqdm(range(TOTAL_T), desc='Processing timestamps'):
    pox = []
    poy = []
    train_t = labels[(labels['timestamp'] == t)]
    for p in pats:
        train_p = train_t[train_t['user_id'] == p]
        if len(train_p) > 0:
            train_take.append(True)
            lab = train_p['label'].tolist()[0]
            pox.append(int(lab))
        else:
            train_take.append(False)
        #test_p = test_t[test_t['user_id'] == p]
    train_labels.append(pox)

feats=[]
for i in range(TOTAL_T):
    feat_t = torch.load(str(Path(feat_dir) /f'feat_tensors{i}.pt'))
    # cast to float32
    feat_t = feat_t.float()
    feats.append(feat_t)

hyper_edges1 = load_hyperedges(loaddir=hypergraph_dir1)
hyper_edges2 = load_hyperedges(loaddir=hypergraph_dir2)

# create model 
in_dim = feats[0].shape[-1]
print('input dimension:', in_dim)

base_hidden_dim = 32#32 
dec_outdim = 128
NUM_EMB_TYPES = 3 # medication, doctor, room
hidden_dim = base_hidden_dim * NUM_EMB_TYPES
if args.model_name in 'HGCN':
    model=HGCN(in_dim, hidden_dim // NUM_EMB_TYPES)
elif args.model_name == 'JHGCN':
    model=JHGCN(in_dim, hidden_dim // NUM_EMB_TYPES)
elif args.model_name == 'GCN':
    model = GCN(in_dim, hidden_dim)

model.to(device)

if WINDOW_SIZE==1:
    ffn=Decoder(hidden_dim, dec_outdim)
else:
    attn=nn.MultiheadAttention(hidden_dim,4,batch_first=True)
    attn=attn.to(device)
    ffn=Decoder((hidden_dim*WINDOW_SIZE), dec_outdim)

ffn=ffn.to(device)
#
if WINDOW_SIZE==1:
    optimizer=torch.optim.Adam(list(model.parameters())
                               +list(ffn.parameters()),lr=0.0001) #0.01
else:
    optimizer=torch.optim.Adam(list(model.parameters())
                               +list(ffn.parameters())
                               +list(attn.parameters()),lr=0.0001)
loss = nn.BCELoss()
#feats=[]
#prevs=[]

epochs=args.epochs
roc=0.0
tr_loss=[]
best=[]
rocs = []

pbar = tqdm(range(epochs))
for e in pbar:
    train_preds=[]
    train_labls=[]
    test_preds=[]
    test_labls=[]
    val_preds=[]
    val_labls=[]
    prevs=[]
    prevs1=[]
    saves=[]
    saves1=[]
    optimizer.zero_grad()
    io=0
    total_loss=0
    best=False
    #j=NUM_OBJS*60
    for t in range(TOTAL_T):
        #
        if t<=SPLIT_T_val: 
            model.train()
            if WINDOW_SIZE>1: attn.train()
            ffn.train()
        else: 
            model.eval()
            if WINDOW_SIZE>1: attn.eval()
            ffn.eval()

        feat_t = feats[t].to(device)
        if args.model_name in ['HGCN', 'JHGCN']:
            # create hypergraph
            hg_d_t1 = Hypergraph(NUM_OBJS, hyper_edges1['d'][t] if len(hyper_edges1['d'][t]) > 0 else [[]]).to(device)
            hg_m_t1 = Hypergraph(NUM_OBJS, hyper_edges1['m'][t] if len(hyper_edges1['m'][t]) > 0 else [[]]).to(device)
            hg_r_t1 = Hypergraph(NUM_OBJS, hyper_edges1['r'][t] if len(hyper_edges1['r'][t]) > 0 else [[]]).to(device)
            hg_d_t2 = Hypergraph(NUM_OBJS, hyper_edges2['d'][t] if len(hyper_edges2['d'][t]) > 0 else [[]]).to(device)
            hg_m_t2 = Hypergraph(NUM_OBJS, hyper_edges2['m'][t] if len(hyper_edges2['m'][t]) > 0 else [[]]).to(device)
            hg_r_t2 = Hypergraph(NUM_OBJS, hyper_edges2['r'][t] if len(hyper_edges2['r'][t]) > 0 else [[]]).to(device)
            #
            out_d1 = model(feat_t, hg_d_t1)
            out_m1 = model(feat_t, hg_m_t1)
            out_r1 = model(feat_t, hg_r_t1)
            out = torch.cat([out_d1, out_m1, out_r1], dim=1)
            #out = torch.cat([out_m1, out_r1], dim=1)
            out_d2 = model(feat_t, hg_d_t2)
            out_m2 = model(feat_t, hg_m_t2)
            out_r2 = model(feat_t, hg_r_t2)
            out2 = torch.cat([out_d2, out_m2, out_r2], dim=1)
            #out2 = torch.cat([out_m2, out_r2], dim=1)
        elif args.model_name == 'GCN':
            path=str(Path(graph_savedir) / f'Graph_{t}.pt')
            graph_t=torch.load(path)
            graph_t=graph_t.to(device)
            out = model(graph_t)

        #if t>SPLIT_T_val:
        #    saves.append(out)
        #    saves1.append(out2)

        prevs.append(out)
        prevs1.append(out2)
        if t<=SPLIT_T_val:
            cl=args.alpha*InfoNCE(out,out2,T=0.6)
            #print(cl,flush=True)
            total_loss+=cl
        if t>= WINDOW_SIZE - 1:

            if WINDOW_SIZE==1:
                out3 = ffn(out)
                # temporal consistency loss
                if t>0:
                    consistency_loss=torch.linalg.vector_norm((out-prevs[(t-1)]),ord=2)
                    total_loss+=args.beta*consistency_loss
            else:
                k=t
                outx=[]
                while(k>=(t- WINDOW_SIZE + 1)):
                    outx.append(prevs[k])
                    k-=1
                assert len(outx)== WINDOW_SIZE
                outx = torch.stack(outx,1)
                outx=outx.type(torch.FloatTensor)
                outx=outx.to(device)
                out1,_ = attn(outx,outx,outx)
                out1=out1.reshape((NUM_OBJS,(WINDOW_SIZE*hidden_dim)))
                out3=ffn(out1)

        if t<=SPLIT_T_val:
            if t>= WINDOW_SIZE - 1:
                choosers=train_take[io:(io+NUM_OBJS)]
                pred_train=out3[choosers]
                tr_lab=train_labels[t]
                train_targets=torch.FloatTensor(tr_lab)
                train_targets=train_targets.to(device)
                train_preds.append(pred_train.squeeze(1))
                train_labls.append(train_targets)
            io+=NUM_OBJS
            if t==SPLIT_T_val:
                train_preds=torch.cat(train_preds,0)
                train_labls=torch.cat(train_labls,0)
                total_loss+=loss(train_preds,train_labls)
                # print('Epoch')
                # print(e)
                # print('Training Loss')
                pbar.set_description(f'Epoch {e}, Training Loss {total_loss.item()}')
                # print(total_loss.item())
                tr_loss.append(total_loss.item())
                total_loss.backward()
                optimizer.step()
        elif t<=SPLIT_T_test:
            choose_val=train_take[io:(io+NUM_OBJS)]
            pred_val=out3[choose_val]
            val_lab=train_labels[t]
            val_targets=torch.FloatTensor(val_lab)
        
            val_targets=val_targets.to(device)
            val_preds.append(pred_val.squeeze(1))
            val_labls.append(val_targets)
            io+=NUM_OBJS
            if t==SPLIT_T_test:
                val_preds=torch.cat(val_preds,0)
                val_labls=torch.cat(val_labls,0)
                v_p=val_preds.detach().cpu().numpy()
                v_l=val_labls.detach().cpu().numpy()
                #print('Test ROC Score:')
                rc=roc_auc_score(v_l,v_p)
                rocs.append(rc)
                if e>=10:
                    if rc>roc:
                        best=True
                        epo=e
                        roc=rc
                        print('\nBetter validation ROC Score!', flush=True)
                        print(roc, flush=True)
                        #best1=saves

                        # save model
                        model_state_dict = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'attn': attn.state_dict() if WINDOW_SIZE > 1 else None,
                            'ffn': ffn.state_dict(),
                            'epoch': e,
                            'train_loss': tr_loss,
                            #'best_feats': best,
                            'rocs': rocs,
                            'best_roc': roc,
                        }
                        torch.save(model_state_dict, str(Path(savedir) / f'model_state_dict_best.pt'))
        else:
            choose_test=train_take[io:(io+NUM_OBJS)]
            pred_test=out3[choose_test]
            test_lab=train_labels[t]
            test_targets=torch.FloatTensor(test_lab)
        
            test_targets=test_targets.to(device)
            test_preds.append(pred_test.squeeze(1))
            test_labls.append(test_targets)
            io+=NUM_OBJS
            if t==TOTAL_T-1:
                test_preds=torch.cat(test_preds,0)
                test_labls=torch.cat(test_labls,0)
                t_p=test_preds.detach().cpu().numpy()
                t_l=test_labls.detach().cpu().numpy()
                #print('Test ROC Score:')
                if best==True:
                    rc1=roc_auc_score(t_l,t_p)
                    print('\nTest ROC Score!', flush=True)
                    print(rc1, flush=True)
                    prc=average_precision_score(t_l,t_p)
                    print('\nTest AUPRC Score!', flush=True)
                    print(prc, flush=True)
        #for 
        #import pdb;pdb.set_trace()
    # save model 
    if (e%100==0) or (e==epochs-1):
        model_state_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'attn': attn.state_dict() if WINDOW_SIZE > 1 else None,
            'ffn': ffn.state_dict(),
            'epoch': e,
            'train_loss': tr_loss,
            'best_feats': best,
            'rocs': rocs,
            'best_roc': roc,
        }
        torch.save(model_state_dict, str(Path(savedir) / f'model_state_dict_{e}.pt'))

print(f'\nFinal Validation ROC-AUC Score: {roc}', flush=True)
print(f'\nFinal Test ROC-AUC Score: {rc1}', flush=True)
print(f'\nFinal Test AUPRC Score: {prc}', flush=True)
#print(cfm)
#print(epo)