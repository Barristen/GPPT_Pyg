import argparse
import random
import torch
import dgl
# from dgl import DGLGraph
# from dgl.data import register_data_args, load_data
# from ogb.nodeproppred  import DglNodePropPredDataset
import numpy as np
import os
from sklearn.metrics import accuracy_score
from get_args import get_my_args
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
def my_load_data(args):
    if args.dataset=='Cora' or args.dataset=='Citeseer' :
        dataset = Planetoid(root='/tmp/' + args.dataset, name=args.dataset, transform=NormalizeFeatures())
        data = dataset[0]
        features = data.x
        labels = data.y
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        
        in_feats = features.size(1)
        n_classes = dataset.num_classes
        n_edges = data.edge_index.size(1) // 2
        # n_edges = data.graph.number_of_edges()
    else:
        data=None
        features=None
        labels=None
        train_mask=None
        val_mask=None
        test_mask=None
        in_feats=None
        n_classes=None
        n_edges=None
    return data,features,labels,train_mask,val_mask,test_mask,in_feats,n_classes,n_edges

def evaluate(model, graph, nid, batch_size, device,sample_list):
    sampler = dgl.dataloading.MultiLayerNeighborSampler(sample_list)
    valid_dataloader = dgl.dataloading.NodeDataLoader(graph, nid.int(), sampler,batch_size=batch_size,shuffle=False,drop_last=False,num_workers=0,device=device)
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for input_nodes, output_nodes, mfgs in valid_dataloader:
            inputs = mfgs[0].srcdata['feat']
            labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
            predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        accuracy = accuracy_score(labels, predictions)
    return accuracy
    
def constraint(device,prompt):
    if isinstance(prompt,list):
        sum=0
        for p in prompt:
            sum=sum+torch.norm(torch.mm(p,p.T)-torch.eye(p.shape[0]).to(device))
        return sum/len(prompt)
    else:
        return torch.norm(torch.mm(prompt,prompt.T)-torch.eye(prompt.shape[0]).to(device))
            
def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def get_init_info(args):
    g,features,labels,train_mask,val_mask,test_mask,in_feats,n_classes,n_edges=my_load_data(args)
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))
    
    if args.gpu < 0:
        device='cpu'
    else:
        device='cuda:'+str(args.gpu)
        torch.cuda.set_device(args.gpu)
        
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    print("use cuda:", args.gpu)

    train_nid = train_mask.nonzero().squeeze()
    val_nid = val_mask.nonzero().squeeze()
    test_nid = test_mask.nonzero().squeeze()
    g = dgl.remove_self_loop(g)
    n_edges = g.number_of_edges()
    if args.gpu >= 0:
        g = g.int().to(args.gpu)
    return g,features,labels,in_feats,n_classes,n_edges,train_nid,val_nid,test_nid,device

def node_mask(train_mask,mask_rate):
    mask_rate=int(mask_rate*10)
    count=0
    for i in range(train_mask.shape[0]):
        if train_mask[i]==True:
            count=count+1
            if count<=mask_rate:
                train_mask[i]=False
                count=count+1
            if count==10:
                count=0
    return train_mask

if __name__ == '__main__':
    args = get_my_args()
    print(args)
    info=my_load_data(args)
    g=info[0]
    labels=torch.sparse.sum(g.adj(),1).to_dense().int().view(-1,)
    print(labels)
    li=list(set(labels.numpy()))
    for i in range(labels.shape[0]):
        labels[i]=li.index(labels[i])
    print(set(labels.numpy()))
    #print(info)