import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import argparse
import tqdm
import random
import os
import torch
import utils
from model import SAGE, compute_acc_unsupervised as compute_acc
from negative_sampler import NegativeSampler
import torch
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.loader import NeighborSampler
from torch_geometric.utils import negative_sampling
from torch_geometric.datasets import Planetoid

class GraphBinaryCrossEntropyLoss(nn.Module):
    def forward(self, z, pos_edge_index, neg_edge_index):
        # Compute the dot product for positive samples
        # z 应该是本次训练批次所需的所有节点的特征
        pos_score = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
        
        # Negative sampling for negative samples
        neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)

        scores = torch.cat([pos_score, neg_score], dim=0)
        labels = torch.cat([torch.ones(pos_score.size(0)), torch.zeros(neg_score.size(0))], dim=0)
        
        # Compute the binary cross entropy loss
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        
        return loss
    
def my_load_data(dataset):
    if dataset in ['Cora', 'CiteSeer']:
        dataset = Planetoid(root='/tmp/'+dataset, name=dataset)
        data = dataset[0]

        features = data.x
        labels = data.y
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        in_feats = features.size(1)
        n_classes = len(torch.unique(labels))
        n_edges = data.edge_index.size(1) // 2

        return data, features, labels, train_mask, val_mask, test_mask, in_feats, n_classes, n_edges
    
def evaluate(model, g, nfeat, labels, train_nids, val_nids, test_nids, device):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        # single gpu
        if isinstance(model, SAGE):
            pred = model.inference(g, nfeat, device, args.batch_size, args.num_workers)
        # multi gpu
        else:
            pred = model.module.inference(g, nfeat, device, args.batch_size, args.num_workers)
    model.train()
    
    return compute_acc(pred, labels, train_nids, val_nids, test_nids)

def smc_evaluate(model, g, nfeat, labels, train_nids, val_nids, test_nids, device):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    device : The GPU device to evaluate on.
    """
    
    model.eval()
    with th.no_grad():
        # single gpu
        if isinstance(model, SAGE):
            pred = model.inference(g, nfeat, device, args.batch_size, args.num_workers)
        # multi gpu
        else:
            pred = model.module.inference(g, nfeat, device, args.batch_size, args.num_workers)
    
    return pred

#### Entry point
def run( device, Data):
    # Unpack data

    train_mask, val_mask, test_mask, n_edges, data = Data
    print('data',data)

    # g = dgl.add_self_loop(g)
    nfeat = data.x
    labels = data.y
    in_feats = nfeat.shape[1] #每个点3703的维度值

    train_nid = th.LongTensor(np.nonzero(train_mask)).squeeze()
    val_nid = th.LongTensor(np.nonzero(val_mask)).squeeze()
    test_nid = th.LongTensor(np.nonzero(test_mask)).squeeze()

    
    model = SAGE(in_feats, n_hidden=128, n_classes=2, n_layers=2, activation = F.relu, dropout=0.5, aggregator_type='mean')
    #print(model)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    # 创建邻居采样器
    train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask, sizes=[25,10], batch_size=4096, shuffle=True, num_workers=12)
    # Training loop
    avg = 0
    iter_pos = []
    iter_neg = []
    iter_d = []
    iter_t = []
    best_eval_acc = 0
    best_test_acc = 0

    for epoch in range(100):

        tic = time.time()
        # Loop over the dataloader to sample the computation dependency graph as a list of blocks.
        tic_step = time.time()
        
        for step, (batch_size, n_id, adjs) in enumerate(train_loader):
            optimizer.zero_grad()
            adjs = [adj.to(device) for adj in adjs]
            # adjs：第L层到第1层采样结果的list
            # edge_index：采样得到的bipartite子图中source节点到target节点的边
            # e_id：edge_index的边在原始大图中的IDs
            # size：bipartite子图的shape

            x = data.x[n_id].to(device)
            out = model(x,adjs)
            #model(x[n_id], adjs)传入了所有bipartite子图的节点特征和边信息
            print('first stage')
            print('out', out.shape)
            # out是模型基于输入特征和图结构计算出的节点嵌入
            # 结合负采样来定义邻居采样器和数据加载器

            # 这里取最后一个adjs元素的边索引，因为它通常是最接近原始输入节点的层
            edge_index, _, size = adjs[-1]
            # 从返回的size中获取目标节点的数量，这通常是你的小批量大小
            src, dst = edge_index
            # 我们只关心小批量中的目标节点，因此使用size[1]来获取它们的索引
            target_nodes = n_id[:size[1]]
            # 正样本边是在当前批次中所有节点的一个子集中采样得到的
            pos_edge_index = edge_index[:, dst < size[1]]
            
            loss_fn = GraphBinaryCrossEntropyLoss()

            # Compute the loss
            loss = loss_fn(out, pos_edge_index, neg_edge_index)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            d_step = time.time()

            t = time.time()
            print('second stage')
            pos_edges = len(pos_edge_index)
            neg_edges = len(neg_edge_index)
            iter_pos.append(pos_edges / (t - tic_step))
            iter_neg.append(neg_edges / (t - tic_step))
            iter_d.append(d_step - tic_step)
            iter_t.append(t - d_step)
            if step % 20 == 0:
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print('[{}]Epoch {:05d} | Step {:05d} | Loss {:.4f} | Speed (samples/sec) {:.4f}|{:.4f} | Load {:.4f}| train {:.4f} | GPU {:.1f} MB'.format(
                    proc_id, epoch, step, loss.item(), np.mean(iter_pos[3:]), np.mean(iter_neg[3:]), np.mean(iter_d[3:]), np.mean(iter_t[3:]), gpu_mem_alloc))
            tic_step = time.time()
            
            if step % args.eval_every == 0 and proc_id == 0:
                eval_acc, test_acc = evaluate(model, g, nfeat, labels, train_nid, val_nid, test_nid, device)#############################################
                print('Eval Acc {:.4f} Test Acc {:.4f}'.format(eval_acc, test_acc))
                if eval_acc > best_eval_acc:
                    best_eval_acc = eval_acc
                    best_test_acc = test_acc
                print('Best Eval Acc {:.4f} Test Acc {:.4f}'.format(best_eval_acc, best_test_acc))

        toc = time.time()
        if proc_id == 0:
            print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
    print(model)
    th.save(model.state_dict(), './data_smc/'+dataset+'_model_'+args.file_id+'.pt')
    # m_state_dict = torch.load('./data_smc/'dataset+'_model_'+args.file_id+'.pt')####
    model.load_state_dict(m_state_dict)####
    #print("aaa")
    pre=smc_evaluate(model, g, nfeat, labels, train_nid, val_nid, test_nid, device)
    res=pre.detach().clone().cpu().data.numpy()
    res=pd.DataFrame(res)
    res.to_csv('./data_smc/'+dataset+'_feat_'+args.file_id+'.csv',header=None,index=None)
    print("##########\n",compute_acc(pre.detach().clone(),labels, train_nid, val_nid, test_nid))
    
    if proc_id == 0:
        print('Avg epoch time: {}'.format(avg / (epoch - 4)))
    
    return model

def main(device):
    dataset = 'CiteSeer'
    data,features,labels,train_mask,val_mask,test_mask,in_feats,n_classes,n_edges=my_load_data(dataset)

    Data = train_mask, val_mask, test_mask, n_edges, data

    run(device, Data)


if __name__ == '__main__':
    print("PyTorch version:", torch.__version__)

    if torch.cuda.is_available():
        print("CUDA is available")
        print("CUDA version:", torch.version.cuda)
        device = torch.device("cuda")
    else:
        print("CUDA is not available")
        device = torch.device("cpu")

    print(device)
    main(device)
