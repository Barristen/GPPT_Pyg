import torch as th
import torch.nn as nn
import torch.functional as F
# import dgl
# import dgl.nn as dglnn
import sklearn.linear_model as lm
import sklearn.metrics as skm
import tqdm
import torch, gc

from torch_geometric.nn import SAGEConv, global_add_pool
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, classes, n_layers, activation, dropout, aggregator_type='gcn'):
        super(SAGE, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.classes = classes
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(SAGEConv(in_feats, n_hidden, normalize=False, aggr=aggregator_type))
            for i in range(1, n_layers - 1):
                self.layers.append(SAGEConv(n_hidden, n_hidden, normalize=False, aggr=aggregator_type))
            self.layers.append(SAGEConv(n_hidden, n_classes, normalize=False, aggr=aggregator_type))
        else:
            self.layers.append(SAGEConv(in_feats, n_classes, normalize=False, aggr=aggregator_type))
        self.fc = nn.Linear(n_hidden, classes)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.dropout(x)
        for l, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if l != self.n_layers - 1:
                x = self.activation(x)
                x = self.dropout(x)
        self.embedding_x = x
        self.pre = self.fc(x)
        return self.pre

    def forward_smc(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.dropout(x)
        for l, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if l != self.n_layers - 1:
                x = self.activation(x)
                x = self.dropout(x)
        self.embedding_x = x
        return x

    def inference(self, loader, device):
        for l, layer in enumerate(self.layers):
            ys = []
            for batch in loader:
                batch = batch.to(device)
                x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
                x = layer(x, edge_index)
                if l != self.n_layers - 1:
                    x = self.activation(x)
                    x = self.dropout(x)
                y = global_add_pool(x, batch_idx)
                ys.append(y.cpu())
            x = torch.cat(ys, dim=0)
        return x


def compute_acc_unsupervised(emb, labels, train_nids, val_nids, test_nids):
    """
    Compute the accuracy of prediction given the labels.
    """
    emb = emb.cpu().numpy()
    labels = labels.cpu().numpy()
    train_nids = train_nids.cpu().numpy()
    train_labels = labels[train_nids]
    val_nids = val_nids.cpu().numpy()
    val_labels = labels[val_nids]
    test_nids = test_nids.cpu().numpy()
    test_labels = labels[test_nids]

    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)

    lr = lm.LogisticRegression(multi_class='multinomial', max_iter=10000)
    lr.fit(emb[train_nids], train_labels)

    pred = lr.predict(emb)
    f1_micro_eval = skm.f1_score(val_labels, pred[val_nids], average='micro')
    f1_micro_test = skm.f1_score(test_labels, pred[test_nids], average='micro')
    return f1_micro_eval, f1_micro_test
