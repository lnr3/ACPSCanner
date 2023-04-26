import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import pandas as pd
import numpy as np
from torch_geometric.nn import GATConv
import scipy.sparse as sp
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn
from torch_geometric.data import DataLoader

from utility import metrics, readfile, readesm, readpdb, readstructure, StructureDataParser
from sklearn.metrics import roc_curve, accuracy_score, auc, matthews_corrcoef, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

def getMetrics(y_true, y_pred, y_proba):
    ACC = accuracy_score(y_true, y_pred)
    MCC = matthews_corrcoef(y_true, y_pred)
    CM = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = CM.ravel()
    Sn = tp / (tp + fn)
    Sp = tn / (tn + fp)
    FPR, TPR, thresholds_ = roc_curve(y_true, y_proba)
    AUC = auc(FPR, TPR)

    Results = np.array([ACC, MCC, Sn, Sp, AUC]).reshape(-1, 5)
    Metrics_ = pd.DataFrame(Results, columns=["ACC", "MCC", "Sn", "Sp", "AUC"])

    return Metrics_

sequence, label = readfile('testing.fasta')
label = np.array(label)
single = label[:, 8]

embedding = pd.read_csv("testing.csv", header=0)
property = embedding.iloc[:, 0:188].values.tolist()
prot = [p for p in property]

dists = readpdb('testpdb', 'testing.fasta')
array = []
for dist in dists:
    adjacency = np.zeros([50, 50])
    n = len(dist)
    for i in range(0, n):
        for j in range(0, n):
            if dist[i][j] < 8:
                adjacency[i][j] = 1
    array.append(adjacency)

esm = readesm('testesm', 'testing.fasta')
node = [item for item in esm]

structure = readstructure(sequence, 'testing.txt')
residue = []
for i in range(len(label)):
    cur = []
    for j in range(0, 250, 5):
        info = [0] * 24
        if structure[i][j] == 'A':
            info[0] = 1
        elif structure[i][j] == 'R':
            info[1] = 1
        elif structure[i][j] == 'N':
            info[2] = 1
        elif structure[i][j] == 'D':
            info[3] = 1
        elif structure[i][j] == 'C':
            info[4] = 1
        elif structure[i][j] == 'Q':
            info[5] = 1
        elif structure[i][j] == 'E':
            info[6] = 1
        elif structure[i][j] == 'G':
            info[7] = 1
        elif structure[i][j] == 'H':
            info[8] = 1
        elif structure[i][j] == 'I':
            info[9] = 1
        elif structure[i][j] == 'L':
            info[10] = 1
        elif structure[i][j] == 'K':
            info[11] = 1
        elif structure[i][j] == 'M':
            info[12] = 1
        elif structure[i][j] == 'F':
            info[13] = 1
        elif structure[i][j] == 'P':
            info[14] = 1
        elif structure[i][j] == 'S':
            info[15] = 1
        elif structure[i][j] == 'T':
            info[16] = 1
        elif structure[i][j] == 'W':
            info[17] = 1
        elif structure[i][j] == 'Y':
            info[18] = 1
        elif structure[i][j] == 'V':
            info[19] = 1

        info[20] = structure[i][j + 1]
        info[21] = structure[i][j + 2]
        info[22] = structure[i][j + 3]
        info[23] = structure[i][j + 4]
        cur.append(info)
    residue.append(cur)

for i in range(len(label)):
    node[i] = torch.cat((node[i], torch.FloatTensor(residue[i])), 1)
    # node[i] = torch.FloatTensor(residue[i])

dataset = []
for i in range(len(label)):
    edge_index_temp = sp.coo_matrix(array[i])
    indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
    edge_index = torch.LongTensor(indices)

    x = node[i]
    x = torch.FloatTensor(x)
    x = x.squeeze(0)

    phy = prot[i]
    phy = torch.FloatTensor(phy)
    phy = phy.unsqueeze(0)

    y = torch.FloatTensor(np.array([1, 0]) if single[i] == 0 else np.array([0, 1]))
    y = y.unsqueeze(0)

    data = Data(x=x, edge_index=edge_index, phy=phy, y=y)
    dataset.append(data)

test_dataset = dataset

class Net(torch.nn.Module):
    """构造GCN模型网络"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GATConv(1280 + 24, 256, heads=3)
        self.conv2 = GATConv(256 * 3, 256)
        self.fc1 = nn.Sequential(
            nn.Linear(188, 128),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Dropout(p=0.2),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256 + 128, 192),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(192, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, data):
        x, edge_index, phy, batch = data.x, data.edge_index, data.phy, data.batch

        x = self.conv1(x, edge_index)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        x = pyg_nn.global_mean_pool(x, batch)

        mid = self.fc1(phy)
        y = self.fc2(torch.cat((x, mid), 1))

        # x = self.fc(x)

        # return F.log_softmax(x, dim=1)
        return y

from sklearn.metrics import accuracy_score
learning_rate = 0.001

def evaluate(loader):
    model.eval()
    pred = []
    label = []
    with torch.no_grad():
        for data in loader:
            pred.extend(model(data).numpy().tolist())
            label.extend(data.y.numpy().tolist())
    return pred, label

loaders = DataLoader(test_dataset, batch_size=20, shuffle=False)

probability = np.zeros((len(label), 9), dtype=float)
predictions = np.zeros((len(label), 9), dtype=int)

for i in range(1, 10):
    model = torch.load('gat_{}_{}.pkl'.format(learning_rate, i))
    pred, cur = evaluate(loaders)
    for j in range(len(cur)):
        probability[j][i - 1] = pred[j][1]
        if pred[j][1] > 0.5:
            predictions[j][i - 1] = 1
    print(getMetrics(label[:, i - 1], predictions[:, i - 1], probability[:, i - 1]))
    file = open('GAT{}.txt'.format(i), 'w')
    for val in probability[:, i - 1]:
        file.write(str(val))
        file.write('\n')

print(metrics(label, predictions, probability))

# file = open('GAT.txt', 'w')
# for sub in pred:
#     for val in sub:
#         file.write(str(val) + ' ')
#     file.write('\n')