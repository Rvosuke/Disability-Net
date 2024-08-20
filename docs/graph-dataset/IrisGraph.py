# IrisGraph.py
import os
import torch
import lingam
import graphviz
import numpy as np
import pandas as pd
import torch.nn.functional as F
from lingam.utils import make_dot
from torch_geometric.nn import GCNConv
from sklearn.datasets import load_iris
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset


np.set_printoptions(precision=2, suppress=True)
np.random.seed(827)

iris = load_iris()
data = iris['data']
feature_names = iris['feature_names']
df = pd.DataFrame(data, columns=feature_names)

model = lingam.DirectLiNGAM()
model.fit(df)

adj_matrix = model.adjacency_matrix_


class IrisGraph(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(IrisGraph, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['iris.data']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        np.savetxt(os.path.join(self.raw_dir, 'iris.data'), iris.data, delimiter=',')

    def process(self):
        # 读取数据
        data_path = os.path.join(self.raw_dir, 'iris.data')
        features = np.loadtxt(data_path, delimiter=',')
        
        data_list = []
        
        # 利用因果邻接矩阵对每一个样本创建一个图
        for i in range(features.shape[0]):
            edge_index = np.argwhere(adj_matrix != 0).T  # 获取非零元素的索引
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            
            node_features = torch.tensor(features[i], dtype=torch.float).view(-1, 1)
            
            data = Data(x=node_features, edge_index=edge_index)
            data_list.append(data)
        
        # 预处理函数应用
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# 使用数据集
dataset = IrisGraph(root='./IrisGraph')
print(dataset.y)  # 输出第一个图的数据
