# Graph Level Predictions on ADNI Data with Hierarchical Clustering

# 概览

蛋白质共表达图传统上被用于识别阿尔茨海默病(AD)的新生物标志物和阐明生物机制。在这个项目中,我们的目标是使用基于个体蛋白质表达配置文件的GNN来预测疾病状态。我们将使用来自阿尔茨海默病神经影像学倡议(ADNI)的脑脊液(CSF)数据。这些数据包含了大约85种蛋白的表达数据,共收集了310个受试者。我们选择这个数据集是因为ADNI是人类AD最大的纵向研究之一;它不仅包含受试者的丰富表型数据(临床信息、神经影像学、生物标志物、认知和遗传配置文件),而且数据访问政策也是最开放的。

这段代码将执行根据蛋白质表达数据进行AD预测的任务。从ADNI数据中,我们将构建一个*邻接矩阵*(基于节点对之间的双重权重中值相关),表示不同蛋白质之间在所有受试者中的相似性。蛋白质表达水平将提供节点信息,每个患者将由一个图表示。

从这里开始,我们将通过分层学习进行*图级预测*,最初具有两级汇聚。我们将使用GCN生成反映局部图结构的嵌入,然后基于ASAPool方法进行聚类。将再次在软聚类上生成嵌入,我们希望这将反映结构信息,之后是另一轮聚类,然后是最终的图级预测。

如果我们想要进行其他工作如因果图，我们可以使用类似于DirectLiNGAM->根据adj进行建图，需要注意的是不同数据集的预处理还有模型最后的评估。

```python
# 安装torch geometric，注意要保证其版本和自己的torch版本一致。
import os
if 'IS_GRADESCOPE_ENV' not in os.environ:
  !pip install torch-geometric \
  torch-sparse \
  torch-scatter \
  torch-cluster \
  -f https://pytorch-geometric.com/whl/torch-2.0.1+cpu.html
  !pip install -q git+https://github.com/snap-stanford/deepsnap.git
  !pip install ogb
import torch
print("PyTorch has version {}".format(torch.__version__))
import pandas as pd
import torch.nn.functional as F

# The PyG built-in GCNConv
from torch_geometric.nn import GCNConv

import torch_geometric.transforms as T
```

# 加载数据集

```python
url1 = 'https://raw.githubusercontent.com/sdos1/cs224w_adni_files/main/protein_adjacency_matrix.csv'
df1 = pd.read_csv(url1)
# Protein Co-Expression Dataset 被读取成为pandas.DataFrame
df1.to_csv('protein_adjacency_matrix.csv')

url2 = 'https://raw.githubusercontent.com/sdos1/cs224w_adni_files/main/final_diagnosis.csv'
df2 = pd.read_csv(url2)
# Diagnosis Dataset 被读取成为pandas.DataFrame
df2.to_csv('final_diagnosis.csv')

url3 = 'https://raw.githubusercontent.com/sdos1/cs224w_adni_files/main/log_transformed_ADNI_expression_data_with_covariates.csv'
df3 = pd.read_csv(url3)
# Patient Expression Dataset 被读取成为pandas.DataFrame
df3.to_csv('log_transformed_ADNI_expression_data_with_covariates.csv')


```

# 导入图和患者级别数据

1.   从CSV文件中加载蛋白质的邻接矩阵。
2.   使用NetworkX库从这个邻接矩阵中创建一个图。
3.   从CSV文件中加载蛋白质表达水平。
4.   从CSV文件中加载最终诊断，并将其转换为二进制分类。

```python
import numpy
import networkx as nx
import csv

# 从CSV文件中读取蛋白质的邻接矩阵，跳过第一行（标题行）和前两列（标签、序号）每种蛋白质作为一个节点。对于其他问题，也可以是因果图的邻接矩阵等
adj = numpy.loadtxt(open("protein_adjacency_matrix.csv", "rb"), delimiter=",", skiprows=1, usecols=numpy.arange(2, 53))

# 图的邻接矩阵和图的抽象数据类表示是等价的，可以二者互换，这里使用networkx，假设使用LiNGAM进行因果发现，可以直接使用make_dot来可视化。
# 使用NetworkX从邻接矩阵创建图（G），parallel_edges和create_using参数默认为None，使用邻接矩阵即可得知图的连接性，进一步可构建图的结构
G = nx.from_numpy_array(adj, parallel_edges=False, create_using=None)  # 对于此API的使用可以查阅NetworkX的文档，这里我们设置参数表示不存在重复边，使用的是无向图

# 打印和可视化图
print(G)
nx.draw(G, with_labels=True)

# 从CSV文件中读取蛋白质表达水平矩阵，跳过第一行（标题行）和前16列（可能是标签或其他元数据）,蛋白质表达水平即节点信息。
expression_mat = numpy.loadtxt(open("log_transformed_ADNI_expression_data_with_covariates.csv", "rb"), delimiter=",", skiprows=1, usecols=numpy.arange(16, 67))

# 从CSV文件中读取诊断列表，诊断结果是关于一张图全局上下文的信息
with open("final_diagnosis.csv") as file_name:
    file_read = csv.reader(file_name)
    diagnosis_list = list(file_read)

# 将诊断信息转换为二进制分类：1表示“AD”（Alzheimer's Disease），0表示其他
binary_diagnosis = []
for i in range(len(diagnosis_list)):
    if i > 0:  # 跳过标题行
        if diagnosis_list[i][1] == "AD":
            binary_diagnosis.append(1)
        else:
            binary_diagnosis.append(0)

```

# 为PyTorch预处理数据

高层目标是创建和加载一组图(拆分为测试、训练和验证集),表示每个个体不同蛋白节点的蛋白表达水平。它们将使用相同的共同图邻接矩阵。我们还包括一个基于浅编码的随机位置编码器,该编码器在所有图中都是相同的。

![1](D:\OneDrive\Works\30-8-2023\1.png)

```python
import torch_geometric.utils
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from tqdm.notebook import tqdm


x_tensor = torch.from_numpy(expression_mat).float()  # 节点矩阵
diagnosis_tensor = torch.Tensor(binary_diagnosis).long()  # 总的图向量，每个元素表示一张图
adj_tensor = torch.from_numpy(adj)  # 邻接矩阵张量化
G_convert = torch_geometric.utils.from_networkx(G)  # 将G转化为PyG的图Data抽象数据类 

positional_encoder = torch.rand(51,3).float()  # 节点位浅层编码，为了保证置换不变性，51是节点个数，3是位置编码随机数

# 归类字典，将样例分为训练集、验证集、测试集，数值要根据自己使用的数据集进行修改
split_idx = {}
split_idx['train'] = torch.tensor(numpy.arange(0, 149))
split_idx['valid'] = torch.tensor(numpy.arange(150, 299))
split_idx['test'] = torch.tensor(numpy.arange(300, 449))

train_list = []
test_list = []
valid_list = []
## 对每个图进行归类
for i in range(len(diagnosis_tensor)):
  x_yeet = x_tensor[i,:]  # 选取节点矩阵中的第i行，即选择第i个图对应的节点信息的向量，注意此处为行向量
  x_scalar = torch.t(torch.reshape(x_yeet, (1, len(x_yeet)))).float()
  x = torch.cat((x_scalar, positional_encoder), 1)
  y = diagnosis_tensor[i]
  if (i in split_idx['train']):
    train_list.append(Data(x=(x), y = y, edge_index=G_convert.edge_index, edge_attr = G_convert.weight))
    # print(train_list[i])
  if (i in split_idx['valid']):
    valid_list.append(Data(x=(x), y = y, edge_index=G_convert.edge_index, edge_attr = G_convert.weight))
  if (i in split_idx['test']):
    test_list.append(Data(x=(x), y = y, edge_index=G_convert.edge_index, edge_attr = G_convert.weight))

print(train_list)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## 如果你使用的是GPU,这里打印的应该是'cuda'
print('Device: {}'.format(device))

train_loader = DataLoader(train_list, batch_size=32, shuffle=False, num_workers=0)
valid_loader = DataLoader(valid_list, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_list, batch_size=32, shuffle=False, num_workers=0)
```

# GCN模型(基础)

我们按如下方式实现基础GCN模型(图形来源,CS224W,colab 2):

![2](D:\OneDrive\Works\30-8-2023\2.png)

```python
# 设置模型参数
args = {
    'device': device,
    'num_layers': 5,
    'hidden_dim': 256,
    'dropout': 0.5,
    'lr': 0.001,
    'epochs': 50,
}
args
```

这段代码定义了一个图卷积网络（GCN）的 PyTorch 类实现。该模型具有多个图卷积（GCNConv）层，批归一化（BatchNorm1d）层，以及一个 LogSoftmax 层。

这个模型在多层图卷积和批归一化后使用ReLU激活和dropout，适用于各种图结构数据的节点分类或图分类任务。

注：代码里有两个地方定义了最后一层的图卷积（GCNConv），这是代码中的一个冗余

```python
class GCN(torch.nn.Module):
    # 初始化GCN模型
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):
        super(GCN, self).__init__()

        # self.convs 存储多个GCNConv层
        self.convs = None
        # self.bns 存储多个BatchNorm1d层
        self.bns = None
        # LogSoftmax 层
        self.softmax = None

        # 创建GCNConv层列表
        self.convs = torch.nn.ModuleList()
        # 创建BatchNorm1d层列表
        self.bns = torch.nn.ModuleList()

        # 根据层级数量初始化不同层
        for l in range(num_layers):
          if l==0:  # 第一层：输入维度 -> 隐藏层维度
            self.convs.append(GCNConv(input_dim, hidden_dim))
          elif l == num_layers-1:  # 最后一层：隐藏层维度 -> 输出维度
            self.convs.append(GCNConv(hidden_dim, output_dim))
          else:  # 中间层：隐藏层维度 -> 隐藏层维度
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
          if l < num_layers-1:
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        # 最后一个 GCNConv 层
        self.last_conv = GCNConv(hidden_dim, output_dim)
        # LogSoftmax层
        self.log_soft = torch.nn.LogSoftmax()

        # dropout 概率
        self.dropout = dropout
        # 是否返回节点嵌入
        self.return_embeds = return_embeds

    # 重置模型参数
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    # 前向传播
    def forward(self, x, adj_t, edge_weight):
        out = None
        # 遍历所有GCNConv层和BatchNorm1d层
        for l in range(len(self.convs)-1):
          x = self.convs[l](x, adj_t, edge_weight)  # 图卷积操作
          x = self.bns[l](x)  # 批量归一化
          x = F.relu(x)  # ReLU激活函数
          x = F.dropout(x, training=self.training)  # Dropout操作

        # 最后一个GCNConv层
        x = self.last_conv(x, adj_t, edge_weight)
        # 判断是否返回节点嵌入
        if self.return_embeds is True:
          out = x
        else:
          out = self.log_soft(x)

        return out

```

# 图级预测模型(继承GCN类)

在高层次上,我们使用早期的GCN模型实现节点分类,然后使用ASAPool函数进行池化。这使我们能够在结构层上执行预测。高层流程在下图中进行了概括。

以第一层GCN为例，模拟数据在GCN模型中的数据流：

1. **输入**: 输入的图数据（一般作为一个`Data`对象）会有一个节点特征矩阵`x`（每行是一个节点，每列是一个特征）。

2. **GCNConv**: 图卷积层（`GCNConv`）首先会根据节点自身和其邻居的特征来更新节点特征。它使用邻接矩阵和节点特征矩阵来执行这个操作。输出是一个新的节点特征矩阵，与输入矩阵具有相同的行数（节点数），但列数（特征数）可能不同，这取决于该层的输出维度。

4. **ReLU激活**: 接着，节点特征矩阵通过一个ReLU激活函数。输出的节点特征矩阵与输入的形状相同，但值经过ReLU激活。

ASAPooling:

1. **输入**: 这个层接收图卷积后的节点特征矩阵。

2. **操作**: ASAPooling进行软聚类，即每个节点被赋予一个与每个聚类相关的权重。这些权重用于汇总节点以生成一个更粗糙（即节点数更少）的图表示。输出通常包括一个新的节点特征矩阵（现在有更少的节点），以及一个新的邻接矩阵和相应的边权重。这个新图的节点特征是原始节点特征的加权组合。


总体而言，在这个`GCN_Graph`模型中，数据首先通过多个GCN和ASA池化层，每一层都会更新和抽象节点和图的表示。最后，一个全局汇聚层将这些图级别的嵌入合并成一个单一的向量，该向量用于最终的图分类或回归任务。这种处理流程是图神经网络常见的层次结构和操作顺序。![3](C:\Users\zeyan\OneDrive\Works\30-8-2023\3.png)

我们应该注意,处理是以小批量进行的。总结一下CS224W Colab2中的描述,为了并行处理一小批图,PyG将图组合成一个单独的断开图的数据对象(**torch\_geometric.data.Batch**)。**torch\_geometric.data.Batch** 继承自 **torch\_geometric.data.Data**,并包含一个额外的名为`batch`的属性。`batch`属性是一个向量,将每个节点映射到其在小批量中的对应图的索引:

batch = [0, ..., 0, 1, ..., n-2, n-1, ..., n-1]

这使我们能够跟踪每个节点属于哪个图。

```python
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import global_add_pool, global_mean_pool

### GCN_Graph 模型用于预测图属性
class GCN_Graph(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(GCN_Graph, self).__init__()

        # 节点嵌入模型，初始输入维度为 input_dim，输出维度为 hidden_dim
        self.gnn_node = GCN(input_dim, hidden_dim,
            hidden_dim, num_layers, dropout, return_embeds=True)
        
        # 注意，后续层的输入和输出维度都设为 hidden_dim
        self.gnn_node_2 = GCN(hidden_dim, hidden_dim,
        hidden_dim, num_layers, dropout, return_embeds=True)

        # 使用 ASAPool 作为池化层
        self.asap = torch_geometric.nn.pool.ASAPooling(in_channels=256, ratio=0.5, dropout=0.1, negative_slope=0.2, add_self_loops=False)

        # 初始化全局均值池化层
        self.pool = global_mean_pool

        # 输出层
        self.linear = torch.nn.Linear(hidden_dim, output_dim)

    def reset_parameters(self):
      self.gnn_node.reset_parameters()
      self.linear.reset_parameters()

    def forward(self, batched_data):
        # 该函数接受一个图的 mini-batch（torch_geometric.data.Batch）作为输入
        # 并返回每个图的预测属性。

        # 提取 mini-batch 的重要属性
        x, edge_index, batch, edge_weight = batched_data.x, batched_data.edge_index, batched_data.batch, batched_data.edge_attr
        embed = x
        out = None

        # 进行两次 GCN + ASAPooling，最后通过全局池化和一个线性层进行图属性预测
        num_graphs = int(len(batch)/51)
        post_GCN_1 = self.gnn_node(embed, edge_index, edge_weight)  # 第一次 GCN
        post_pool_1 = self.asap(post_GCN_1, edge_index)  # 第一次 ASAPooling
        post_GCN_2 = self.gnn_node_2(post_pool_1[0], post_pool_1[1], post_pool_1[2])  # 第二次 GCN
        post_pool_2 = self.asap(post_GCN_2, post_pool_1[1])  # 第二次 ASAPooling
        ultimate_gcn = self.gnn_node_2(post_pool_2[0], post_pool_2[1], post_pool_2[2])  # 最终的 GCN

        glob_pool = self.pool(ultimate_gcn, post_pool_2[3], num_graphs)  # 全局池化
        out = self.linear(glob_pool)  # 输出层

        return out

```

```python
def train(model, device, data_loader, optimizer, loss_fn):
    # 这个函数用于训练模型，
    # 利用给定的优化器和损失函数进行训练
    model.train()
    loss = 0

    # 遍历数据加载器来获取每一个批次的数据
    for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
        # 将批次数据移动到指定的设备上（通常是CPU或GPU）
        batch = batch.to(device)

        # 如果批次中只有一个数据点或所有数据点属于同一个图，则跳过该批次
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            # 使用is_labeled来忽略标签为nan（即未标记）的目标，当计算训练损失时。
            is_labeled = batch.y == batch.y

            # 接下来我们：
            # 1. 清零优化器的梯度
            # 2. 将数据喂入模型进行前向传播
            # 3. 使用`is_labeled`遮罩来过滤输出和标签
            # 4. 将过滤后的输出和标签输入损失函数

            optimizer.zero_grad()  # 清零梯度
            out = model(batch)  # 前向传播得到输出
            # 过滤标签和输出，然后计算损失
            loss = loss_fn(out[is_labeled].squeeze(), batch.y[is_labeled].to(torch.float32).squeeze())

            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新模型参数

    return loss.item()  # 返回损失值

```

```python
# 评估函数
def eval(model, device, loader, evaluator, save_model_results=False, save_file=None):
    model.eval()  # 将模型设置为评估模式
    y_true = []  # 存储实际标签的列表
    y_pred = []  # 存储预测标签的列表

    # 遍历数据加载器以获取每个批次的数据
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        # 将批次数据移动到指定的设备上
        batch = batch.to(device)

        # 如果批次中只有一个数据点，则跳过该批次
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():  # 禁用梯度计算，以加速评估过程
                pred = model(batch)  # 进行前向传播得到预测

            # 将预测和实际标签存储到列表中
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    # 将列表转换为张量，并拼接到一起
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    # 准备输入到评估器的字典
    input_dict = {"y_true": y_true, "y_pred": y_pred}

    # 如果需要保存模型预测结果
    if save_model_results:
        print("Saving Model Predictions")

        # 创建一个包含两列（y_pred 和 y_true）的pandas数据帧
        data = {}
        data['y_pred'] = y_pred.reshape(-1)
        data['y_true'] = y_true.reshape(-1)

        df = pd.DataFrame(data=data)
        # 保存为CSV文件
        df.to_csv('ogbg-molhiv_graph_' + save_file + '.csv', sep=',', index=False)

    # 返回评估结果
    return evaluator.eval(input_dict)

```

为GCN_Graph模型设置参数。

我们的数据具有嵌入大小4,这是我们的输入维度,我们的输出预测的维度为1。

我们使用来自PygGraphPropPredDataset的ROC AUC评估指标。

```python
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
	
if 'IS_GRADESCOPE_ENV' not in os.environ:
  model = GCN_Graph(4, args['hidden_dim'],
              1, args['num_layers'],
              args['dropout']).to(device)
  evaluator = Evaluator(name='ogbg-molhiv')

  dataset = PygGraphPropPredDataset(name='ogbg-molhiv')  # 这个数据集要根据自己的数据集进行更换
```

```python
import copy

if 'IS_GRADESCOPE_ENV' not in os.environ:
  model.reset_parameters()

  optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
  loss_fn = torch.nn.BCEWithLogitsLoss()

  best_model = None
  best_valid_acc = 0

  for epoch in range(1, 1 + args["epochs"]):
    print('Training...')
    loss = train(model, device, train_loader, optimizer, loss_fn)

    print('Evaluating...')
    train_result = eval(model, device, train_loader, evaluator)
    val_result = eval(model, device, valid_loader, evaluator)
    test_result = eval(model, device, test_loader, evaluator)

    train_acc, valid_acc, test_acc = train_result[dataset.eval_metric], val_result[dataset.eval_metric], test_result[dataset.eval_metric]
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        best_model = copy.deepcopy(model)
    print(f'Epoch: {epoch:02d}, '
          f'Loss: {loss:.4f}, '
          f'Train: {100 * train_acc:.2f}%, '
          f'Valid: {100 * valid_acc:.2f}% '
          f'Test: {100 * test_acc:.2f}%')
```

```python
if 'IS_GRADESCOPE_ENV' not in os.environ:
  train_acc = eval(best_model, device, train_loader, evaluator)[dataset.eval_metric]
  valid_acc = eval(best_model, device, valid_loader, evaluator, save_model_results=True, save_file="valid")[dataset.eval_metric]
  test_acc  = eval(best_model, device, test_loader, evaluator, save_model_results=True, save_file="test")[dataset.eval_metric]

  print(f'Best model: '
      f'Train: {100 * train_acc:.2f}%, '
      f'Valid: {100 * valid_acc:.2f}% '
      f'Test: {100 * test_acc:.2f}%')
```

