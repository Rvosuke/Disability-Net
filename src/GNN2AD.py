# GNN2AD.py
from ogb.graphproppred.mol_encoder import AtomEncoder  # 用于原子编码
from torch_geometric.nn import global_add_pool, global_mean_pool  # 全局池化层

# 定义 GCN_Graph 类，继承自 PyTorch 的 Module 类
class GCN_Graph(torch.nn.Module):
    # 初始化函数，设置模型的参数
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(GCN_Graph, self).__init__()

        # 节点嵌入模型，输入维度为 input_dim，输出维度为 hidden_dim
        self.gnn_node = GCN(input_dim, hidden_dim,
            hidden_dim, num_layers, dropout, return_embeds=True)

        # 第二层 GCN，输入和输出维度都设置为 hidden_dim
        self.gnn_node_2 = GCN(hidden_dim, hidden_dim,
        hidden_dim, num_layers, dropout, return_embeds=True)

        # 设置ASAPool层
        self.asap = torch_geometric.nn.pool.ASAPooling(in_channels=256, ratio=0.5, dropout=0.1, negative_slope=0.2, add_self_loops=False)

        # 初始化全局平均池化层
        self.pool = global_mean_pool

        # 输出线性层，用于图属性预测
        self.linear = torch.nn.Linear(hidden_dim, output_dim)

    # 参数重置函数
    def reset_parameters(self):
      self.gnn_node.reset_parameters()
      self.linear.reset_parameters()

    # 前向传播函数
    def forward(self, batched_data):
        # 提取mini-batch中的重要属性
        x, edge_index, batch, edge_weight = batched_data.x, batched_data.edge_index, batched_data.batch, batched_data.edge_attr
        embed = x  # 初始嵌入
        out = None  # 初始化输出

        # 计算mini-batch中图的数量
        num_graphs = int(len(batch)/51)

        # 第一次 GCN 传播和ASAPooling
        post_GCN_1 = self.gnn_node(embed, edge_index, edge_weight)
        post_pool_1 = self.asap(post_GCN_1, edge_index)

        # 第二次 GCN 传播和ASAPooling
        post_GCN_2 = self.gnn_node_2(post_pool_1[0], post_pool_1[1], post_pool_1[2])
        post_pool_2 = self.asap(post_GCN_2, post_pool_1[1])

        # 最终的GCN传播
        ultimate_gcn = self.gnn_node_2(post_pool_2[0], post_pool_2[1], post_pool_2[2])

        # 全局池化
        glob_pool = self.pool(ultimate_gcn, post_pool_2[3], num_graphs)  

        # 使用线性层进行图属性预测
        out = self.linear(glob_pool)    

        return out  # 返回预测结果
