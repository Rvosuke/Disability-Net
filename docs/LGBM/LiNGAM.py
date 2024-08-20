# DirectLiNGAM.py

import numpy as np
import pandas as pd
# Graphviz 用于可视化图形结构，在展示 LiNGAM模型的因果关系时可能会用到。
import graphviz
# LiNGAM 库是一种用于发现变量之间因果关系的统计模型
import lingam
# make_dot 函数用于生成可视化Graphviz对象（点）
from lingam.utils import make_dot
from sklearn.datasets import load_iris

# 设置numpy格式
np.set_printoptions(precision=2, suppress=True)
np.random.seed(827)

# 加载Iris数据集，转化为DataFreame
iris = load_iris()
data = iris['data']
feature_names = iris['feature_names']
df = pd.DataFrame(data, columns=feature_names)

# 创建DirectLiNGAM模型
model = lingam.DirectLiNGAM()
model.fit(df)

# 打印因果顺序、邻接矩阵
print(model.causal_order_)
print(model.adjacency_matrix_)

# 绘制因果图
dot = make_dot(model.adjacency_matrix_)
dot.format = 'png'
dot.render('Iris')
