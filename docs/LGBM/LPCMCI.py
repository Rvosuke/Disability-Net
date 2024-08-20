# LPCMCI.py
import tigramite
import numpy as np
from tigramite.pcmci import PCMCI
from tigramite.lpcmci import LPCMCI
from matplotlib import pyplot as plt
from tigramite import plotting as tp
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.toymodels import structural_causal_processes as toys


# 模拟数据
def lin(x): return x

links = {0: [((0, -1), 0.9, lin), ((1, 0), 0.6, lin)],
         1: [],
         2: [((2, -1), 0.9, lin), ((1, -1), 0.4, lin)],
         3: [((3, -1), 0.9, lin), ((2, -2), -0.5, lin)]                                    
        }

# 噪声项
random_state = np.random.RandomState(827)
noises = noises = [random_state.randn for j in links.keys()]

data_full, nonstationarity_indicator = toys.structural_causal_process(
    links=links, T=500, noises=noises, seed=827)
assert not nonstationarity_indicator

# 移除未知样例
data_obs = data_full[:, [0, 2, 3]]

# 已知样例个数
N = data_obs.shape[1]

# 初始化数据集
var_names = [r'$X^{%d}$' % j for j in range(N)]
dataframe = pp.DataFrame(data_obs, var_names=var_names)

tp.plot_timeseries(dataframe, figsize=(15, 5));
plt.show()

# 创建 LPCMCI 对象
model = LPCMCI(dataframe=dataframe,
                cond_ind_test=ParCorr(significance='analytic'),  # 传入条件独立性测试对象，设置显著性方法为 'analytic'
                verbosity=1)  # 输出详细程度设置为1

# 运行 LPCMCI 算法
results = model.run_lpcmci(tau_max=5,  # 设置最大的时间延迟
                            pc_alpha=0.01) # 设置显著性水平

# 绘制学习到的时间序列 DPAG（有向无环图）
tp.plot_time_series_graph(graph=results['graph'],       
                          val_matrix=results['val_matrix'])
plt.show()

# 绘制学习到的 DPAG（有向无环图），压缩了时间维度
tp.plot_graph(graph=results['graph'],
              val_matrix=results['val_matrix'])
plt.show()
