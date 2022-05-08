import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 80  # 图形分辨率
pd.options.display.notebook_repr_html = False  # 表格显示
# 相关系数矩阵
np.random.seed(1)  # 随机种子
sim_AIMed_feature = np.array([[1.0000, 0.9562, 0.9232, 0.8513, 0.9629],
                              [0.9562, 1.0000, 0.9473, 0.8705, 0.9527],
                              [0.9232, 0.9473, 1.0000, 0.8711, 0.9323],
                              [0.8513, 0.8705, 0.8711, 1.0000, 0.9036],
                              [0.9629, 0.9527, 0.9323, 0.9036, 1.0000]])
sim_BioInfer_feature = np.array([[1.0000, 0.9260, 0.8535, 0.8072, 0.9371],
                                 [0.9260, 1.0000, 0.9173, 0.8232, 0.9412],
                                 [0.8535, 0.9173, 1.0000, 0.7875, 0.8815],
                                 [0.8072, 0.8232, 0.7875, 1.0000, 0.8655],
                                 [0.9371, 0.9412, 0.8815, 0.8655, 1.0000]])
sim_HPRD50_feature = np.array([[1.0000, 0.9186, 0.8947, 0.6757, 0.9039],
                               [0.9186, 1.0000, 0.9304, 0.7085, 0.9319],
                               [0.8947, 0.9304, 1.0000, 0.7174, 0.9191],
                               [0.6757, 0.7085, 0.8237, 1.0000, 0.7757],
                               [0.9039, 0.9319, 0.9191, 0.7757, 1.0000]])
sim_IEPA_feature = np.array([[1.0000, 0.9267, 0.8478, 0.7994, 0.9260],
                             [0.9267, 1.0000, 0.9072, 0.8215, 0.9353],
                             [0.8478, 0.9072, 1.0000, 0.8237, 0.8973],
                             [0.7994, 0.8215, 0.8237, 1.0000, 0.8789],
                             [0.9260, 0.9353, 0.8973, 0.8789, 1.0000]])
sim_LLL_feature = np.array([[1.0000, 0.9631, 0.9277, 0.9096, 0.9442],
                            [0.9631, 1.0000, 0.9669, 0.9086, 0.9734],
                            [0.9277, 0.9669, 1.0000, 0.8297, 0.9576],
                            [0.9096, 0.9086, 0.8297, 1.0000, 0.8919],
                            [0.9442, 0.9734, 0.9576, 0.8919, 1.0000]])
sim_centralized_feature = np.array([[],
                                    [],
                                    [],
                                    [],
                                    []])
sim_PGR_feature = np.array([[1.0000, 0.9194, 0.9464, 0.8752, 0.9365],
                            [0.9194, 1.0000, 0.9277, 0.9515, 0.9727],
                            [0.9464, 0.9277, 1.0000, 0.9018, 0.9332],
                            [0.8752, 0.9515, 0.9018, 1.0000, 0.9507],
                            [0.9365, 0.9727, 0.9332, 0.9507, 1.0000]])

columns = ['AIMed', 'BioInfer', 'HPRD50', 'IEPA', 'LLL']
mat_AIMed_feature = pd.DataFrame(sim_AIMed_feature, columns=columns, index=columns)
sns.heatmap(mat_AIMed_feature)
plt.show()

mat_BioInfer_feature = pd.DataFrame(sim_BioInfer_feature, columns=columns, index=columns)
sns.heatmap(mat_BioInfer_feature)
plt.show()

mat_HPRD50_feature = pd.DataFrame(sim_HPRD50_feature, columns=columns, index=columns)
sns.heatmap(mat_HPRD50_feature)
plt.show()

mat_IEPA_feature = pd.DataFrame(sim_IEPA_feature, columns=columns, index=columns)
sns.heatmap(mat_IEPA_feature)
plt.show()

mat_LLL_feature = pd.DataFrame(sim_LLL_feature, columns=columns, index=columns)
sns.heatmap(mat_LLL_feature)
plt.show()

mat_PGR_feature = pd.DataFrame(sim_PGR_feature, columns=columns, index=columns)
sns.heatmap(mat_PGR_feature)
plt.show()
