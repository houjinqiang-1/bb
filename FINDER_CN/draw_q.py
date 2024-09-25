import numpy as np
import matplotlib.pyplot as plt

# 读取 npy 文件
q_values = np.load('/home/gww/yangchen/FINDER-pytorch-multiplex/FINDER_CN/q_list/q_list_5.npy')

# 绘制 q 值分布图
plt.figure(figsize=(10, 6))
plt.hist(q_values, bins=50, alpha=0.75, color='blue', edgecolor='black')

# 设置图表标题和标签
plt.title('Distribution of Q Values')
plt.xlabel('Q Value')
plt.ylabel('Frequency')

# 显示网格线
plt.grid(True)

# 显示图表
plt.show()
plt.savefig('q_5')