import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')

# 定义文件路径列表
dataset = "celegans_connectome_multiplex_23"
# dataset = 'netsci_co-authorship_multiplex_12'
# dataset = 'drosophila_melanogaster_multiplex_56'

# dataset = 'us_air_transportation_american_delta_multiplex_12'
file_paths = [
    f"./results/FINDER_ND_weightQ/g0.5-degree/StepRatio_0.0000/MaxCCList_Strategy_{dataset}.txt",
    f"./results/FINDER_ND_weightQ/g0.5-degree/StepRatio_0.0000/MaxCCList_Strategy_{dataset}_new.txt"
    # f"./results/CI/none_cost/MaxCCList_Strategy_{dataset}.txt",
    # f"./results/minsum/{dataset}/{dataset}.txt",
    # f"./results/FINDER_ori/none_cost/StepRatio_0.0000/MaxCCList_Strategy_{dataset}.txt",
    # f"./results/NIRM_MCC/{dataset}/{dataset}.txt",
    # f"./results/FINDER_ND_weightQ/g0.5/StepRatio_0.0000/MaxCCList_Strategy_{dataset}.txt",
]

# 初始化数据列表
values = []

# 循环读取文件并将每行的值存储在列表中
for file_path in file_paths:
    with open(file_path, "r") as file:
        lines = file.readlines()
        values.append([float(line.strip()) for line in lines])

# 计算x和y的值
x_values = []
y_values = []

for values_list in values:
    total_lines = len(values_list)
    N = 214
    x_values.append([i / N for i in range(0, total_lines)])
    y_values.append(values_list)

# 设置字体和字体大小
font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 14}

plt.text(0.05, 0.95, 'A', transform=plt.gcf().transFigure, fontsize=18, fontweight='normal', fontname='Times New Roman',
         verticalalignment='top')

plt.rc('font', **font)
plt.rc('axes', titlesize=16)  # 标题字体大小
plt.rc('axes', labelsize=12)  # x 和 y 轴标签的字体大小
plt.rc('xtick', labelsize=12)  # x 轴刻度标签的字体大小
plt.rc('ytick', labelsize=12)  # y 轴刻度标签的字体大小

# 绘制图形
labels = ["Ori", "new"]
# colors = ['#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', 'r']
# marker = ['o', 's', '^', 'D', '+', '*']

# for i in range(len(values)):
#     plt.plot(x_values[i], y_values[i], label=f'{labels[i]}', c=colors[i], linewidth=2, marker=marker[i], markevery=0.1)
for i in range(len(values)):
    plt.plot(x_values[i], y_values[i], label=f'{labels[i]}')

# plt.xlabel('Fraction of removal node costs')
# plt.ylabel('Normalized LMCC of residual network')

# 设置图例字体
legend_font = {'family': 'Times New Roman',
               'weight': 'normal',
               'size': 10}

plt.legend(prop=legend_font)

title_font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 15}
plt.title(f'{dataset}', fontdict=title_font)
plt.xlim(0, 0.5)
plt.savefig(f"{dataset}.png")
# plt.xticks(np.arange(0, 0.1, 0.02))
# plt.show()