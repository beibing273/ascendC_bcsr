import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif']=['STFangsong'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

# --- 数据 ---
# 直接从 time.txt 文件中提取的数据
files = ['Bai', 'Gset', 'HB', 'JGD', 'Pajek', 'VDOL']
coo_op_runner_times = [515.6159, 516.6048, 506.7893, 568.3672, 566.1916, 513.8848]
# coo_times = [1.9841, 1.1913, 1.2642, 11.3441, 3.4673, 2.6565]
csc_op_runner_times = [501.4587, 486.9855, 484.2329, 557.8752, 541.6517, 503.8995]
# csc_times = [1.8942, 2.5030, 1.7475, 11.8675, 5.6768, 10.6015]
matmul_op_runner_times = [513.1319,520.7822,510.6478,601.6085,662.0627,586.2662]
# matmul_times = [9.8621, 3.9694, 3.6408, 11.3441, 70.5576, 88.7935]

# baseline: BCSR
bcsr_op_runner_times = [521.5197, 554.0280, 511.0478, 716.5158, 706.7642, 616.3957]
# bcsr_mmad_times = [11.0793, 30.5775, 7.0956, 161.5639, 147.4451, 71.1448]

# BCSR Bai opt_v1:
# 项目 'aclnnBcsrSpmmCustom' 的分析结果：
#   共找到 72 个样本。
#   总耗时: 595.0228 ms
#   平均耗时: 8.2642 ms


# 'Bai.txt'

# 项目 'opRunner.RunOp' 的分析结果：
#   共找到 72 个样本。
#   总耗时: 37549.4189 ms
#   平均耗时: 521.5197 ms

# 项目 'aclnnBcsrSpmmCustom' 的分析结果：
#   共找到 72 个样本。
#   总耗时: 797.7105 ms
#   平均耗时: 11.0793 ms

# 'Gset.txt'

# 项目 'opRunner.RunOp' 的分析结果：
#   共找到 33 个样本。
#   总耗时: 18282.9245 ms
#   平均耗时: 554.0280 ms

# 项目 'aclnnBcsrSpmmCustom' 的分析结果：
#   共找到 33 个样本。
#   总耗时: 1009.0559 ms
#   平均耗时: 30.5775 ms

# 'HB.txt'

# 项目 'opRunner.RunOp' 的分析结果：
#   共找到 179 个样本。
#   总耗时: 91477.5533 ms
#   平均耗时: 511.0478 ms

# 项目 'aclnnBcsrSpmmCustom' 的分析结果：
#   共找到 179 个样本。
#   总耗时: 1270.1178 ms
#   平均耗时: 7.0956 ms

# 'JGD_Homology.txt'

# 项目 'opRunner.RunOp' 的分析结果：
#   共找到 111 个样本。
#   总耗时: 79533.2558 ms
#   平均耗时: 716.5158 ms

# 项目 'aclnnBcsrSpmmCustom' 的分析结果：
#   共找到 111 个样本。
#   总耗时: 17933.5957 ms
#   平均耗时: 161.5639 ms

# 'Pajek.txt'

# 项目 'opRunner.RunOp' 的分析结果：
#   共找到 93 个样本。
#   总耗时: 65729.0680 ms
#   平均耗时: 706.7642 ms

# 项目 'aclnnBcsrSpmmCustom' 的分析结果：
#   共找到 93 个样本。
#   总耗时: 13712.3898 ms
#   平均耗时: 147.4451 ms

# 'VDOL.txt'

# 项目 'opRunner.RunOp' 的分析结果：
#   共找到 91 个样本。
#   总耗时: 56092.0127 ms
#   平均耗时: 616.3957 ms

# 项目 'aclnnBcsrSpmmCustom' 的分析结果：
#   共找到 91 个样本。
#   总耗时: 6474.1766 ms
#   平均耗时: 71.1448 ms
# --- 可视化 ---

# 设置 x 轴的位置
x = np.arange(len(files))
width = 0.8  # 条形的宽度

fig, ax = plt.subplots(figsize=(14, 8),dpi=200)

# 绘制 COO 格式的条形
rects1 = ax.bar(x - width/3, coo_op_runner_times, width/3, label='COO', color='lightcoral')

# 绘制 CSC 格式的条形
rects2 = ax.bar(x , csc_op_runner_times, width/3, label='CSC', color='skyblue')

# 绘制 MatMul 格式的条形
rects3 = ax.bar(x + width/3, matmul_op_runner_times, width/3, label='MatMul', color='lightgreen')

# 添加标题、标签和图例
ax.set_ylabel('平均耗时 (ms)')
ax.set_title('不同数据集中 COO 与 CSC 格式以及 MatMul 的平均耗时对比')
ax.set_xticks(x)
# 将 X 轴标签设置为不倾斜且居中
ax.set_xticklabels(files)
ax.legend()

# 在条形上方显示数值
ax.bar_label(rects1, padding=3, fmt='%.2f')
ax.bar_label(rects2, padding=3, fmt='%.2f')
ax.bar_label(rects3, padding=3, fmt='%.2f')

# # 使用对数刻度，因为两组数据的值差距很大
# ax.set_yscale('log')
# ax.set_ylabel('平均耗时 (ms) - 对数刻度')

fig.tight_layout()

# 保存图表到文件
output_image_path = 'image.png'
plt.savefig(output_image_path)

plt.show()
