import matplotlib.pyplot as plt
import numpy as np

# 生成示例数据
x = np.linspace(0, 10, 100)                 # x: 从 0 到 10 的 100 个点
y1 = np.sin(x)                              # 第一条曲线
y2 = np.cos(x)                              # 第二条曲线

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(7, 4))

# 绘制两条折线
ax.plot(x, y1, label='sin(x)', color='tab:blue', linewidth=2)
ax.plot(x, y2, label='cos(x)', color='tab:orange', linewidth=2, linestyle='--')

# 添加标题和坐标轴标签
ax.set_title('Matplotlib 折线图示例', fontsize=14)
ax.set_xlabel('X 轴')
ax.set_ylabel('Y 轴')

# 添加网格和图例
ax.grid(True, alpha=0.3)
ax.legend()

# 显示图形
plt.show()
