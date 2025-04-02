import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


# 定义非线性方程组
def equations(vars):
    x, y = vars
    f1 = x ** 2 + y ** 2 - 4  # 方程 1
    f2 = np.exp(x) + y - 1  # 方程 2
    return [f1, f2]


# 创建一个包装器，用于记录每次迭代的残差
class ResidualTracker:
    def __init__(self, func):
        self.func = func  # 原始方程组
        self.residual_history = []  # 用于存储每次调用的残差平方和

    def __call__(self, vars):
        residual = self.func(vars)  # 计算当前残差
        residual_sum = np.sum(np.array(residual) ** 2)  # 残差平方和
        self.residual_history.append(residual_sum)  # 保存到历史记录
        return residual


# 初始猜测值
initial_guess = [1, 1]

# 包装原始方程组
tracker = ResidualTracker(equations)

# 使用最小二乘求解
result = least_squares(tracker, initial_guess, verbose=2, ftol=1e-8, xtol=1e-8, gtol=1e-8)

# 打印优化结果
print("优化结果：")
print("x =", result.x[0])
print("y =", result.x[1])

# 绘制迭代次数与残差平方和的变化曲线
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(tracker.residual_history) + 1), tracker.residual_history, marker='o', linestyle='-', color='blue')
plt.title('Residual Sum of Squares vs Iteration Number')
plt.xlabel('Iteration Number')
plt.ylabel('Residual Sum of Squares')
plt.grid(True)
plt.show()
