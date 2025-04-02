import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）


# 定义非线性方程组的残差函数
def residuals(x):
    """
    计算非线性方程组的残差:
    f1(x,y) = x**2 + y**2 - 1 = 0
    f2(x,y) = x**2 - y = 0
    """
    return [
        x[0] ** 2 + x[1] ** 2 - 1,  # 第一个方程的残差
        x[0] ** 2 - x[1]  # 第二个方程的残差
    ]


# 设置初始猜测值
x0 = np.array([0.5, 0.5])

# 使用least_squares求解
result = least_squares(
    residuals,  # 残差函数
    x0,  # 初始猜测值
    method='lm',  # Levenberg-Marquardt算法
    ftol=1e-15,  # 函数收敛容差
    xtol=1e-15,  # 参数收敛容差
    verbose=2  # 输出迭代信息
)

# 输出结果
print("\n求解状态:", "成功" if result.success else "失败")
print("迭代次数:", result.nfev)
print("最终解:", result.x)
print("最终残差:", result.fun)

# 从cost属性获取残差历史
cost_history = np.sqrt(2 * result.cost)  # 将cost转换为残差范数

# 绘制收敛性分析图
plt.figure(figsize=(10, 6))
plt.semilogy(range(result.nfev), [cost_history] * result.nfev, 'b-', label='残差')
plt.grid(True)
plt.xlabel('迭代次数')
plt.ylabel('残差 (对数尺度)')
plt.title('非线性方程组求解收敛性分析')
plt.legend()
plt.show()

# 验证结果
x_sol = result.x
print("\n验证结果:")
print(f"方程1的残差: {x_sol[0] ** 2 + x_sol[1] ** 2 - 1:.2e}")
print(f"方程2的残差: {x_sol[0] ** 2 - x_sol[1]:.2e}")
