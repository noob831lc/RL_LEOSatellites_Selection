import numpy as np
from scipy.optimize import least_squares
import time
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）


# 生成真实数据
def true_model(t, A, omega, phi, gamma, offset):
    return A * np.exp(-gamma * t) * np.cos(omega * t + phi) + offset


# 定义残差函数
def residuals(params, t, y):
    A, omega, phi, gamma, offset = params
    y_model = true_model(t, A, omega, phi, gamma, offset)
    return y_model - y


# 解析雅可比矩阵
def jac_analytic(params, t, y):
    A, omega, phi, gamma, offset = params
    exp_term = np.exp(-gamma * t)
    cos_term = np.cos(omega * t + phi)
    sin_term = np.sin(omega * t + phi)

    J = np.zeros((len(t), len(params)))

    # ∂r/∂A
    J[:, 0] = exp_term * cos_term
    # ∂r/∂omega
    J[:, 1] = -A * exp_term * t * sin_term
    # ∂r/∂phi
    J[:, 2] = -A * exp_term * sin_term
    # ∂r/∂gamma
    J[:, 3] = -A * t * exp_term * cos_term
    # ∂r/∂offset
    J[:, 4] = 1

    return J


# 生成数据
np.random.seed(42)
t_data = np.linspace(0, 10, 200)
true_params = [1.5, 2.0, 0.5, 0.3, 0.1]  # A, omega, phi, gamma, offset
y_true = true_model(t_data, *true_params)
noise = 0.05 * np.random.normal(size=len(t_data))
y_data = y_true + noise

# 初始猜测（故意设置得较远）
x0 = [1.0, 1.5, 0.0, 0.2, 0.0]

# 不同jac选项的比较
jac_methods = {
    '2-point': '2-point',
    '3-point': '3-point',
    'cs': 'cs',
    'analytic': jac_analytic
}

results = {}
for method_name, jac in jac_methods.items():
    start_time = time.time()
    result = least_squares(residuals, x0, args=(t_data, y_data),
                           jac=jac, method='trf', verbose=0)
    end_time = time.time()

    results[method_name] = {
        'params': result.x,
        'cost': result.cost,
        'nfev': result.nfev,
        'njev': result.njev,
        'time': end_time - start_time,
        'success': result.success
    }

# 打印比较结果
print("性能比较：")
print("\n真实参数:", true_params)
print("\n{:<10} {:<15} {:<10} {:<10} {:<10} {:<10}".format(
    "方法", "计算时间(s)", "函数评估", "雅可比评估", "最终代价", "成功"))
print("-" * 65)

for method, result in results.items():
    print("{:<10} {:<15.6f} {:<10d} {:<10d} {:<10.6f} {:<10}".format(
        method, result['time'], result['nfev'], result['njev'],
        result['cost'], str(result['success'])))

# 绘制拟合结果
plt.figure(figsize=(12, 8))

# 原始数据点
plt.scatter(t_data, y_data, c='gray', alpha=0.5, label='观测数据')
plt.plot(t_data, y_true, 'k--', label='真实模型')

# 不同方法的拟合结果
colors = ['b', 'r', 'g', 'm']
for (method, result), color in zip(results.items(), colors):
    y_fit = true_model(t_data, *result['params'])
    plt.plot(t_data, y_fit, color, label=f'{method}拟合')

plt.xlabel('时间 t')
plt.ylabel('振幅')
plt.title('阻尼振荡模型拟合比较')
plt.legend()
plt.grid(True)
plt.show()

# 打印参数估计结果
print("\n参数估计结果：")
print("\n{:<10} {:<12} {:<12} {:<12} {:<12} {:<12}".format(
    "方法", "A", "omega", "phi", "gamma", "offset"))
print("-" * 70)

for method, result in results.items():
    params = result['params']
    print("{:<10} {:<12.6f} {:<12.6f} {:<12.6f} {:<12.6f} {:<12.6f}".format(
        method, params[0], params[1], params[2], params[3], params[4]))
