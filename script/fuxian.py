import numpy as np
from numpy.linalg import lstsq
import cmath


def construct_linear_system(k_vals, G_vals, P):
    """
    构造并返回线性方程组 (W, G), 其中:
    - k_vals:   采样索引列表, 长度为 2P
    - G_vals:   对应的观测值列表, 长度为 2P
    - P:        路径数 P_tau
    输出:
    - W:        (2P, 2P) 维矩阵
    - G_vector: (2P, )   观测列向量
    这里仅示例性地构造一个简化模型, 实际实现需结合具体的通分展开
    """

    # 长度检查
    if len(k_vals) != 2 * P or len(G_vals) != 2 * P:
        raise ValueError("需要 2P 个采样点及观测值")

    # 构造观测向量 G
    G_vector = np.array(G_vals, dtype=np.complex128).reshape(-1, 1)

    # 在此只是做一个示意: G(k_m) * poly_b(W^k_m) - poly_a(W^k_m) = 0
    # poly_a, poly_b 的阶数各为 P, 故未知量总数 2P
    # 令 poly_a(x) = a0 + a1*x + ... + a_{P-1}*x^{P-1}
    #    poly_b(x) = b0 + b1*x + ... + b_{P-1}*x^{P-1}
    # 选取 b0 = 1 也可行, 但此处不做固定, 以保持一般性

    # W 矩阵的每一行, 对应一个采样点 k_m:
    #   row_m = [ - x^0, - x^1, ..., - x^{P-1},
    #             G(k_m)* x^0, G(k_m)* x^1, ..., G(k_m)* x^{P-1} ]
    # 其中 x = W^(k_m), W = e^{-j*2π/N}, 这里可直接用 e^{ - j * 2*pi * k/N } .

    # 示例中, N 默认为某个固定值, 用以生成 x. 若已知 N, 可根据需要传参.
    N = 32  # 这里仅做演示

    def W_k(k):
        return np.exp(-1j * 2.0 * np.pi * k / N)

    # 构造 W 矩阵
    W_matrix = []
    for m in range(2 * P):
        k_m = k_vals[m]
        x_m = W_k(k_m)  # W^(k_m)
        # 构造 x^0, x^1, ..., x^{P-1}
        powers_x = np.array([x_m ** r for r in range(P)], dtype=np.complex128)

        # TODO: 这里可根据 "G(k_m)*poly_b - poly_a" 整理出行向量
        g_m = G_vals[m]
        # 负号是因为 G(k_m)*b(...) - a(...) = 0
        row_m_a = - powers_x  # 对应 a0..a_{P-1}
        row_m_b = g_m * powers_x  # 对应 b0..b_{P-1}

        # 合并
        row_m = np.concatenate([row_m_a, row_m_b])
        W_matrix.append(row_m)

    W_matrix = np.array(W_matrix, dtype=np.complex128)

    return W_matrix, G_vector


def solve_linear_system(W, G):
    """
    使用最小二乘求解 theta = [a0..a_{P-1}, b0..b_{P-1}]^T
    """
    # 这里直接用 numpy.linalg.lstsq
    theta, residuals, rank, s = lstsq(W, G, rcond=None)
    return theta, residuals


def polynomial_roots(theta, P):
    """
    给定求解得到的 theta (长度 2P), 前一半是 (a0..a_{P-1}) 向量
    后一半是 (b0..b_{P-1}) 向量
    此处示例要对 b(x) = b0 + b1 x + ... + b_{P-1} x^{P-1} 的多项式进行求根
    (可视具体情况补上最高阶系数)
    """
    a = theta[:P].flatten()  # a0..a_{P-1}
    b = theta[P:].flatten()  # b0..b_{P-1}

    # 如果多项式阶数要到 P, 那 b 是长度 P:
    # b(x) = b0 + b1 x + ... + b_{P-1} x^{P-1}
    # 若实际中需要 b_P = 1, 则可在这里进行拼接.
    # 做一个示例: b_poly = [b_{P-1}, ..., b_0], 方便 np.roots
    b_poly = np.concatenate([[1.0], b[::-1]])
    # 注意: 这意味着我们假设最高阶系数是 1,
    # 也可根据场景不同进行额外处理.

    # 求根
    roots_b = np.roots(b_poly)

    return a, b, roots_b


# ==========================#
#       主流程示例        #
# ==========================#

if __name__ == "__main__":
    # 假设我们有 2 条路径 (P=2), 因此需要 4 个采样点
    k_vals_ex = [2, 5, 10, 13]  # 假设在时延 bin 下选取的采样点索引
    G_vals_ex = [1.2 + 0.5j, 0.9 - 0.3j, 1.05 + 0.2j, 0.88 + 0.07j]  # 虚拟观测

    P = 2

    W_mat, G_vec = construct_linear_system(k_vals_ex, G_vals_ex, P)
    theta_hat, residuals = solve_linear_system(W_mat, G_vec)

    print("线性方程组求解得到的 theta = [a0.., b0..]:")
    print(theta_hat)
    print("残差:", residuals)

    # 对分母多项式求根, 恢复 z_i
    a_hat, b_hat, roots_b = polynomial_roots(theta_hat, P)
    print("a_hat =", a_hat)
    print("b_hat =", b_hat)
    print("分母多项式的根 =", roots_b)

    # 如需得到各路径的 k_{d_i}, 可用 angle(roots_b):
    # k_d_i = (N/(2π)) * arg(roots_b[i])
    # 这里只是示例输出:
    for r in roots_b:
        kd_est = (32 / (2.0 * np.pi)) * np.angle(r)
        print(f"估计的kd = {kd_est:.3f}")