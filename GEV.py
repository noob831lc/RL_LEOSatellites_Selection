import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False


def compute_gev_vector(phi, beta):
    x = np.sin(phi) * np.cos(beta)
    y = np.sin(phi) * np.sin(beta)
    z = np.cos(phi)
    return np.array([x, y, z])


def random_gev_params():
    phi = np.random.uniform(np.pi / 3, 2 * np.pi / 3)
    beta = np.random.uniform(0, 2 * np.pi)
    return phi, beta


n_combinations = 4
combinations_data = []

for i in range(n_combinations):
    phi1, beta1 = random_gev_params()
    phi2, beta2 = random_gev_params()
    gev1 = compute_gev_vector(phi1, beta1)
    gev2 = compute_gev_vector(phi2, beta2)
    dot_val = np.clip(np.dot(gev1, gev2), -1.0, 1.0)
    angle = np.degrees(np.arccos(dot_val))
    dgdop = np.random.uniform(40, 100)

    combinations_data.append((gev1, gev2, angle, dgdop))

fig = plt.figure(figsize=(12, 10))
for idx, (gev1, gev2, angle, dgdop) in enumerate(combinations_data):
    ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
    ax.scatter(0, 0, 0, color='k', marker='o')
    ax.quiver(0, 0, 0, gev1[0], gev1[1], gev1[2], color='red',
              arrow_length_ratio=0.15, linewidth=2, label='卫星1')
    ax.quiver(0, 0, 0, gev2[0], gev2[1], gev2[2], color='blue',
              arrow_length_ratio=0.15, linewidth=2, label='卫星2')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_title("组合 {}".format(idx + 1), fontsize=14)
    abs_diff = abs(90 - angle)
    ax.text2D(0.05, 0.90, "DGDOP = {:.1f}".format(dgdop),
              transform=ax.transAxes, fontsize=12, color='green')
    ax.text2D(0.05, 0.80, "|90°-∠GEVs| = {:.1f}°".format(abs_diff),
              transform=ax.transAxes, fontsize=12, color='purple')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

plt.tight_layout()
plt.show()
