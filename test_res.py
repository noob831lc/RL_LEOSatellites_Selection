import numpy as np

from utils.Position_Fun import starlink_doppler_positioning


if __name__ == "__main__":
    r_s_demo = np.array([
        [1.2e6, 2.1e6, 2.1e6],
        [2.2e6, 2.8e6, 2.6e6],
        [3.1e6, 1.2e6, 3.0e6],
        [2.1e6, 3.2e6, 3.6e6],
        [2.1e6, 3.2e6, 3.6e6]
    ])
    v_s_demo = np.array([
        [2500.0, 2200.0, 700.0],
        [2100.0, 2350.0, 650.0],
        [2200.0, 2500.0, 800.0],
        [2400.0, 2100.0, 1000.0],
        [2400.0, 2100.0, 1000.0]
    ])
    fD_obs_demo = np.array([-12345.6, -11300.2, -12580.9, -13210.0, -13210.0])

    # 调用函数
    result = starlink_doppler_positioning(
        r_s=r_s_demo,
        v_s=v_s_demo,
        fD_obs=fD_obs_demo,
        z0=0.0,  # 固定高度
        f_c=11.325e9,  # 星链导频
        c=3e8,  # 光速
        x0_guess=1e3,
        y0_guess=1e3,
        f_u_guess=0.0
    )
