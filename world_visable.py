from datetime import datetime, timezone
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pylab import mpl
from matplotlib.colors import BoundaryNorm

mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

import numpy as np
import matplotlib.pyplot as plt
from skyfield.api import load, wgs84
import math

satellites = load.tle_file('satellite_data\\starlink_tle_20250106_144506.txt')
print(f'加载了 {len(satellites)} 颗卫星')

ts = load.timescale()
t = ts.from_datetime(datetime(2025, 1, 6, 5, 0, 0, tzinfo=timezone.utc))

R = 6371000
R0 = R + 100
lat_grid = np.arange(-90, 91, 1)
lon_grid = np.arange(-180, 181, 1)
lon2d, lat2d = np.meshgrid(lon_grid, lat_grid)
lat_rad = np.radians(lat2d)
lon_rad = np.radians(lon2d)
density = np.zeros_like(lat2d, dtype=float)

for sat in satellites:
    subpoint = sat.at(t).subpoint()
    sat_lat = subpoint.latitude.degrees
    sat_lon = subpoint.longitude.degrees
    sat_height = subpoint.elevation.m
    R1 = R + sat_height
    sat_lat_rad = math.radians(sat_lat)
    sat_lon_rad = math.radians(sat_lon)
    delta_lon = lon_rad - sat_lon_rad
    cos_psi = np.sin(sat_lat_rad) * np.sin(lat_rad) + np.cos(sat_lat_rad)*np.cos(lat_rad)*np.cos(delta_lon)
    cos_psi = np.clip(cos_psi, -1, 1)
    psi = np.arccos(cos_psi)
    elev = np.degrees(np.arctan2(R1 * np.cos(psi) - R0, R1 * np.sin(psi)))
    mask = elev >= 25
    density[mask] += 1

# 确定色彩条边界值
max_density = np.max(density)
# 设置边界为0, 10, 20, 30, 40, 50, 60, 70，根据数据调整
levels = [0, 10, 20, 30, 40, 50, 60, 70]
# 如果数据范围更小，可以调整为更适合的边界值
# 检查最大值是否需要调整边界
if max_density > 70:
    levels = np.arange(0, max_density + 10, 10)

# 创建分段的颜色映射
cmap = plt.cm.jet
norm = BoundaryNorm(levels, cmap.N)

fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='white')
ax.coastlines()

# 使用norm参数来分段显示颜色
mesh = ax.pcolormesh(lon_grid, lat_grid, density, transform=ccrs.PlateCarree(),
                    shading='auto', cmap=cmap, norm=norm)

# 设置colorbar使用相同的分段
cbar = plt.colorbar(mesh, ax=ax, label='可见卫星数量', ticks=levels)

# 添加网格线
gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.5, color='black', alpha=0.5, linestyle='-')
gl.xlocator = plt.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
gl.ylocator = plt.FixedLocator([-90, -60, -30, 0, 30, 60, 90])

plt.xlabel('经度 (°)')
plt.ylabel('纬度 (°)')
plt.xlim((-180, 180))
plt.ylim((-90, 90))

ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])

# 添加度数符号到刻度标签
ax.set_xticklabels([f'{x}°' for x in [-180, -120, -60, 0, 60, 120, 180]])
ax.set_yticklabels([f'{y}°' for y in [-90, -60, -30, 0, 30, 60, 90]])
plt.title('全球范围内 Starlink 可见卫星分布')

# 保存图片（可选）
# plt.savefig('starlink_distribution.png', dpi=300, bbox_inches='tight')

plt.show()
