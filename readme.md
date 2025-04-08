# 卫星选择与强化学习模型项目

## 项目简介

本项目旨在通过算法选择和强化学习模型来优化卫星选择过程。项目包含多个模块，包括算法实现、强化学习模型训练与测试、卫星数据处理等。

## 项目结构

```
.gitignore
algorithm/
    __init__.py
    Selection.py
requirements.txt
RL_model/
    __init__.py
    model.py
    test.py
    train.py
sat_selection.py
satellite_data/
    starlink_tle_20241219_145248.txt
    starlink_tle_20241219_145508.txt
    starlink_tle_20241219_145604.txt
    starlink_tle_20241224_143810.txt
    starlink_tle_20241224_202700.txt
    starlink_tle_20241224_215525.txt
    starlink_tle_20241225_163426.txt
    starlink_tle_20241225_174032.txt
    ...
test_skyfield.py
test.py
utils/
    __init__.py
    ...
world_visable.py
```

## 项目内容

### 1. 算法模块

- `algorithm/Selection.py`: 包含用于卫星选择的算法实现。

### 2. 强化学习模型模块

- `RL_model/model.py`: 定义了强化学习模型的结构和方法。
- `RL_model/train.py`: 包含用于训练强化学习模型的脚本。
- `RL_model/test.py`: 包含用于测试强化学习模型的脚本。

### 3. 卫星数据

- `satellite_data/`: 目录下包含多个卫星数据文件，这些文件以时间戳命名，存储了不同时间点的卫星轨道数据。

### 4. 工具模块

- `utils/`: 目录下包含多个工具脚本，用于支持项目的各种功能。

### 5. 其他脚本

- `sat_selection.py`: 主脚本，用于运行卫星选择过程。
- `test_skyfield.py`: 用于测试 Skyfield 库的脚本。
- `test.py`: 包含项目的测试脚本。
- `world_visable.py`: 用于计算卫星在世界范围内可见性的脚本。

## 安装与运行

### 依赖安装

请确保已安装 `pip`，然后运行以下命令安装项目依赖：

```sh
pip install -r requirements.txt
```

### 运行项目

根据需要运行不同的脚本，例如：

```sh
python sat_selection.py
```

## 贡献

欢迎贡献代码和提出建议。请提交 Pull Request 或 Issue。

## 许可证

本项目采用 MIT 许可证。