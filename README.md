# duetpy

一个面向图片特征识别与筛选的 Python 工具项目，主要用于：

- 提取图片/人像/人脸 embedding（基于 SigLIP2）
- 训练二分类特征模型（PyTorch）
- 批量推理并按阈值筛选图片
- 支持脚本化数据处理与离线评估

## 项目特点

- 提供 `train.py` 与 `predict_*.py` 的完整训练/推理流程
- 支持 `person`、`face`、`all` 等不同特征位置
- 包含 `app/` 与 `infrastructure/` 模块，方便扩展为服务化接口
- 附带多个 `tool/` 下的数据处理脚本，便于业务场景快速复用

## 目录结构

```text
duetpy/
├── app/                    # 特征提取与业务逻辑
├── infrastructure/         # 模型与图像处理工具
├── model/                  # 视觉模型定义
├── tool/                   # 辅助脚本（数据清洗/统计/转换）
├── train.py                # 模型训练入口
├── predict_single_pic.py   # 单图预测
├── predict_batch.py        # 批量预测
└── README.md
```

## 环境要求

- Python 3.10+
- 推荐 macOS + Apple Silicon（项目中默认设备为 `mps`）

常见依赖（按代码使用推断）：

- `torch`
- `torchvision`
- `transformers`
- `opencv-python`
- `pillow`
- `numpy`
- `pandas`
- `tqdm`
- `flask`
- `requests`
- `scikit-learn`

## 快速开始

1. 克隆仓库

```bash
git clone https://github.com/NIKI678531/duetpy.git
cd duetpy
```

2. 安装依赖

```bash
pip install -U pip
pip install torch torchvision transformers opencv-python pillow numpy pandas tqdm flask requests scikit-learn
```

3. 训练模型（示例）

```bash
python train.py PictureHighClarity person
```

4. 批量预测（示例）

```bash
python predict_batch.py PictureHighClarity person model_weight/isPictureHighClarity/xxx.pth /path/to/images 0.65 1 female
```

## 注意事项

- 代码中存在本地绝对路径（如 `/Users/...`），迁移到新机器时需要改为你自己的路径。
- 默认使用 `mps`，若你没有 Apple GPU，可改为 `cpu` 或 `cuda`。
- 建议将真实业务数据与模型权重放在仓库外部目录并通过配置传入。

## 许可证

当前仓库未声明开源许可证。如需开源，建议补充 `MIT` 或 `Apache-2.0`。
