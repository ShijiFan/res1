# HSI & Earth Observation Foundation Models: Baselines & Repositories

本项目面向高光谱（Hyperspectral Imaging, HSI）与地球观测（Earth Observation, EO）两阶段研究策略，整理关键开源仓库、核心观测算子/SRF基线、跨传感器基础模型以及适配迁移实验框架。

---

## 🎯 核心研究路线与规划

依据研究策略规划，实验按照以下主线推进：
```
阶段一：物理观测与SRF算子基线 ➔ 高光谱基础模型与特征表征 ➔ 跨传感器适配
                                     ⬇
阶段二：小物理接口 / Adapter 迁移 ➔ 既有大容量 EO 基础模型微调 (无需从零重训)
```

1. **SRF / 观测算子强基线**：建立基于光谱响应函数（Spectral Response Function, SRF）与 $A_s$ 观测模型的物理卷积与重采样基准。
2. **高光谱基础模型（HSI Foundation Models）**：引入 Sensor-agnostic 架构作为跨传感器强基准，并在大容量 Spectral Backbone 上对照接口有效性。
3. **EO 基础模型迁移适配**：面向已有 Earth Observation 基础模型，通过轻量级物理响应接口（Adapter/Tokenizer 适配），低成本迁移并公平对比。

---

## 📦 核心开源仓库清单与优先级

### 1. 物理观测模型与 SRF 重采样基线

| 仓库名称 | 推荐优先级 | 角色与定位 | 仓库链接 |
| :--- | :---: | :--- | :--- |
| **EnMAP-Box** | ⭐⭐⭐⭐⭐ | **SRF / 观测算子强基准**<br>包含基于光谱响应函数（SRF）的高斯卷积重采样、波长/FWHM 重采样代码，直接贴合 $A_s$/SRF 物理观测模型。 | [EnMAP-Box/enmap-box](https://github.com/EnMAP-Box/enmap-box) |
| **Spectral Python (SPy)** | ⭐⭐⭐⭐ | **高光谱数据管线与轻量基线**<br>标准高光谱数据读取、ENVI 头文件解析、预处理与轻量分类/降维基线。 | [spectralpython/spectral](https://github.com/spectralpython/spectral) |

### 2. 高光谱基础模型（HSI Backbone & Sensor-Agnostic）

| 仓库名称 | 推荐优先级 | 角色与定位 | 仓库链接 |
| :--- | :---: | :--- | :--- |
| **HySens** | ⭐⭐⭐⭐⭐ | **跨传感器直接对照基线**<br>Sensor-agnostic 高光谱基础模型，提供预训练权重与分类、变化检测、融合、回归等多下游任务评测框架。 | [zhu-xlab/HySens](https://github.com/zhu-xlab/HySens) |
| **HyperSIGMA** | ⭐⭐⭐⭐ | **大容量 HSI Backbone 对照**<br>武汉大学 Sigma 组的高性能高光谱基础模型，用于验证传感器接口在强 Spectral Encoder 下的泛化性。 | [WHU-Sigma/HyperSIGMA](https://github.com/WHU-Sigma/HyperSIGMA) |

### 3. 地球观测 (EO) 基础模型适配与微调 (第二阶段)

| 仓库名称 | 推荐优先级 | 角色与定位 | 仓库链接 |
| :--- | :---: | :--- | :--- |
| **TerraMind** | ⭐⭐⭐⭐⭐ | **EO 基础模型迁移实验**<br>IBM 开源的多模态地球观测基础模型，提供公开权重、微调流程与 Tokenizer 规范，契合“小物理接口 + 既有模型迁移”。 | [IBM/terramind](https://github.com/IBM/terramind) |
| **TerraTorch** | ⭐⭐⭐⭐ | **统一微调与任务适配架构**<br>基于 PyTorch Lightning 与 TorchGeo 构建的地球观测基础模型微调与评测工具箱。 | [torchgeo/terratorch](https://github.com/torchgeo/terratorch) |

---

## 🧪 推荐实验推进顺序

1. **Step 1：固化观测算子与数据管线**
   - 提取 `EnMAP-Box` 中的 SRF 卷积与重采样代码模块；
   - 建立统一的传感器参数表（中心波长、FWHM、响应曲线）；
   - 利用 `spectral` 构建标准的高光谱数据读写与评估管线。
2. **Step 2：跨传感器与 Backbone 对照**
   - 在 `HySens` 上评估 sensor-agnostic 方案的基线指标；
   - 接入 `HyperSIGMA` 作为对比 backbone，观察特征表征与光谱保真度。
3. **Step 3：小物理接口 + EO 基础模型迁移**
   - 避免从零训练大模型；
   - 基于 `TerraMind` 与 `TerraTorch` 挂载物理响应转换接口，在下游任务上对比直接 fine-tuning 与物理引导迁移的增益。

---

## 🛠️ 快速克隆与引用指南

> **建议**：为避免污染主仓库与冗余代码，优先采用 `git submodule` 锁定第三方仓库特定 commit，或保留链接与引用：

```bash
# 1. 克隆本仓库
git clone https://github.com/ShijiFan/res1.git
cd res1

# 2. 按需添加外部依赖作为子模块 (示例)
git submodule add https://github.com/EnMAP-Box/enmap-box.git external/enmap-box
git submodule add https://github.com/zhu-xlab/HySens.git external/HySens
git submodule add https://github.com/WHU-Sigma/HyperSIGMA.git external/HyperSIGMA
git submodule add https://github.com/IBM/terramind.git external/terramind
git submodule add https://github.com/torchgeo/terratorch.git external/terratorch

# 3. 初始化并拉取子模块
git submodule update --init --recursive
```

---

## 📑 引用与相关资源

- **EnMAP-Box**: https://github.com/EnMAP-Box/enmap-box
- **HySens**: https://github.com/zhu-xlab/HySens
- **HyperSIGMA**: https://github.com/WHU-Sigma/HyperSIGMA
- **Spectral Python**: https://github.com/spectralpython/spectral
- **TerraMind**: https://github.com/IBM/terramind
- **TerraTorch**: https://github.com/torchgeo/terratorch
