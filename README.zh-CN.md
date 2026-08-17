<h1 align="center">医学影像工作站 + 重建实验室</h1>

<p align="center"><strong>CT 阅片 · AI 分割 · 可复现重建研究</strong></p>

<p align="center"><a href="README.md">English</a> · <strong>简体中文</strong></p>

<p align="center">
  <a href="https://github.com/sunce764/medical-imaging-workstation/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/sunce764/medical-imaging-workstation/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python 3.10" src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&amp;logoColor=white">
  <img alt="PySide6 / Qt6" src="https://img.shields.io/badge/GUI-PySide6%20%2F%20Qt6-41CD52?logo=qt&amp;logoColor=white">
  <a href="LICENSE"><img alt="专有许可" src="https://img.shields.io/badge/License-Proprietary-lightgrey"></a>
  <img alt="仅供教学科研" src="https://img.shields.io/badge/⚠️-教学%2F科研·非医疗器械-critical">
</p>

基于 **PySide6（Qt6）** 的桌面 **CT 影像工作站**，在一个应用内整合临床式 DICOM 阅片、AI 多器官分割与断层重建教学实验室。仓库另附**三项正式记录的量化研究及一项 57 例扩展验证**，把产品代码转化为可测、可复现的证据。

> [!WARNING]
> **仅供教学与科研。** 本软件不是经认证的医疗器械，不得用于临床诊断。AI 分割和器官定量均为自动估计，不构成临床结论。

![AI 多器官分割叠加](docs/img/gui_axial_segmentation.png)

## 一览

| 产品 | 运行环境 | 证据 | 数据边界 |
|---|---|---|---|
| DICOM 阅片 + AI 分割 + 重建实验室 | Python 3.10 · PySide6/Qt6 · CPU-only | 研究 I–III + 57 例肺叶验证 | 合成模体与公开去标识研究 CT；仓库不提交 PHI |

## 界面

| AI 多器官分割 | 三平面 MPR + 十字线联动 |
|:---:|:---:|
| ![轴位分割](docs/img/gui_axial_segmentation.png) | ![三平面 MPR](docs/img/gui_mpr_triplanar.png) |

> 截图使用 **TotalSegmentator-CT-Lite**（CC-BY-4.0）公开去标识研究数据；本仓库不包含 PHI。

## 核心能力

| 模块 | 能力 |
|---|---|
| **临床阅片** | 解剖排序 DICOM 加载 · 三平面 MPR + 十字线联动 · 6 套窗位预设 + 反色 · 三平面厚层 MIP / MinIP / AIP · 9 个测量和标注工具 · 椭圆 ROI 统计 · 四角 PACS 叠加 · Cine 播放 · 双序列随访对比 |
| **AI 分割** | 25 类后台滑窗 ONNX 推理（含 5 个肺叶）· 三平面彩色叠加与可点图例 · 光标 HUD · 器官统计与 CSV 导出 · marching cubes 三维表面预览、形状特征与 STL 导出 · 画笔/橡皮编辑和撤销 · 逐体素置信度 |
| **重建实验室** | 内置解析 Shepp-Logan 模体 · Radon 投影 · BP / 含 5 种滤波器的 FBP / DFR · 从零实现的 DMR、ART、SIRT · 误差图与 RMSE · 学习式 CNN 后处理，并在界面展示训练视角和输入滤波器限制 |
| **安全与审阅** | 显示层脱敏 · 常驻 AI 免责声明 · 模型说明卡（实测出处及未测边界）· 中英双语界面 |

## 快速开始

```bash
conda env create -f environment.yml     # 创建 Python 3.10 环境
conda activate dicom_gui
python main.py                           # 空载启动
python main.py --data /path/to/dicom_dir # 或启动时加载 DICOM 目录
```

- **CPU-only，无需 GPU。** 在参考机器上整卷 AI 推理约需 100 秒。
- 模型权重（`models/organs.onnx.data`，119 MB）**不随仓库分发**；缺失时自动降级为经典连通域算法。详见[架构说明 → 模型](docs/ARCHITECTURE.md#segmentation-model)。

## 量化证据

实验直接调用随软件发布的产品管线，而非另写一份替代实现。研究 I–III 见[技术报告](docs/technical_report.md)，脚本和已提交结果由 [`experiments/`](experiments/README.md) 统一索引。

| 证据线 | 实测结果 | 适用边界 |
|---|---|---|
| **研究 I —— 重建剂量-质量** | 误差在 ≈180 视角后饱和；最优 FBP 滤波器从稀疏角的平滑滤波切换为稠密角的锐利 Ram-Lak；Poisson 光子噪声下 ART 是已测方法中最鲁棒者。 | 解析二维 Shepp-Logan 模体；矩阵法限制在 ≈64×64。[预印本稿](docs/preprint_recon.md) |
| **研究 II —— 模型出处与 Dice** | 标签重叠混淆矩阵将未文档化 ONNX 模型确认为 TotalSegmentator v2 `class_map_part_organs`；21 个在场器官平均 Dice ≈0.92，并纠正两处标签错误。 | 仅一例带真值的公开 CT（*n = 1*）；可支持出处判断，但不能给出总体 Dice。 |
| **研究 III —— 学习式稀疏角重建** | 自实现 1.9M 参数残差 U-Net 将 RMSE 降低 **3–6 倍**，病灶对比度保留率从 0.87 提升至 **0.96–1.00**；虚假结构率为 1.7%，分布外增益比为 0.81。 | 使用无噪声合成投影；幻觉率是有利条件下的下界，不能外推至光子饥饿的低剂量 CT。 |
| **扩展验证 —— 肺叶** | 57 例公开 CT 的五肺叶平均 Dice 为 **0.8867**（95% CI **[0.859, 0.914]**）；右肺上叶为 0.727，而原单例为 0.967。 | 只验证五个肺叶；原 21 器官总体数值仍是单例估计。[`seg3d_eval.py`](experiments/seg3d_eval.py) · [`seg3d_report.py`](experiments/seg3d_report.py) |

## 工程与测试

- 原 God-object 已拆分为 **5 个 UI mixin + 8 个无 Qt 计算模块**。
- CI 运行 **307 项数据无关检查**；依赖本地研究数据的交互测试不在 CI 内。
- 重建算法测试断言数值正确性，而非只检查输出“有限”；DICOM 读取对畸形元数据作防御处理。

```bash
python tests/test_gui.py                     # 完整回归；需本地 RIDER 数据
SKIP_REAL_DATA=1 python tests/test_gui.py    # CI 使用的 307 项数据无关检查
ruff check .                                 # 静态检查
coverage run tests/test_gui.py && coverage report
```

<details>
<summary><strong>覆盖率详情</strong></summary>

离屏 Qt 覆盖率约 79%。八个无 Qt 计算模块（`recon`、`quantify`、`segmentation`、`mpr_geometry`、`followup`、`projection`、`mesh3d`、`registration`）独立单测覆盖 77–100%；合成鼠标 press/move/release 序列会断言信号载荷（`graphics_view` 91%）。重建实验室 UI 调度（`recon_lab` 44%）仍是覆盖最低层。因此，CI 全绿只代表数据无关子集通过，不等于所有本地数据交互测试均已运行。

</details>

## 文档

| 文档 | 语言 | 用途 |
|---|---|---|
| [架构说明](docs/ARCHITECTURE.md) | 英文 | 模块图、God-object 分解、分割模型出处和 AI 管线契约 |
| [软件说明书](docs/manual_zh.md) · [English](docs/manual_en.md) · [PDF](docs/manual_zh.pdf) | 中文 · EN | 按界面截图讲解全部用户功能 |
| [技术报告](docs/technical_report.md) | 英文 | 研究 I–III 的方法、图表与结果 |
| [预印本稿 —— 研究 I](docs/preprint_recon.md) | 英文 | 学术格式的稀疏视角 / 低剂量重建研究 |
| [实验](experiments/README.md) | 英文 | 可复现实验脚本、图表与 CSV 输出 |
| [变更记录](CHANGELOG.md) | 英文 | 可审计的缺陷修复与审查记录 |
| [第三方许可](THIRD_PARTY_NOTICES.md) | 英文 | 已对上游核实的集成组件许可 |

## 安全边界与已知限制

- **非临床器械：**无监管认证、临床验证档案、审计追踪或访问控制。
- **仅做显示层脱敏：**屏幕与导出文件名会隐藏 PHI，但不会清洗底层 DICOM 标签和烧录文字。
- **AI 泛化仍有未测部分：**57 例扩展只覆盖五个肺叶，21 器官总体仍是 *n = 1*。输入 spacing 未重采样，因此 1.5 mm 各向同性实测范围之外的准确度未知。
- **重建限于教学范围：**DMR / ART 矩阵重建受最小二乘成本限制，实用上限约 64×64。研究 III 使用无噪声合成投影，不能证明低剂量临床表现。
- **随访为刚性而非形变配准：**平面内配准在测试中将整体平移造成的 MAE 从 321 HU 降至 13 HU，但不会校正呼吸引起的器官形变；差异结果只能作定性参考，不能视为临床变化量。

## 许可与版权

© 2026 **盛超（Sheng Chao）、赖胜圣（Lai Shengsheng）**。保留所有权利。

本软件由上述两位著作权人共同享有权利，已向中国版权保护中心提交列明双方的计算机软件著作权登记申请。本仓库仅供教学、科研与作品集审阅，**未授予任何复制、修改或再分发许可**；如需使用，请联系著作权人。集成的第三方组件依其各自许可，详见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
