# 医学影像工作站 + 重建实验室

[English](README.md) · **简体中文**

[![CI](https://github.com/sunce764/medical-imaging-workstation/actions/workflows/ci.yml/badge.svg)](https://github.com/sunce764/medical-imaging-workstation/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![GUI](https://img.shields.io/badge/GUI-PySide6%20%2F%20Qt6-41CD52?logo=qt&logoColor=white)
[![License](https://img.shields.io/badge/License-Proprietary-lightgrey)](LICENSE)
![非医疗器械](https://img.shields.io/badge/⚠️-教学%2F科研·非医疗器械-critical)

基于 **PySide6（Qt6）** 的桌面 **CT 影像工作站**，在一个应用内集成**临床阅片**、**AI 多器官分割**与 **CT 断层重建教学实验室**。除软件本身，仓库另附**两项可复现的量化研究**，把内置算法转化为可测的结论。

> **定位声明**：本软件为**影像教学 / 科研工具**，**不是经认证的医疗器械，不得用于临床诊断**。AI 分割与器官定量为自动推断，仅供参考。

![AI 多器官分割叠加](docs/img/gui_axial_segmentation.png)

---

## 亮点

- **临床阅片** —— 并行读盘、按解剖 Z 坐标排序；三平面 MPR + 十字线联动；6 套临床窗位预设；测量 / 标注 / 椭圆 ROI 密度；双序列随访对比（解剖配准）。
- **AI 多器官分割** —— 后台滑窗 ONNX 推理（25 类，含 5 个肺叶）。所用模型原本来源未知，其出处经**实测反推确证** —— 即 TotalSegmentator v2 `class_map_part_organs`，在一例带真值的公开 CT 上 21 个在场器官**平均 Dice ≈ 0.92**（*n = 1*）。
- **重建实验室（教学）** —— 正向 Radon 与解析反投影（BP、含 5 种滤波器的 FBP）基于 scikit-image；**DFR、DMR、ART、SIRT 四种反解算法为本项目从零实现**。
- **两项量化研究** —— 重建剂量-质量权衡（含一个反直觉的滤波器翻转发现）与 AI 模型出处/Dice 验证，均直调产品代码、完全可复现、不使用患者数据。
- **面向审阅的工程** —— God-object 拆分为 5 个 UI mixin + 4 个无 Qt 计算模块；**165 项**离屏 Qt 回归测试 + CI（重建算法有数值正确性断言，而非仅验「有限」）；防御式 DICOM 处理。

## 界面

| AI 多器官分割 | 三平面 MPR + 十字线联动 |
|:---:|:---:|
| ![轴位分割](docs/img/gui_axial_segmentation.png) | ![三平面 MPR](docs/img/gui_mpr_triplanar.png) |

> 演示数据为公开的 **TotalSegmentator-CT-Lite**（CC-BY-4.0）—— 非患者数据、无 PHI。

## 快速开始

```bash
conda env create -f environment.yml     # 创建 dicom_gui 环境（Python 3.10）
conda activate dicom_gui
python main.py                           # 空载启动（不加载任何数据）
python main.py --data /path/to/dicom_dir # 或启动即加载指定 DICOM 目录
```

- **CPU-only，无需 GPU。** 整卷 AI 推理约 100 秒。
- 模型权重（`models/organs.onnx.data`，119MB）**未随仓库分发**；缺权重时自动降级为纯数学连通域分割。获取方式见 [架构文档 → 模型](docs/ARCHITECTURE.md#segmentation-model)。

## 功能

| 模块 | 能力 |
|---|---|
| **临床阅片** | 解剖排序 DICOM 加载 · 三平面 MPR + 十字线联动 · 6 套窗位预设 + 反色 · 9 个测量/标注工具 · 椭圆 ROI（均值±SD / 最值 HU / 面积）· 四角 PACS 叠加 · Cine 播放 · 双序列随访对比 |
| **AI 分割** | 后台自动推理 · 三平面彩色叠加 + 可点图例 · 光标 HUD（HU / 坐标 / 器官）· 器官定量（体积 mL、均值±SD、中位数、p5–p95、最值 HU）+ CSV 导出 · 画笔/橡皮编辑 + 撤销 |
| **重建实验室** | Radon 投影（60–360°，1–4× 采样）· BP / FBP（5 种滤波器）/ DFR · DMR（最小二乘）/ ART / SIRT，附误差图 + RMSE |
| **合规** | 显示层脱敏 · 常驻 AI 免责声明 · 中英双语界面切换 |

## 量化研究

两项研究均直调产品代码、不使用患者数据。方法、图表与结论见[技术报告](docs/technical_report.md)。

- **研究一 —— CT 重建的剂量-质量权衡。** 在 Shepp-Logan 体模上，误差在 ≈ 180 视角后饱和；最优 FBP 滤波器随剂量**翻转**（稀疏角平滑滤波器胜出，稠密角锐利 Ram-Lak 胜出）；在 Poisson 光子噪声下，约束迭代（ART）最鲁棒，而朴素最小二乘在近方阵区失稳。整理为[预印本稿](docs/preprint_recon.md)。
- **研究二 —— 出处确证与 Dice 验证。** 在一例带真值的公开 CT 上跑该未文档化 ONNX 模型、算标签重叠混淆矩阵，恢复出标签映射（恒等对角线），21 个器官平均 Dice ≈ 0.92 —— 同时验证推理管线正确并纠正两处错标。

复现见 [`experiments/`](experiments/README.md)（脚本 + 图表 + CSV）。

## 测试

```bash
python tests/test_gui.py                     # 完整回归：165 项（需同目录 RIDER 真实数据）
SKIP_REAL_DATA=1 python tests/test_gui.py    # 数据无关子集（CI 使用）
ruff check .                                 # 静态检查
coverage run tests/test_gui.py && coverage report
```

离屏 Qt，退出码 0 = 全部通过。覆盖率 ≈ 70%；四个无 Qt 计算模块（`recon` / `quantify` / `segmentation` / `mpr_geometry`）均有独立单测。CI 每次 push/PR 只跑数据无关子集 —— 故「CI 全绿」**不等于**全部 165 项都跑过（交互层测试需本地真实数据）。

## 文档

| 文档 | 语言 | 内容 |
|---|---|---|
| [架构说明](docs/ARCHITECTURE.md) | 英文 | 模块布局、God-object 分解、分割模型逆向工程、AI 管线契约 |
| [软件说明书](docs/manual_zh.md) · [English](docs/manual_en.md) · [PDF](docs/manual_zh.pdf) | 中文 · EN | 按功能逐一图文说明的完整用户手册 |
| [技术报告](docs/technical_report.md) | 英文 | 两项量化研究 —— 方法、图表、结论 |
| [预印本稿（研究一）](docs/preprint_recon.md) | 英文 | 稀疏视角 / 低剂量重建，学术格式 |
| [实验](experiments/README.md) | 英文 | 可复现脚本 + 图表 + CSV |
| [变更记录](CHANGELOG.md) | 英文 | 缺陷排查与修复的审查小结 |
| [第三方许可](THIRD_PARTY_NOTICES.md) | 英文 | 集成组件的许可（已对上游核实） |

## 已知限制

- **非临床器械** —— 无监管认证、无算法验证文档、无审计 / 访问控制。
- **脱敏为显示层** —— 隐去屏幕与导出文件名中的 PHI，但**不清洗**底层 DICOM 标签与烧录文字。
- **Dice ≈ 0.92 基于单一带真值序列（*n = 1*）** —— 标签映射已确证，但该数字不应被当作总体估计。
- 矩阵重建（DMR/ART）受 `lstsq` 成本限制，实用上限约 64×64（教学用途）；AI/重建/标注默认作用于当前主序列。
- 双序列随访对比只做配准与并排显示，**不给出任何定量变化**（无两次扫描间的体积或 HU 差值）。

## 许可 · 版权

© 2026 **盛超 (Sheng Chao)、赖胜圣 (Lai Shengsheng)**。保留所有权利。

本软件为两位共同著作权人共有（已按此向中国版权保护中心申请软件著作权登记）。本仓库仅供教学 / 科研与作品集审阅，**未授予**任何复制、修改或再分发许可，如需使用请联系著作权人。集成的第三方组件依其各自许可 —— 见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
