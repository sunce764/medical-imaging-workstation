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
- **三项量化研究** —— 重建剂量-质量权衡（含一个反直觉的滤波器翻转发现）、AI 模型出处/Dice 验证，以及**学习式稀疏角重建研究：把「幻觉」实测出来而非假设不存在**。三项均直调产品代码、完全可复现、不使用患者数据。
- **面向审阅的工程** —— God-object 拆分为 5 个 UI mixin + 8 个无 Qt 计算模块；**325 项**离屏 Qt 回归测试 + CI（重建算法有数值正确性断言，而非仅验「有限」）；防御式 DICOM 处理。

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
| **临床阅片** | 解剖排序 DICOM 加载 · 三平面 MPR + 十字线联动 · 6 套窗位预设 + 反色 · **三平面厚层投影（MIP / MinIP / AIP）** · 9 个测量/标注工具 · 椭圆 ROI（均值±SD / 最值 HU / 面积）· 四角 PACS 叠加 · Cine 播放 · 双序列随访对比 + 差异定量 |
| **AI 分割** | 后台自动推理 · 三平面彩色叠加 + 可点图例 · 光标 HUD（HU / 坐标 / 器官）· 器官定量（体积 mL、均值±SD、中位数、p5–p95、最值 HU）+ CSV 导出 · **三维表面重建（marching cubes）—— 可拖动旋转预览 + 形状特征 + STL 导出** · 画笔/橡皮编辑 + 撤销 |
| **重建实验室** | Radon 投影（60–360°，1–4× 采样）· BP / FBP（5 种滤波器）/ DFR · DMR（最小二乘）/ ART / SIRT，附误差图 + RMSE · **学习式 CNN 后处理重建（研究三），训练视角与输入滤波器的限制直接标在界面上** |
| **合规** | 显示层脱敏 · 常驻 AI 免责声明 · 中英双语界面切换 |

## 量化研究

三项研究均直调产品代码、不使用患者数据。方法、图表与结论见[技术报告](docs/technical_report.md)。

- **研究一 —— CT 重建的剂量-质量权衡。** 在 Shepp-Logan 体模上，误差在 ≈ 180 视角后饱和；最优 FBP 滤波器随剂量**翻转**（稀疏角平滑滤波器胜出，稠密角锐利 Ram-Lak 胜出）；在 Poisson 光子噪声下，约束迭代（ART）最鲁棒，而朴素最小二乘在近方阵区失稳。整理为[预印本稿](docs/preprint_recon.md)。
- **研究三 —— 学习式稀疏角重建：它恢复了什么，又编造了什么。** 自实现的 1.9M 参数残差 U-Net 作 FBP 后处理，稀疏角 RMSE 较最优线性滤波低 **3–6 倍**，病灶对比度保留率从**与剂量无关的 0.87 天花板**（滤波器的固有代价，15–60 视角几乎不变）提升到 **0.96–1.00**。关键在于两种通常被一笔带过的失效被实测了：有/无病灶的配对模体把**虚假结构检出率钉在 1.7%（30% 阈值以上为 0%）**，而训练中从未出现的分布外形状（尖角方块、非凸多边形）保持 **0.81 的增益比**——说明改善来自通用去伪影而非形状记忆。唯一的真实极限在采样频率本身。
- **研究二 —— 出处确证与 Dice 验证。** 在一例带真值的公开 CT 上跑该未文档化 ONNX 模型、算标签重叠混淆矩阵，恢复出标签映射（恒等对角线），21 个器官平均 Dice ≈ 0.92 —— 同时验证推理管线正确并纠正两处错标。

复现见 [`experiments/`](experiments/README.md)（脚本 + 图表 + CSV）。

## 测试

```bash
python tests/test_gui.py                     # 完整回归：325 项（需同目录 RIDER 真实数据）
SKIP_REAL_DATA=1 python tests/test_gui.py    # 数据无关子集（CI 使用）
ruff check .                                 # 静态检查
coverage run tests/test_gui.py && coverage report
```

离屏 Qt，退出码 0 = 全部通过。覆盖率 ≈ 79% —— 八个无 Qt 计算模块（`recon` / `quantify` / `segmentation` / `mpr_geometry` / `followup` / `projection` / `mesh3d` / `registration`）均有独立单测，覆盖 77–100%；鼠标交互以合成的 press/move/release 序列驱动并断言发出的信号载荷（`graphics_view` 91%）。重建实验室的 UI 调度（`recon_lab` 44%）是目前覆盖最低的一层。CI 每次 push/PR 只跑数据无关子集 —— 故「CI 全绿」**不等于**全部 325 项都跑过（交互层测试需本地真实数据）。

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
- **随访对比可做平面内刚性配准，但无形变配准。** 两序列先按解剖 z 坐标对齐，勾选「配准」后再做平面内刚性配准（相位相关求平移 + 旋转搜索，实测把整体平移造成的 MAE 从 321 降到 13 HU），随后给出 Δ均值 / 平均绝对差 / RMSE / 差值图。**但呼吸导致的器官形变不会被校正**——刚性配准只处理体位,不改变解剖内部的相对关系。所报差异仍只宜作定性参考，不是临床意义上的变化量。器官级体积变化不可用，因为既往序列没有分割结果。

## 许可 · 版权

© 2026 **盛超 (Sheng Chao)、赖胜圣 (Lai Shengsheng)**。保留所有权利。

本软件为两位共同著作权人共有（已按此向中国版权保护中心申请软件著作权登记）。本仓库仅供教学 / 科研与作品集审阅，**未授予**任何复制、修改或再分发许可，如需使用请联系著作权人。集成的第三方组件依其各自许可 —— 见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
