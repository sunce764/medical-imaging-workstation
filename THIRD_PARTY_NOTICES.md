# 第三方组件与许可 · Third-Party Notices

本文件如实记录本项目集成或再分发的第三方成果及其**上游声明的许可**，并给出可自行核对的一手来源
URL。每条均在 2026-07-17 直读上游 `LICENSE` 原文 / PyPI 元数据核实。

> **免责**：本文件陈述的是「上游声明了什么」这一**事实**，附一手出处供复核；它**不是法律意见**。
> 在公开发布、商业使用或再分发前，请自行（或经法律专业人士）确认合规。
> 凡本文件标注为**待确认**的，即为一手来源无法给出定论者——**宁可标明不确定，也不臆断**。

---

## 一、AI 分割模型（本仓库**确在再分发**其计算图）

本仓库 git 追踪并分发 `models/organs.onnx`（45 KB，**仅计算图、不含权重**）。
119 MB 权重 `models/organs.onnx.data` **未**随本仓库分发（获取方式见 `README.md` →「模型说明」）。

| 项目 | 上游声明的许可 | 一手来源 |
|---|---|---|
| **TotalSegmentator**（代码） | **Apache-2.0** | [LICENSE 原文](https://raw.githubusercontent.com/wasserth/TotalSegmentator/master/LICENSE) |
| **TotalSegmentator 权重 — `total` 任务**（含 Task 291 `class_map_part_organs`，即本模型来源） | **Apache-2.0** | [README「Subtasks」节](https://github.com/wasserth/TotalSegmentator)：该任务列于「Openly available for any usage (Apache-2.0 license)」一组；权重托管于该 Apache-2.0 仓库自身的 [v2.0.0-weights release](https://github.com/wasserth/TotalSegmentator/releases) |
| **nnU-Net v2**（本模型的架构来源） | **Apache-2.0** | [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) |

**须知的上游细节（避免外推）**：TotalSegmentator **按任务分档授权**，并非全仓统一。除上表的
`total` 任务（Apache-2.0）外，另有若干任务「需单独许可（非商业免费，商业需联系上游）」，
`brain_aneurysm` 更是 CC-BY-NC-4.0 且无商业许可。**本项目仅使用 `total` 任务的 Task 291**，
落在 Apache-2.0 一档。**若将来更换模型，须重新核对该任务的授权档位，不可由本条外推。**

**再分发义务（依 Apache-2.0 §4）**：
- **§4(a) 需向接收者提供一份 Apache-2.0 许可副本** —— 本文件的链接与本节即为此目的；如需完整文本见
  <https://www.apache.org/licenses/LICENSE-2.0>。
- **§4(b) 修改声明**：`models/organs.onnx` **不是上游原始文件**，而是由上游权重经
  `torch.onnx` 导出的计算图（导出器字符串为 `pytorch 2.11.0`）。特此声明该改动。
- **§4(d) NOTICE 传递义务：不触发** —— 该条以「上游 Work 自带 NOTICE 文件」为前提，而
  TotalSegmentator 与 nnU-Net 的仓库根目录**均无 NOTICE 文件**（已经 GitHub Contents API 核实）。
  故本项目**不主张、也不虚构**任何 NOTICE 义务。
- **学术引用**（上游 README 要求，属学术惯例而非许可强制）：
  - Wasserthal J. et al. *TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images.*
    Radiology: Artificial Intelligence, 2023.
  - Isensee F. et al. *nnU-Net: a self-configuring method for deep learning-based biomedical image
    segmentation.* Nature Methods, 2021.

**待确认**：45 KB 的**无权重计算图**是否构成上游权重的 *Derivative Work*（Apache-2.0 §1），
一手来源无法给出定论——§1 同时规定「Derivative Works shall not include works that remain
separable from, or merely link (or bind by name) to the interfaces of, the Work」，而本图仅携带
层命名结构、不含权重数值。本项目**按「构成」从严处理**（即照 §4 履行上述义务），但如实标注该
定性本身未有定论。

---

## 二、数据集（**均未**随本仓库分发）

| 数据 | 许可 | 用途 | 一手来源 |
|---|---|---|---|
| **TotalSegmentator-CT-Lite**（`s0029` 单例） | **CC-BY-4.0** | `experiments/seg_validate.py` 的分割验证 | [HuggingFace: YongchengYAO/TotalSegmentator-CT-Lite](https://huggingface.co/datasets/YongchengYAO/TotalSegmentator-CT-Lite) |
| **TotalSegmentator 上游原始数据集**（CT-Lite 的母本） | **CC-BY-4.0** | 同上（间接） | [Zenodo 10047292](https://zenodo.org/records/10047292) |
| **RIDER Lung CT**（本地 `肺癌/`） | TCIA 公开数据集；已由 **CTP 去标识**（`PatientIdentityRemoved=YES`） | 完整回归测试的本地数据 | [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/) |

**CC-BY-4.0 的署名要求**：使用上述数据产出的图表（`experiments/results/` 下的 `seg_confusion.png`
等）已在 `README.md`、`docs/technical_report.md`、`experiments/README.md` 中标注来源与许可。
**本仓库不转载任何上述数据本身。**

---

## 三、运行时依赖（仅 `import` 使用，**未** vendored 其源码）

版本以 `requirements.txt` / `experiments/requirements-experiments.txt` 锁定为准。

| 组件 | 版本 | 上游声明的许可 | 一手来源 |
|---|---|---|---|
| **PySide6**（Qt for Python） | 6.11.0 | **LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only**（被许可人择一；另由 The Qt Company 提供独立商业许可）。本项目适用 **LGPL-3.0-only** | [PyPI](https://pypi.org/project/PySide6/) · [Qt 许可](https://www.qt.io/licensing/) |
| **shiboken6**（PySide6 绑定运行时，随 PySide6 自动装入） | 6.11.0 | 同 PySide6 | [PyPI](https://pypi.org/project/shiboken6/) |
| **pydicom** | 3.0.2 | **MIT** | [PyPI](https://pypi.org/project/pydicom/) |
| **NumPy** | 2.2.6 | **BSD-3-Clause** | [PyPI](https://pypi.org/project/numpy/) |
| **SciPy** | 1.15.3 | **BSD-3-Clause** | [PyPI](https://pypi.org/project/scipy/) |
| **scikit-image** | 0.25.2 | **BSD-3-Clause**（主体）；**非单一许可**——另含 BSD-2-Clause 与 MIT 覆盖的特定文件 | [PyPI](https://pypi.org/project/scikit-image/) |
| **ONNX Runtime** | 1.23.2 | **MIT** | [PyPI](https://pypi.org/project/onnxruntime/) |
| **NiBabel** | 5.4.2 | **MIT**（核心包）；其 `COPYING` 另含 BSD-3-Clause / PDDL-1.0 / 自定义宽松许可覆盖的附带组件 | [PyPI](https://pypi.org/project/nibabel/) |
| **Matplotlib** | 3.10.8 | **Matplotlib License Agreement**（其自有许可，基于 PSF 许可改写、BSD-compatible）。**注意：SPDX 无对应 identifier，勿标作 `PSF-2.0`** | [PyPI](https://pypi.org/project/matplotlib/) |
| **remotezip** | 0.12.3 | **MIT** | [PyPI](https://pypi.org/project/remotezip/) |

上述组件的著作权与许可归各自作者所有，本项目未修改其源码，亦未随仓库分发其代码或二进制。

**待确认（LGPL）**：本项目以纯 Python 源码形式 `import` PySide6、**不静态链接、不分发任何 Qt 二进制**。
该形态下 LGPL-3.0 的具体义务边界（是否需随附 LGPL 许可文本、是否需提供「可重新链接」声明），
以及它与本仓库 `LICENSE`（"All rights reserved"、仅供审阅的专有式许可）的相容性判断，
**本文件不下结论**——该问题的对抗性核验未能完成，且属法律判断。**在公开仓库或对外分发前应予确认。**

---

## 四、本项目自身

本项目自研代码的著作权见 [`LICENSE`](LICENSE)：© 2026 盛超 (Sheng Chao)、赖胜圣 (Lai Shengsheng)，
两位共同著作权人共有。

---

*本文件由代码与上游一手来源核对生成，最后核实日期：2026-07-17。上游许可可能随时间变更，
复核时请以当时的上游 `LICENSE` 原文为准。*
