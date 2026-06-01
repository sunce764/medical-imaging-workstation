# 医学影像工作站 Pro · Recon Lab

一款功能齐全的桌面医学影像工作站,集 **DICOM 阅片**、**多种 CT 重建算法** 与 **AI 肺部分割** 于一体,基于 PySide6 构建。

## 功能演示

![主界面](01_main.png)
![DFR 重建](02_dfr.png)
![FBP 对比](03_fbp.png)

## 技术栈

Python · PySide6 · PyDicom · NumPy · SciPy · scikit-image · ONNX Runtime

## 核心功能

**影像阅片**
- 233 层真实患者 CT,支持 Axial / Coronal / Sagittal 三方向 MPR 导航
- 6 种临床预设窗位(肺 / 纵隔 / 骨 / 血管 / 腹部 / 脑)
- 缩放、平移、交互调窗、十字线联动、测量与病灶标注

**CT 重建实验室**
- 解析法:BP(反投影)、FBP(滤波反投影,5 种滤波器)、DFR(直接傅里叶重建)
- 迭代法:ART、SIRT、DMR(基于系统矩阵的代数重建)
- 正弦图(sinogram)生成与重建结果对比

**AI 辅助**
- ONNX 异步后台推理引擎,自动肺部分割(daemon 线程,不阻塞 UI)

## 项目结构

| 文件 | 职责 |
|------|------|
| `main.py` | 主窗口与业务逻辑(`MedicalViewer`) |
| `graphics_view.py` | 自定义 `QGraphicsView`:影像交互 / 调窗 / 标注 / MPR 十字线 |
| `recon.py` | 纯计算重建算法(BP/FBP/DFR/ART/SIRT/DMR),不依赖 Qt |
| `ai_engine.py` | 异步 AI 肺分割推理引擎 |
| `constants.py` | 跨 UI 与业务层共享的常量(工具 ID / MPR 平面) |
| `style.qss` | Qt 界面样式 |

> 注:模型权重(`organs.onnx`)与 DICOM 影像数据体积较大、属敏感数据,**不包含在仓库中**,需自行准备并放入对应目录。

## 运行

```bash
pip install PySide6 pydicom numpy scipy scikit-image onnxruntime
python main.py
```

## 授权

详见 [LICENSE](LICENSE)。
