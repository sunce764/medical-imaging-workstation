# 医学影像工作站

一款功能齐全的医学影像工作站，具备 CT 重建（BP/FBP/DFR）和 AI 分割功能，采用 PySide6 和 PyDicom 构建。

## 功能演示

![主界面](01_main.png)
![DFR重建](02_dfr.png)
![FBP对比](03_fbp.png)

## 技术栈
- Python · PySide6 · PyDicom · NumPy · Scikit-Image · SciPy · ONNX

## 核心功能
- 233层真实患者CT导航，支持Axial/Coronal/Sagittal三方向MPR
- 6种临床预设窗（肺/纵隔/骨/血管/腹部/脑）
- BP、FBP（5种滤波器）、DFR三套重建算法
- ONNX异步AI肺部分割引擎
