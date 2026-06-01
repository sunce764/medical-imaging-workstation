# =============================================================================
# AI 推理引擎模块
# 负责：异步后台 AI 肺部分割推理
# 设计：纯 Python daemon 线程，避免继承 QThread 的析构崩溃风险
# =============================================================================

import os
import time
import threading
import numpy as np
import scipy.ndimage as ndimage
from PySide6.QtCore import QTimer

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


class AutoAIEngineThread:
    """在后台线程中执行 AI 肺部分割推理，完成后通过 QTimer 安全回调主线程。"""

    def __init__(self, volume_hu, callback, model_path="lung_seg_model.onnx"):
        # volume_hu: 完整的 3D HU 值体素数组，shape=(Z, H, W)，float32
        # callback: 推理完成后调用，签名为 callback(mask_array, elapsed_ms)
        # model_path: ONNX 模型文件路径，相对于当前工作目录
        self.volume_hu = volume_hu
        self.callback = callback
        self.model_path = model_path
        self._thread = None

    def start(self):
        # daemon=True 确保主窗口关闭后不会因后台线程阻塞进程退出
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def isRunning(self):
        return self._thread is not None and self._thread.is_alive()

    def _run(self):
        """推理主体，运行在后台线程，严禁在此处操作任何 Qt 对象（非线程安全）。"""
        start_t = time.perf_counter()

        # HU 值归一化到 [0, 1]：肺窗范围 -1000~400 HU 覆盖空气到软组织
        # 超出范围的 HU 值（骨骼 > 400）通过 clip 截断，防止影响网络输入分布
        norm_vol = np.clip(self.volume_hu, -1000, 400)
        norm_vol = (norm_vol - (-1000)) / (400 - (-1000))

        final_mask = None

        # === 路径1：真实 ONNX 深度学习推理 ===
        if HAS_ONNX and os.path.exists(self.model_path):
            try:
                # 医学分割模型标准输入维度：(Batch=1, Channel=1, D, H, W)
                # np.newaxis 两次相当于在最前面插入两个维度
                input_tensor = norm_vol[np.newaxis, np.newaxis, ...].astype(np.float32)
                session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
                input_name = session.get_inputs()[0].name
                ort_outs = session.run(None, {input_name: input_tensor})
                # 假设模型输出为 sigmoid 概率图，阈值 0.5 进行二值化分割
                pred_prob = ort_outs[0][0, 0, ...]
                final_mask = (pred_prob > 0.5).astype(np.uint8)
            except Exception as e:
                print(f"ONNX 推理失败，降级为数学算法: {e}")

        # === 路径2：纯数学算法降级（无模型文件时自动启用）===
        # 算法原理：肺部在 CT 中为低密度空气区域（HU < -300）
        # 先找所有空气区域，剔除与图像边界相连的"体外空气"（背景），
        # 剩余的内部空气连通域即为左右肺，取体积最大的两个。
        if final_mask is None:
            try:
                # 步骤1：阈值分割，提取所有低密度区域（空气 + 肺部）
                air_mask = (self.volume_hu < -300).astype(np.uint8)

                # 步骤2：3D 连通域标记，把相互接触的空气体素归为同一组
                labels, _ = ndimage.label(air_mask)

                # 步骤3：找出与六个边界面相交的连通域标签——这些是体外背景
                border_labels = set(labels[0,:,:].flatten()) | set(labels[-1,:,:].flatten()) | \
                                set(labels[:,0,:].flatten()) | set(labels[:,-1,:].flatten()) | \
                                set(labels[:,:,0].flatten()) | set(labels[:,:,-1].flatten())

                # 步骤4：从空气掩码中剔除所有边界连通域，留下纯内部空气（即肺）
                internal_air = np.copy(air_mask)
                for bl in border_labels:
                    if bl != 0:
                        internal_air[labels == bl] = 0

                # 步骤5：对内部空气再次连通域标记，分离左右肺
                labels_int, _ = ndimage.label(internal_air)
                counts = np.bincount(labels_int.flatten())
                counts[0] = 0  # 标签0是背景，排除在外

                # 步骤6：取体积最大的连通域为主肺叶，若第二大超过主肺的 5% 则一并纳入（双肺）
                final_mask = np.zeros_like(internal_air)
                if len(counts) > 1:
                    l1 = counts.argmax()
                    final_mask[labels_int == l1] = 1
                    max_vol = counts[l1]
                    counts[l1] = 0
                    if counts.max() > max_vol * 0.05:
                        l2 = counts.argmax()
                        final_mask[labels_int == l2] = 1
            except Exception:
                # 任何异常均返回全零掩码，保证 UI 不崩溃
                final_mask = np.zeros_like(self.volume_hu, dtype=np.uint8)

        end_t = time.perf_counter()
        # 关键：QTimer.singleShot 把回调投递到主线程的事件循环，
        # 这是从后台线程安全更新 Qt UI 的标准做法（禁止在子线程直接调用任何 Qt 方法）
        QTimer.singleShot(0, lambda: self.callback(final_mask, (end_t - start_t) * 1000))
