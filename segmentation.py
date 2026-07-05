# =============================================================================
# 肺分割数学降级纯计算模块
# 负责：无 AI 模型 / ONNX 推理失败时，用纯数学连通域算法从 HU 体积粗分割双肺。
# 设计：无任何 Qt/UI/线程依赖，输入 HU 数组、输出 uint8 蒙版，故可脱离
#       AutoAIEngineThread 独立单元测试（见 tests/test_gui.py::test_lung_fallback）。
# =============================================================================

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndimage

from constants import LUNG_FALLBACK_LABEL


def segment_lungs_fallback(volume_hu: np.ndarray, air_hu: float = -300.0,
                           label: int = LUNG_FALLBACK_LABEL) -> np.ndarray:
    """纯数学肺分割降级。返回 uint8 蒙版（label=肺，0=背景），异常时返回全零。

    算法原理：肺部在 CT 中为低密度空气区域（HU < air_hu）。
      1. 阈值分割提取所有低密度区域（空气 + 肺）
      2. 3D 连通域标记，把相互接触的空气体素归为同一组
      3. 找出与六个边界面相交的连通域 → 体外背景空气，剔除
      4. 剩余内部空气再次连通域标记，取体积最大者为主肺；
         若次大 ≥ 主肺体积 5% 则一并纳入（双肺）
    """
    try:
        # 步骤1：阈值分割，提取所有低密度区域（空气 + 肺部）
        air_mask = (volume_hu < air_hu).astype(np.uint8)

        # 步骤2：3D 连通域标记，把相互接触的空气体素归为同一组
        labels, _ = ndimage.label(air_mask)

        # 步骤3：找出与六个边界面相交的连通域标签——这些是体外背景
        border_labels = (set(labels[0, :, :].flatten()) | set(labels[-1, :, :].flatten())
                         | set(labels[:, 0, :].flatten()) | set(labels[:, -1, :].flatten())
                         | set(labels[:, :, 0].flatten()) | set(labels[:, :, -1].flatten()))

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
        mask = np.zeros_like(internal_air)
        if len(counts) > 1:
            l1 = counts.argmax()
            mask[labels_int == l1] = label
            max_vol = counts[l1]
            counts[l1] = 0
            if counts.max() > max_vol * 0.05:
                l2 = counts.argmax()
                mask[labels_int == l2] = label
        return mask
    except Exception:
        # 任何异常均返回全零掩码，保证调用方（UI）不崩溃
        return np.zeros_like(volume_hu, dtype=np.uint8)
