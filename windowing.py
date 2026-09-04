"""显示用窗位范围；不改变体素、不推断 HU 或其它物理单位。"""

import math

import numpy as np


def raw_display_window(volume, padding_ranges=()):
    """按有限存储值范围适配灰度，忽略逐片 DICOM padding（无 Qt）。

    返回 (WW, WL, WL 下限, WL 上限, WW 上限)。常量卷也保留非零窗宽；
    全 padding / 空卷退到 1/0，不用 HU 肺窗冒充未知单位的显示默认值。
    """
    low, high = math.inf, -math.inf
    for i, frame in enumerate(volume):
        values = np.asarray(frame)
        valid = np.isfinite(values)
        if i < len(padding_ranges) and padding_ranges[i][0] is not None:
            start, end = padding_ranges[i]
            end = start if end is None else end
            valid &= (values < min(start, end)) | (values > max(start, end))
        values = values[valid]
        if values.size:
            low = min(low, float(values.min()))
            high = max(high, float(values.max()))
    if not math.isfinite(low):
        low = high = 0
    width = max(1, math.ceil(high - low))
    level = round((high + low) / 2)
    return width, level, min(-1200, math.floor(low)), max(1200, math.ceil(high)), max(4000, width)
