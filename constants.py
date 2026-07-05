# =============================================================================
# 全局常量定义模块
# 负责：跨 UI 与业务逻辑层共享的工具/平面/视图标识符
# 提取动机：原先散落在 graphics_view.py（UI 组件模块）中却被 main.py 业务层
#           大量引用，导致业务层不必要地依赖 UI 组件模块；独立 constants.py
#           让两侧都从中性模块导入，解开耦合。
# =============================================================================

import os as _os

import numpy as _np

# 工具栏工具 ID 枚举（用整数而非 Enum 方便与 QButtonGroup.idClicked 直接对接）
# 前 6 个为原有工具；6/7=分割修正（画笔/橡皮）；8=椭圆 ROI 密度测量
(TOOL_POINTER, TOOL_RULER, TOOL_DRAW, TOOL_CROP, TOOL_RECT_CROP, TOOL_AI_TRACK,
 TOOL_SEG_BRUSH, TOOL_SEG_ERASE, TOOL_ROI) = range(9)

# MPR 三平面常量，与 combo_plane 下拉框的索引严格对应
AXIAL = 0      # 横断面：沿 Z 轴切片，最常用的阅片视角
CORONAL = 1    # 冠状面：沿 Y 轴切片，前后方向
SAGITTAL = 2   # 矢状面：沿 X 轴切片，左右方向

# =============================================================================
# AI 多器官分割相关常量
# =============================================================================
# organs.onnx：25 类胸腹多器官分割模型（含 5 个肺叶类）。
# 外部权重 organs.onnx.data 必须与 .onnx 同目录（ONNX 相对 onnx 文件路径解析）。
MODEL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "models", "organs.onnx")
# 器官名候选表（由真实数据解剖推断，可被官方标签表替换）
LABELS_JSON = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "models", "organ_labels_candidate.json")

# 手动 3D 追踪结果占用的专属标签值，避开 0-24 的模型类别，防止与器官语义冲突
MANUAL_TRACK_LABEL = 255
# 数学降级算法（无模型时）分出的肺，复用右肺叶标签以显示为肺色
LUNG_FALLBACK_LABEL = 10

# 25 类器官调色板（RGB）。索引即类别号；0=背景在 LUT 中设为全透明。
# 肺叶用红橙(右)/蓝青(左)区分左右；心脏粉、气管黄、腹部器官各异。
LABEL_COLORS = {
    1:  (229, 57, 53),    # 主动脉/大血管
    5:  (236, 64, 122),   # 心脏
    6:  (161, 136, 127),  # 肠道/直肠气体
    10: (239, 83, 80),    # 右肺叶
    11: (255, 138, 101),  # 右肺叶
    12: (66, 165, 245),   # 左肺叶
    13: (41, 182, 246),   # 左肺叶
    14: (38, 198, 218),   # 左肺叶
    15: (120, 144, 156),  # 颈部软组织
    16: (255, 235, 59),   # 气管
    18: (156, 204, 101),  # 脾
    19: (255, 213, 79),   # 膀胱?
    20: (255, 167, 38),   # 胃
    21: (141, 110, 99),   # 肝
    22: (224, 224, 224),  # 骨/结肠?
    23: (126, 87, 194),   # 肾/肾上腺?
    MANUAL_TRACK_LABEL: (255, 255, 255),  # 手动 3D 追踪：白色
}
# 未列出的 unknown 类（2,3,4,7,8,9,17,24）落到此暗灰，基本不出现
_UNKNOWN_COLOR = (96, 96, 96)
_MASK_ALPHA = 110  # 蒙版叠加透明度（0-255）

# 预构建 (256,4) RGBA 查找表：ov = LABEL_LUT[slice_labels] 一步向量化上色
LABEL_LUT = _np.zeros((256, 4), dtype=_np.uint8)
for _i in range(1, 25):
    _c = LABEL_COLORS.get(_i, _UNKNOWN_COLOR)
    LABEL_LUT[_i] = (*_c, _MASK_ALPHA)
LABEL_LUT[MANUAL_TRACK_LABEL] = (*LABEL_COLORS[MANUAL_TRACK_LABEL], _MASK_ALPHA)
