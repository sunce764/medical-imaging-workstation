# =============================================================================
# 全局常量定义模块
# 负责：跨 UI 与业务逻辑层共享的工具/平面/视图标识符
# 提取动机：原先散落在 graphics_view.py（UI 组件模块）中却被 main.py 业务层
#           大量引用，导致业务层不必要地依赖 UI 组件模块；独立 constants.py
#           让两侧都从中性模块导入，解开耦合。
# =============================================================================

# 工具栏工具 ID 枚举（用整数而非 Enum 方便与 QButtonGroup.idClicked 直接对接）
TOOL_POINTER, TOOL_RULER, TOOL_DRAW, TOOL_CROP, TOOL_RECT_CROP, TOOL_AI_TRACK = range(6)

# MPR 三平面常量，与 combo_plane 下拉框的索引严格对应
AXIAL = 0      # 横断面：沿 Z 轴切片，最常用的阅片视角
CORONAL = 1    # 冠状面：沿 Y 轴切片，前后方向
SAGITTAL = 2   # 矢状面：沿 X 轴切片，左右方向
