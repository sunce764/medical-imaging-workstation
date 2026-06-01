# =============================================================================
# 医学影像自定义视图组件模块
# 负责：QGraphicsView 扩展，实现影像交互（缩放/平移/调窗/标注/MPR 十字线）
# =============================================================================

import math
import uuid

from PySide6.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                               QGraphicsLineItem, QGraphicsTextItem, QGraphicsPolygonItem,
                               QGraphicsRectItem, QGraphicsPathItem)
from PySide6.QtCore import Qt, QPoint, QRectF, QLineF, QPointF, Signal
from PySide6.QtGui import (QMouseEvent, QPainter, QPen, QColor, QFont,
                            QPolygonF, QBrush, QPainterPath, QPixmap)

# 常量集中定义在 constants.py，此模块只需要工具 ID（鼠标事件分发用）
from constants import (TOOL_POINTER, TOOL_RULER, TOOL_DRAW, TOOL_CROP,
                       TOOL_RECT_CROP, TOOL_AI_TRACK)


# =========================================================================
# 自定义医学影像视图组件
# 继承 QGraphicsView，在标准图形视图框架基础上叠加：
#   - 右键拖拽调节窗宽/窗位 (Windowing)
#   - 鼠标中键平移 (Pan)
#   - Ctrl+滚轮缩放 (Zoom)
#   - 普通滚轮切片 (Scroll)
#   - 标注工具：测距卡尺、自由画笔、套索、矩形截取、AI追踪框
#   - MPR 十字准线叠加
# =========================================================================
class MedicalGraphicsView(QGraphicsView):
    # --- 自定义信号，所有业务逻辑解耦到 MedicalViewer 中处理 ---
    clicked_pos = Signal(QPoint)       # 鼠标左键点击（指针工具），用于测量 HU 值
    wheel_scrolled = Signal(int)       # 滚轮滚动量（非 Ctrl），用于切换切片
    annotation_added = Signal(dict)    # 标注完成，携带标注数据字典
    crop_requested = Signal(list)      # 截取/套索完成，携带多边形顶点列表
    track_requested = Signal(QRectF)   # 3D 追踪框选完成，携带矩形区域
    annotation_deleted = Signal(str)   # Delete 键删除选中标注，携带标注 id
    window_changed = Signal(int, int)  # 右键拖拽调节窗宽窗位的增量 (dWW, dWL)
    mouse_hovered = Signal(QPoint)     # 鼠标移动，携带场景坐标（整数像素），用于 MPR 联动十字线

    def __init__(self, view_id):
        super().__init__()
        self.view_id = view_id         # 视图编号 1~4，用于日志和信号路由
        self.current_tool = TOOL_POINTER
        self.pixel_spacing = (1.0, 1.0)  # (行间距mm, 列间距mm)，从 DICOM PixelSpacing 读取

        # --- 场景与图层堆叠 ---
        # Z轴：image_item(0) < mask_item(1) < 十字线(2) < 标注(默认0，后续addItem时不设置)
        # mask_item 覆盖在影像上方，使用半透明 RGBA 叠加显示 AI 分割蒙版
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.image_item = QGraphicsPixmapItem()
        self.scene.addItem(self.image_item)
        self.mask_item = QGraphicsPixmapItem()
        self.mask_item.setZValue(1)   # 确保蒙版始终渲染在影像图层上方
        self.scene.addItem(self.mask_item)

        # --- MPR 十字准线（橙色虚线）---
        # 十字线初始位置 (0,0)，在 draw_crosshair() 中动态更新坐标
        self.vline = QGraphicsLineItem()
        self.hline = QGraphicsLineItem()
        pen_cross = QPen(QColor("#F39C12"), 1, Qt.DashLine)  # 橙色虚线，1px 不遮挡诊断信息
        self.vline.setPen(pen_cross); self.hline.setPen(pen_cross)
        self.vline.setZValue(2); self.hline.setZValue(2)     # 最顶层，始终可见
        self.scene.addItem(self.vline); self.scene.addItem(self.hline)
        self.vline.hide(); self.hline.hide()                 # 默认隐藏，MPR 联动开启后才显示

        # --- 渲染与交互设置 ---
        self.setRenderHint(QPainter.Antialiasing)            # 标注线条抗锯齿
        self.setRenderHint(QPainter.SmoothPixmapTransform)   # 影像双线性插值缩放（临床模式默认开启）
        self.setDragMode(QGraphicsView.NoDrag)               # 默认无拖拽，中键时临时切换为 ScrollHandDrag
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("background-color: #000000; border: none;")
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)  # 必须开启，否则鼠标未按下时不触发 mouseMoveEvent（影响 MPR 十字线）

        # --- 绘图状态变量 ---
        self.is_drawing = False          # 当前是否处于绘制/框选操作中
        self.is_windowing = False        # 右键拖拽调窗是否激活
        self.last_mouse_pos = None       # 上一帧鼠标位置，用于计算右键拖拽的增量
        self.temp_item = self.temp_rect_item = self.temp_text = self.start_pos = self.current_path = None
        self.polygon_points = []         # 套索/多边形工具的中间顶点列表

    def set_image(self, pixmap, mask_qimg=None, pixel_spacing=(1.0, 1.0)):
        """更新当前视图的影像和蒙版。

        fitInView 守卫逻辑：
          仅当变换矩阵的 m11（水平缩放因子）接近 1.0 时才执行自动适配。
          m11 == 1.0 意味着视图处于"原始/未缩放"状态（刚切换切片或初始化）。
          一旦用户通过 Ctrl+滚轮放大，m11 != 1.0，后续切片更新不会重置缩放，
          保留医生的放大查看状态——这是医学影像工作站的基本用户体验要求。
        """
        self.image_item.setPixmap(pixmap)
        self.pixel_spacing = pixel_spacing
        rect = pixmap.rect()
        # sceneRect 必须随图像尺寸更新，否则 fitInView 会基于旧尺寸计算缩放比，
        # 导致多平面（冠状/矢状面）分辨率不同时显示错位
        self.scene.setSceneRect(QRectF(rect))
        # 十字线覆盖整个图像范围，坐标系原点在图像左上角
        self.vline.setLine(0, 0, 0, rect.height())
        self.hline.setLine(0, 0, rect.width(), 0)
        if mask_qimg:
            self.mask_item.setPixmap(from_image(mask_qimg))
            self.mask_item.show()
        else:
            self.mask_item.hide()
        # m11 接近 1.0 的容差 1e-6，防止浮点误差将"未缩放"状态误判为"已缩放"
        if abs(self.transform().m11() - 1.0) < 1e-6:
            self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def draw_crosshair(self, x, y, show=True):
        """在场景坐标 (x, y) 处绘制 MPR 十字准线。
        坐标越界或 show=False 时自动隐藏，防止十字线超出影像范围造成视觉干扰。
        """
        if show and self.image_item.pixmap():
            w, h = self.image_item.pixmap().width(), self.image_item.pixmap().height()
            if 0 <= x < w and 0 <= y < h:
                self.vline.setLine(x, 0, x, h)
                self.hline.setLine(0, y, w, y)
                self.vline.show(); self.hline.show()
                return
        self.vline.hide(); self.hline.hide()

    def clear_annotations(self):
        """移除场景中的所有标注图元，保留底层影像、蒙版和十字线图元。
        每次 update_display() 刷新影像后都需要调用，防止标注重影（旧标注叠加在新帧上）。
        """
        for item in self.scene.items():
            if item not in [self.image_item, self.mask_item, self.vline, self.hline]:
                self.scene.removeItem(item)

    def resizeEvent(self, event):
        """布局变化（分割条拖动、窗口缩放）触发自动重新适配影像到视图。
        这是 resizeEvent 的核心用途：当父容器尺寸改变后，视图需要重新计算
        fitInView 以填满新空间。仅在有实际影像时触发，避免对空视图操作。
        """
        super().resizeEvent(event)
        px = self.image_item.pixmap()
        if px and not px.isNull():
            self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        """滚轮事件分发：
        - Ctrl + 滚轮：以鼠标位置为锚点缩放（每档 ×1.15 或 ÷1.15）
        - 普通滚轮：发射 wheel_scrolled 信号，由父窗口决定切换哪个方向的切片
        """
        if event.modifiers() == Qt.ControlModifier:
            # AnchorUnderMouse 使缩放中心固定在光标下方，符合影像阅片习惯
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            z = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(z, z)
        else:
            self.wheel_scrolled.emit(event.angleDelta().y())

    def mousePressEvent(self, event):
        """鼠标按下：根据当前工具初始化对应的绘制状态，或开始平移/调窗操作。"""

        # 中键按下：将拖拽模式切换为 ScrollHandDrag（手型游标），实现图像平移
        # 技巧：Qt 的 ScrollHandDrag 需要左键触发，这里构造一个假的左键事件欺骗基类
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            fake = QMouseEvent(event.type(), event.position(), event.globalPosition(),
                               Qt.LeftButton, Qt.LeftButton, event.modifiers())
            super().mousePressEvent(fake)
            return

        # 右键按下：进入窗宽/窗位拖拽调节模式（放射科标准交互：水平拖 → 调 WW，垂直拖 → 调 WL）
        if event.button() == Qt.RightButton:
            self.is_windowing = True
            self.last_mouse_pos = event.position().toPoint()
            self.setCursor(Qt.SizeAllCursor)  # 四向箭头光标提示用户可以拖拽
            return

        # 将视图坐标（像素点）转换为场景坐标（影像像素坐标），供各工具使用
        sp = self.mapToScene(event.position().toPoint())

        if event.button() == Qt.LeftButton:
            self.is_drawing = True

            if self.current_tool == TOOL_POINTER:
                # 指针工具：点击发射 HU 测量信号，同时允许拖拽平移（ScrollHandDrag）
                self.clicked_pos.emit(event.position().toPoint())
                self.setDragMode(QGraphicsView.ScrollHandDrag)
                super().mousePressEvent(event)

            elif self.current_tool == TOOL_RULER:
                # 卡尺工具：记录起点，创建临时直线和距离文字标签，在 Move 中实时更新
                self.start_pos = sp
                pen = QPen(QColor("#FF3366"), 2)
                self.temp_item = QGraphicsLineItem(QLineF(sp, sp))
                self.temp_item.setPen(pen)
                self.scene.addItem(self.temp_item)
                self.temp_text = QGraphicsTextItem("")
                self.temp_text.setDefaultTextColor(QColor("#FF3366"))
                self.temp_text.setFont(QFont("Arial", 11, QFont.Bold))
                self.scene.addItem(self.temp_text)

            elif self.current_tool == TOOL_DRAW:
                # 自由画笔：用 QPainterPath 记录连续轨迹，RoundCap/RoundJoin 让线条圆润
                self.current_path = QPainterPath(sp)
                self.polygon_points = [(sp.x(), sp.y())]
                self.temp_item = QGraphicsPathItem(self.current_path)
                pen = QPen(QColor("#00ADB5"), 2)
                pen.setCapStyle(Qt.RoundCap)
                pen.setJoinStyle(Qt.RoundJoin)
                self.temp_item.setPen(pen)
                self.scene.addItem(self.temp_item)

            elif self.current_tool == TOOL_CROP:
                # 套索工具：连续记录顶点，形成任意多边形，半透明黄色填充预览
                self.polygon_points = [sp]
                self.temp_item = QGraphicsPolygonItem(QPolygonF(self.polygon_points))
                self.temp_item.setPen(QPen(QColor("#F1C40F"), 2, Qt.DashLine))
                self.temp_item.setBrush(QBrush(QColor(241, 196, 15, 50)))  # alpha=50 半透明
                self.scene.addItem(self.temp_item)

            elif self.current_tool in [TOOL_RECT_CROP, TOOL_AI_TRACK]:
                # 矩形截取/AI追踪：橙色矩形 vs 紫色矩形，视觉区分不同功能
                self.start_pos = sp
                c = QColor("#9B59B6") if self.current_tool == TOOL_AI_TRACK else QColor("#E67E22")
                self.temp_rect_item = QGraphicsRectItem(QRectF(sp, sp))
                self.temp_rect_item.setPen(QPen(c, 2, Qt.DashLine))
                self.temp_rect_item.setBrush(QBrush(QColor(c.red(), c.green(), c.blue(), 40)))
                self.scene.addItem(self.temp_rect_item)

    def mouseMoveEvent(self, event):
        """鼠标移动：实时更新绘制预览 / 调窗增量 / MPR 十字线位置。"""
        sp = self.mapToScene(event.position().toPoint())

        # 非调窗状态下，发射鼠标悬浮信号供 MPR 联动十字线更新
        # 调窗期间不更新十字线，避免光标移动引起视图混乱
        if not self.is_windowing:
            real_coord = self.get_real_coordinates(event.position().toPoint())
            if real_coord:
                self.mouse_hovered.emit(QPoint(real_coord[0], real_coord[1]))

        if self.is_windowing:
            if self.last_mouse_pos is not None:
                curr_pos = event.position().toPoint()
                dx = curr_pos.x() - self.last_mouse_pos.x()
                dy = curr_pos.y() - self.last_mouse_pos.y()
                # ×2 放大灵敏度：原始像素增量较小，乘以系数后 WW/WL 变化更明显
                self.window_changed.emit(dx * 2, dy * 2)
            self.last_mouse_pos = event.position().toPoint()
            return

        if self.is_drawing:
            if self.current_tool == TOOL_RULER and self.temp_item:
                # 实时更新直线终点，并换算为毫米距离显示
                # pixel_spacing[0]=行间距(mm/像素)，[1]=列间距，分别对应 Y 轴和 X 轴
                self.temp_item.setLine(QLineF(self.start_pos, sp))
                d = math.sqrt(
                    ((sp.x() - self.start_pos.x()) * self.pixel_spacing[1]) ** 2 +
                    ((sp.y() - self.start_pos.y()) * self.pixel_spacing[0]) ** 2
                )
                self.temp_text.setPlainText(f"{d:.1f} mm")
                self.temp_text.setPos(sp.x() + 10, sp.y() + 10)  # 标签跟随终点偏移显示

            elif self.current_tool == TOOL_DRAW and self.temp_item:
                # 追加路径节点，实时渲染自由曲线
                self.current_path.lineTo(sp)
                self.temp_item.setPath(self.current_path)
                self.polygon_points.append((sp.x(), sp.y()))

            elif self.current_tool == TOOL_CROP and self.temp_item:
                # 套索：每个 Move 事件都追加顶点，形成连续多边形
                self.polygon_points.append(sp)
                self.temp_item.setPolygon(QPolygonF(self.polygon_points))

            elif self.current_tool in [TOOL_RECT_CROP, TOOL_AI_TRACK] and self.temp_rect_item:
                # normalized() 保证无论拖拽方向如何，矩形 left < right, top < bottom
                self.temp_rect_item.setRect(QRectF(self.start_pos, sp).normalized())

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """鼠标释放：结束绘制操作，提交标注数据或触发相应信号。"""
        # 中键释放：恢复无拖拽模式
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.NoDrag)
            super().mouseReleaseEvent(event)
            return
        # 右键释放：退出调窗模式，光标恢复箭头
        if event.button() == Qt.RightButton:
            self.is_windowing = False
            self.last_mouse_pos = None
            self.setCursor(Qt.ArrowCursor)
            return

        if event.button() == Qt.LeftButton and self.is_drawing:
            self.is_drawing = False

            if self.current_tool == TOOL_POINTER:
                self.setDragMode(QGraphicsView.NoDrag)  # 指针松开后恢复无拖拽

            elif self.current_tool == TOOL_RULER and self.temp_item:
                # 移除临时预览图元，将最终数据打包为字典发射信号
                # 由 MedicalViewer.handle_annotation_added 接收并持久化
                p2 = self.mapToScene(event.position().toPoint())
                d = {'id': str(uuid.uuid4()), 'type': 'ruler',
                     'p1': (self.start_pos.x(), self.start_pos.y()),
                     'p2': (p2.x(), p2.y())}
                self.scene.removeItem(self.temp_item)
                self.scene.removeItem(self.temp_text)
                self.annotation_added.emit(d)

            elif self.current_tool == TOOL_DRAW and self.temp_item:
                d = {'id': str(uuid.uuid4()), 'type': 'path', 'points': self.polygon_points}
                self.scene.removeItem(self.temp_item)
                self.annotation_added.emit(d)

            elif self.current_tool == TOOL_CROP and self.temp_item:
                pts = [(p.x(), p.y()) for p in self.polygon_points]
                self.scene.removeItem(self.temp_item)
                if len(pts) >= 3:  # 至少三点才能构成多边形，过滤误触
                    self.crop_requested.emit(pts)

            elif self.current_tool == TOOL_RECT_CROP and getattr(self, 'temp_rect_item', None):
                r = self.temp_rect_item.rect()
                self.scene.removeItem(self.temp_rect_item)
                if r.width() > 5:  # 过滤宽度过小的误点（< 5 像素视为无效框选）
                    self.crop_requested.emit([
                        (r.left(), r.top()), (r.right(), r.top()),
                        (r.right(), r.bottom()), (r.left(), r.bottom())
                    ])

            elif self.current_tool == TOOL_AI_TRACK and getattr(self, 'temp_rect_item', None):
                r = self.temp_rect_item.rect()
                self.scene.removeItem(self.temp_rect_item)
                if r.width() > 5:
                    self.track_requested.emit(r)  # 发射 QRectF，供 3D 连通域追踪使用

            # 清理临时图元引用，防止悬空指针
            self.temp_item = self.temp_rect_item = self.temp_text = None

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        """Delete/Backspace 键删除当前选中的标注图元（需图元设置了 toolTip 作为 ID）。"""
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            for item in self.scene.selectedItems():
                # toolTip 存储标注的 UUID，通过信号通知父窗口从数据层删除
                if item.toolTip():
                    self.annotation_deleted.emit(item.toolTip())
        super().keyPressEvent(event)

    def get_real_coordinates(self, pos):
        """将视图坐标转换为影像像素坐标，越界则返回 None。"""
        sp = self.mapToScene(pos)
        x, y = int(sp.x()), int(sp.y())
        if (self.image_item.pixmap() and
                0 <= x < self.image_item.pixmap().width() and
                0 <= y < self.image_item.pixmap().height()):
            return x, y
        return None


def from_image(qimg):
    """QImage → QPixmap 辅助函数，集中处理 fromImage 调用。"""
    return QPixmap.fromImage(qimg)
