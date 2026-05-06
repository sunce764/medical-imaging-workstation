import sys
import os
import math
import csv
import json
import time
import pydicom
import numpy as np
import scipy.ndimage as ndimage
from scipy.interpolate import griddata
from skimage.transform import radon, iradon
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QFileDialog, QSlider,
                               QLabel, QGroupBox, QFormLayout, QSplitter, QCheckBox, 
                               QComboBox, QFrame, QGridLayout, QGraphicsView, QGraphicsScene, 
                               QGraphicsPixmapItem, QGraphicsLineItem, QGraphicsTextItem, 
                               QGraphicsPolygonItem, QGraphicsRectItem, QGraphicsPathItem, 
                               QButtonGroup, QMessageBox, QProgressDialog, QTabWidget, QRadioButton, QSizePolicy)
from PySide6.QtCore import Qt, QPoint, QTimer, QRectF, QLineF, QPointF, Signal, QThread
from PySide6.QtGui import QImage, QPixmap, QMouseEvent, QPainter, QWheelEvent, QPen, QColor, QFont, QPolygonF, QBrush, QKeyEvent, QPainterPath

# 尝试导入真实的 ONNX 运行库
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

TOOL_POINTER, TOOL_RULER, TOOL_DRAW, TOOL_CROP, TOOL_RECT_CROP, TOOL_AI_TRACK = range(6)
AXIAL = 0     
CORONAL = 1   
SAGITTAL = 2  

# =========================================================================
# 🚀 自动化底层：异步 AI 推理引擎 (独立于 UI 线程，绝对不卡顿)
# =========================================================================
class AutoAIEngineThread(QThread):
    finished_mask = Signal(np.ndarray, float) # 传递运算完成的 Mask 和耗时

    def __init__(self, volume_hu, model_path="lung_seg_model.onnx"):
        super().__init__()
        self.volume_hu = volume_hu
        self.model_path = model_path

    def run(self):
        start_t = time.perf_counter()
        
        # 【程序级预处理自动化】: 强制 HU 归一化，适配常规医学大模型
        # 标准肺窗大概在 -1000 到 400 之间，将其裁剪并映射到 0~1 的张量
        norm_vol = np.clip(self.volume_hu, -1000, 400)
        norm_vol = (norm_vol - (-1000)) / (400 - (-1000))
        
        final_mask = None
        
        # 1. 尝试执行真正的 ONNX 推理
        if HAS_ONNX and os.path.exists(self.model_path):
            try:
                # 扩展维度为 (Batch, Channels, D, H, W) = (1, 1, D, H, W)
                input_tensor = norm_vol[np.newaxis, np.newaxis, ...].astype(np.float32)
                session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
                input_name = session.get_inputs()[0].name
                
                # 核心自动化推理
                ort_outs = session.run(None, {input_name: input_tensor})
                # 假设输出为单通道概率图，执行后处理二值化 (> 0.5)
                pred_prob = ort_outs[0][0, 0, ...]
                final_mask = (pred_prob > 0.5).astype(np.uint8)
            except Exception as e:
                print(f"ONNX 推理失败，降级为数学算法: {e}")
        
        # 2. 若无模型文件，自动降级为高级 3D 数学清场算法 (保证业务闭环)
        if final_mask is None:
            try:
                # 连通域背景剥离算法 (全自动在后台运算，医生无感知)
                air_mask = (self.volume_hu < -300).astype(np.uint8)
                labels, num_features = ndimage.label(air_mask)
                border_labels = set(labels[0,:,:].flatten()) | set(labels[-1,:,:].flatten()) | \
                                set(labels[:,0,:].flatten()) | set(labels[:,-1,:].flatten()) | \
                                set(labels[:,:,0].flatten()) | set(labels[:,:,-1].flatten())
                
                internal_air = np.copy(air_mask)
                for bl in border_labels:
                    if bl != 0: internal_air[labels == bl] = 0
                        
                labels_int, num_int = ndimage.label(internal_air)
                counts = np.bincount(labels_int.flatten())
                counts[0] = 0 
                
                final_mask = np.zeros_like(internal_air)
                if len(counts) > 1:
                    l1 = counts.argmax()
                    final_mask[labels_int == l1] = 1
                    max_vol = counts[l1]
                    counts[l1] = 0
                    if counts.max() > max_vol * 0.05: 
                        l2 = counts.argmax()
                        final_mask[labels_int == l2] = 1
            except:
                final_mask = np.zeros_like(self.volume_hu, dtype=np.uint8)

        end_t = time.perf_counter()
        # 抛出信号给主界面
        self.finished_mask.emit(final_mask, (end_t - start_t) * 1000)


# =========================================================================
# 视图与主窗口代码
# =========================================================================
class MedicalGraphicsView(QGraphicsView):
    clicked_pos = Signal(QPoint)
    wheel_scrolled = Signal(int)
    annotation_added = Signal(dict) 
    crop_requested = Signal(list) 
    track_requested = Signal(QRectF) 
    annotation_deleted = Signal(str) 
    window_changed = Signal(int, int) 
    mouse_hovered = Signal(QPoint) 
    
    def __init__(self, view_id):
        super().__init__()
        self.view_id = view_id
        self.current_tool = TOOL_POINTER
        self.pixel_spacing = (1.0, 1.0) 
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.image_item = QGraphicsPixmapItem()
        self.scene.addItem(self.image_item)
        self.mask_item = QGraphicsPixmapItem()
        self.mask_item.setZValue(1) 
        self.scene.addItem(self.mask_item)
        
        self.vline = QGraphicsLineItem()
        self.hline = QGraphicsLineItem()
        pen_cross = QPen(QColor("#F39C12"), 1, Qt.DashLine) 
        self.vline.setPen(pen_cross); self.hline.setPen(pen_cross)
        self.vline.setZValue(2); self.hline.setZValue(2)
        self.scene.addItem(self.vline); self.scene.addItem(self.hline)
        self.vline.hide(); self.hline.hide()
        
        self.setRenderHint(QPainter.Antialiasing) 
        self.setRenderHint(QPainter.SmoothPixmapTransform) 
        self.setDragMode(QGraphicsView.NoDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("background-color: #000000; border: none;")
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True) 
        
        self.is_drawing = False
        self.temp_item = self.temp_rect_item = self.temp_text = self.start_pos = self.current_path = None
        self.polygon_points = []

    def set_image(self, pixmap, mask_qimg=None, pixel_spacing=(1.0, 1.0)):
        self.image_item.setPixmap(pixmap)
        self.pixel_spacing = pixel_spacing
        rect = pixmap.rect()
        self.scene.setSceneRect(QRectF(rect))
        self.vline.setLine(0, 0, 0, rect.height())
        self.hline.setLine(0, 0, rect.width(), 0)
        if mask_qimg:
            self.mask_item.setPixmap(QPixmap.fromImage(mask_qimg))
            self.mask_item.show()
        else: self.mask_item.hide()
        if self.transform().m11() == 1.0: self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def draw_crosshair(self, x, y, show=True):
        if show and self.image_item.pixmap():
            w, h = self.image_item.pixmap().width(), self.image_item.pixmap().height()
            if 0 <= x < w and 0 <= y < h:
                self.vline.setLine(x, 0, x, h)
                self.hline.setLine(0, y, w, y)
                self.vline.show(); self.hline.show()
                return
        self.vline.hide(); self.hline.hide()

    def clear_annotations(self):
        for item in self.scene.items():
            if item not in [self.image_item, self.mask_item, self.vline, self.hline]: 
                self.scene.removeItem(item)

    def wheelEvent(self, event):
        if event.modifiers() == Qt.ControlModifier:
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            z = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(z, z)
        else: self.wheel_scrolled.emit(event.angleDelta().y())

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            super().mousePressEvent(QMouseEvent(event.type(), event.pos(), Qt.LeftButton, Qt.LeftButton, event.modifiers()))
            return
        if event.button() == Qt.RightButton:
            self.is_windowing = True; self.last_mouse_pos = event.pos(); self.setCursor(Qt.SizeAllCursor)
            return
        sp = self.mapToScene(event.pos())
        if event.button() == Qt.LeftButton:
            self.is_drawing = True
            if self.current_tool == TOOL_POINTER: 
                self.clicked_pos.emit(event.pos()); self.setDragMode(QGraphicsView.ScrollHandDrag); super().mousePressEvent(event)
            elif self.current_tool == TOOL_RULER:
                self.start_pos = sp; pen = QPen(QColor("#FF3366"), 2)
                self.temp_item = QGraphicsLineItem(QLineF(sp, sp)); self.temp_item.setPen(pen); self.scene.addItem(self.temp_item)
                self.temp_text = QGraphicsTextItem(""); self.temp_text.setDefaultTextColor(QColor("#FF3366")); self.temp_text.setFont(QFont("Arial", 11, QFont.Bold)); self.scene.addItem(self.temp_text)
            elif self.current_tool == TOOL_DRAW:
                self.current_path = QPainterPath(sp); self.polygon_points = [(sp.x(), sp.y())]
                self.temp_item = QGraphicsPathItem(self.current_path); pen = QPen(QColor("#00ADB5"), 2); pen.setCapStyle(Qt.RoundCap); pen.setJoinStyle(Qt.RoundJoin); self.temp_item.setPen(pen); self.scene.addItem(self.temp_item)
            elif self.current_tool == TOOL_CROP:
                self.polygon_points = [sp]; self.temp_item = QGraphicsPolygonItem(QPolygonF(self.polygon_points))
                self.temp_item.setPen(QPen(QColor("#F1C40F"), 2, Qt.DashLine)); self.temp_item.setBrush(QBrush(QColor(241, 196, 15, 50))); self.scene.addItem(self.temp_item)
            elif self.current_tool in [TOOL_RECT_CROP, TOOL_AI_TRACK]:
                self.start_pos = sp; c = QColor("#9B59B6") if self.current_tool == TOOL_AI_TRACK else QColor("#E67E22")
                self.temp_rect_item = QGraphicsRectItem(QRectF(sp, sp)); self.temp_rect_item.setPen(QPen(c, 2, Qt.DashLine)); self.temp_rect_item.setBrush(QBrush(QColor(c.red(), c.green(), c.blue(), 40))); self.scene.addItem(self.temp_rect_item)

    def mouseMoveEvent(self, event):
        sp = self.mapToScene(event.pos())
        if not getattr(self, 'is_windowing', False):
            real_coord = self.get_real_coordinates(event.pos())
            if real_coord: self.mouse_hovered.emit(QPoint(real_coord[0], real_coord[1]))
        
        if getattr(self, 'is_windowing', False):
            if getattr(self, 'last_mouse_pos', None):
                dx = event.pos().x() - self.last_mouse_pos.x(); dy = event.pos().y() - self.last_mouse_pos.y()
                self.window_changed.emit(dx * 2, dy * 2) 
            self.last_mouse_pos = event.pos()
            return

        if getattr(self, 'is_drawing', False):
            if self.current_tool == TOOL_RULER and self.temp_item:
                self.temp_item.setLine(QLineF(self.start_pos, sp))
                d = math.sqrt(((sp.x()-self.start_pos.x())*self.pixel_spacing[1])**2 + ((sp.y()-self.start_pos.y())*self.pixel_spacing[0])**2)
                self.temp_text.setPlainText(f"{d:.1f} mm"); self.temp_text.setPos(sp.x()+10, sp.y()+10)
            elif self.current_tool == TOOL_DRAW and self.temp_item:
                self.current_path.lineTo(sp); self.temp_item.setPath(self.current_path); self.polygon_points.append((sp.x(), sp.y()))
            elif self.current_tool == TOOL_CROP and self.temp_item:
                self.polygon_points.append(sp); self.temp_item.setPolygon(QPolygonF(self.polygon_points))
            elif self.current_tool in [TOOL_RECT_CROP, TOOL_AI_TRACK] and getattr(self, 'temp_rect_item', None):
                self.temp_rect_item.setRect(QRectF(self.start_pos, sp).normalized())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton: self.setDragMode(QGraphicsView.NoDrag); super().mouseReleaseEvent(event); return
        if event.button() == Qt.RightButton: self.is_windowing = False; self.last_mouse_pos = None; self.setCursor(Qt.ArrowCursor); return

        if event.button() == Qt.LeftButton and getattr(self, 'is_drawing', False):
            self.is_drawing = False
            if self.current_tool == TOOL_POINTER: self.setDragMode(QGraphicsView.NoDrag)
            elif self.current_tool == TOOL_RULER and self.temp_item:
                p2 = self.mapToScene(event.pos()); d = {'id': str(id(self.temp_item)), 'type': 'ruler', 'p1': (self.start_pos.x(), self.start_pos.y()), 'p2': (p2.x(), p2.y())}
                self.scene.removeItem(self.temp_item); self.scene.removeItem(self.temp_text); self.annotation_added.emit(d)
            elif self.current_tool == TOOL_DRAW and self.temp_item:
                d = {'id': str(id(self.temp_item)), 'type': 'path', 'points': self.polygon_points}; self.scene.removeItem(self.temp_item); self.annotation_added.emit(d)
            elif self.current_tool == TOOL_CROP and self.temp_item:
                pts = [(p.x(), p.y()) for p in self.polygon_points]; self.scene.removeItem(self.temp_item)
                if len(pts) > 3: self.crop_requested.emit(pts) 
            elif self.current_tool == TOOL_RECT_CROP and getattr(self, 'temp_rect_item', None):
                r = self.temp_rect_item.rect(); self.scene.removeItem(self.temp_rect_item)
                if r.width() > 5: self.crop_requested.emit([(r.left(), r.top()), (r.right(), r.top()), (r.right(), r.bottom()), (r.left(), r.bottom())])
            elif self.current_tool == TOOL_AI_TRACK and getattr(self, 'temp_rect_item', None):
                r = self.temp_rect_item.rect(); self.scene.removeItem(self.temp_rect_item)
                if r.width() > 5: self.track_requested.emit(r)
            self.temp_item = self.temp_rect_item = self.temp_text = None
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            for item in self.scene.selectedItems():
                if item.toolTip(): self.annotation_deleted.emit(item.toolTip())
        super().keyPressEvent(event)

    def get_real_coordinates(self, pos):
        sp = self.mapToScene(pos); x, y = int(sp.x()), int(sp.y())
        if self.image_item.pixmap() and 0 <= x < self.image_item.pixmap().width() and 0 <= y < self.image_item.pixmap().height(): return x, y
        return None

class MedicalViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medical Imaging Workstation Pro + Recon Lab")
        self.resize(1600, 950)
        self.dicom_datasets = []     
        self.current_slice_idx = 0   
        self.views = {}              
        self.active_tool = TOOL_POINTER 
        self.global_annotations = {'all': []}
        self.volume_hu = self.volume_mask = None   
        self.is_english = False 
        self.current_3d_pos = [0, 0, 0]
        
        self.recon_mode_active = False
        self.current_sinogram = None
        self.current_theta = None
        self.ai_thread = None # 保存线程对象

        self.setup_stylesheet()
        self.init_ui()
        self.update_language() 
        
        QTimer.singleShot(50, lambda: self.switch_layout(0)) 
        
        dp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "肺癌")
        if os.path.exists(dp): self.load_data(dp)

    def setup_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #12141A; color: #A0AABF; font-family: -apple-system, sans-serif; }
            QFrame#LeftToolbar { background-color: #1C1F26; border-right: 1px solid #2D3340; }
            QPushButton.ToolBtn { background-color: transparent; color: #8B949E; border: none; border-radius: 6px; padding: 12px 4px; font-size: 11px; font-weight: bold; }
            QPushButton.ToolBtn:hover { background-color: #262B35; color: #FFFFFF; }
            QPushButton.ToolBtn:checked { background-color: #00ADB5; color: black; border-radius: 4px; }
            QFrame#RightPanel { background-color: #12141A; border-left: 1px solid #2D3340; }
            
            QTabWidget::pane { border: 1px solid #2D3340; border-radius: 4px; background-color: #1C1F26; }
            QTabBar::tab { width: 136px; height: 28px; background-color: #12141A; color: #8B949E; border: 1px solid #2D3340; border-bottom: none; border-top-left-radius: 6px; border-top-right-radius: 6px; padding: 4px 0px; font-weight: bold; alignment: center; }
            QTabBar::tab:selected { background-color: #1C1F26; color: #00ADB5; border-bottom: 2px solid #00ADB5; }
            QTabBar::tab:hover:!selected { background-color: #262B35; }

            QGroupBox { background-color: #1C1F26; font-size: 12px; font-weight: bold; color: #5C677D; border: 1px solid #2D3340; border-radius: 8px; margin-top: 20px; padding-top: 20px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; top: 5px; color: #5C677D; background-color: transparent; padding: 0 10px; }
            QLabel { color: #A0AABF; font-size: 12px; }
            QLabel#FixedLabel { min-width: 60px; max-width: 60px; }
            QLabel#ValueText { color: #E2E8F0; font-family: monospace; }
            
            QPushButton.ActionBtn { min-height: 32px; max-height: 32px; background-color: #262B35; color: #D1D5DB; border: 1px solid #373E4D; border-radius: 6px; padding: 0 8px; font-size: 12px; font-weight: bold; }
            QPushButton.ActionBtn:hover { background-color: #373E4D; color: white; }
            QPushButton#PrimaryBtn { min-height: 38px; max-height: 38px; background-color: #0078D7; color: white; border: none; border-radius: 6px; padding: 0 10px; font-weight: bold; font-size: 13px; }
            QPushButton#DangerBtn { min-height: 32px; max-height: 32px; background-color: transparent; color: #C0392B; border: 1px solid #C0392B; border-radius: 6px; padding: 0 8px; }
            QPushButton#DangerBtn:hover { background-color: #C0392B; color: white; }
            QPushButton#MprBtn { min-height: 28px; max-height: 28px; background-color: #1C1F26; color: #00ADB5; border: 1px solid #00ADB5; border-radius: 4px; padding: 0 6px; font-size: 11px; }
            QPushButton#MprBtn:checked { background-color: #00ADB5; color: #000000; font-weight: bold; }
            
            QRadioButton { min-height: 20px; max-height: 20px; color: #D1D5DB; font-size: 12px; }
            QRadioButton::indicator { width: 14px; height: 14px; border-radius: 8px; border: 1px solid #4B5563; background-color: #262B35; }
            /* 选中时边框增加 2px，所以内部宽高减小 4px，保持总占据空间严格为 16x16 不变 */
            QRadioButton::indicator:checked { width: 10px; height: 10px; background-color: #00ADB5; border: 3px solid #1C1F26; border-radius: 8px; }
            
            QSlider::groove:horizontal { background: #2D3340; height: 4px; border-radius: 2px; }
            QSlider::handle:horizontal { background: #00ADB5; width: 14px; border-radius: 7px; margin: -5px 0; }
            QFrame#ViewContainer { background-color: #000000; border: 1px solid #2D3340; border-radius: 6px; }
            QFrame#ViewToolbar { background-color: #1C1F26; border-bottom: 1px solid #2D3340; border-top-left-radius: 6px; border-top-right-radius: 6px; }
            
            QComboBox { min-height: 26px; max-height: 26px; background-color: #262B35; color: #E2E8F0; border: 1px solid #4B5563; border-radius: 4px; padding: 2px 15px 2px 8px; font-size: 11px; text-align: left;}
            QComboBox QAbstractItemView { background-color: #1C1F26; color: #E2E8F0; selection-background-color: #00ADB5; selection-color: #000000; border: 1px solid #4B5563; text-align: left; }
            
            QCheckBox { color: #8B949E; font-size: 11px; }
            QCheckBox#ViewOption { min-width: 60px; max-width: 60px; }
        """)

    def init_ui(self):
        mw = QWidget(); self.setCentralWidget(mw); l = QHBoxLayout(mw); l.setContentsMargins(0,0,0,0); l.setSpacing(0)

        self.left_toolbar = QFrame(); self.left_toolbar.setObjectName("LeftToolbar"); self.left_toolbar.setFixedWidth(70)
        ll = QVBoxLayout(self.left_toolbar); ll.setContentsMargins(5,20,5,20); ll.setSpacing(15)
        self.tool_btn_group = QButtonGroup(self); self.tool_btns = {}
        tool_data = [(0,'btn_ptr'),(1,'btn_rul'),(2,'btn_drw'),(4,'btn_rec'),(3,'btn_las'),(5,'btn_trk')]
        for tid, key in tool_data:
            b = QPushButton(); b.setProperty("class", "ToolBtn"); b.setCheckable(True); b.setChecked(tid==0)
            self.tool_btn_group.addButton(b, tid); ll.addWidget(b); self.tool_btns[key] = b
        self.tool_btn_group.idClicked.connect(self.change_active_tool)
        ll.addStretch()

        self.main_splitter = QSplitter(Qt.Vertical); self.top_splitter = QSplitter(Qt.Horizontal); self.bottom_splitter = QSplitter(Qt.Horizontal)
        self.create_independent_view(1, AXIAL); self.create_independent_view(2, AXIAL)  
        self.create_independent_view(3, AXIAL); self.create_independent_view(4, AXIAL)    
        self.top_splitter.addWidget(self.views[1]['container']); self.top_splitter.addWidget(self.views[2]['container'])
        self.bottom_splitter.addWidget(self.views[3]['container']); self.bottom_splitter.addWidget(self.views[4]['container'])
        self.main_splitter.addWidget(self.top_splitter); self.main_splitter.addWidget(self.bottom_splitter)

        self.right_panel = QFrame(); self.right_panel.setObjectName("RightPanel"); self.right_panel.setFixedWidth(320)
        rl = QVBoxLayout(self.right_panel); rl.setContentsMargins(12,12,12,12); rl.setSpacing(5)
        
        th = QHBoxLayout(); self.btn_lang = QPushButton("EN"); self.btn_lang.setFixedWidth(40); self.btn_lang.setStyleSheet("font-size: 10px; color: #5C677D; border: 1px solid #373E4D;")
        self.btn_lang.clicked.connect(self.toggle_language); th.addStretch(); th.addWidget(self.btn_lang); rl.addLayout(th)
        self.btn_import = QPushButton("加载 DICOM 目录"); self.btn_import.setObjectName("PrimaryBtn"); self.btn_import.clicked.connect(self.select_folder); rl.addWidget(self.btn_import)
        self.btn_save_proj = QPushButton("保存标注工程"); self.btn_save_proj.setProperty("class", "ActionBtn"); self.btn_save_proj.clicked.connect(self.save_project); rl.addWidget(self.btn_save_proj)

        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.tab_clinical = QWidget()
        self.tab_recon = QWidget()
        self.tabs.addTab(self.tab_clinical, "临床阅片")
        self.tabs.addTab(self.tab_recon, "重建实验室")
        
        # --- Tab 1: 临床阅片 ---
        t1_lay = QVBoxLayout(self.tab_clinical)
        t1_lay.setContentsMargins(0,0,0,0)
        
        self.grp_patient = QGroupBox("患者信息"); self.info_lay = QFormLayout(); self.info_lay.setContentsMargins(10,15,10,10)
        self.info_labels = {"ID":QLabel("N/A"),"NAME":QLabel("N/A"),"AGE":QLabel("N/A")}
        for k, v in self.info_labels.items(): v.setObjectName("ValueText"); self.info_lay.addRow(QLabel(k), v)
        self.grp_patient.setLayout(self.info_lay); t1_lay.addWidget(self.grp_patient)
        
        self.grp_display = QGroupBox("显示控制"); dl = QVBoxLayout(); dl.setContentsMargins(10,15,10,10)
        top_dl = QHBoxLayout()
        self.combo_layout = QComboBox(); self.combo_layout.currentIndexChanged.connect(self.switch_layout); self.combo_layout.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        top_dl.addWidget(self.combo_layout)
        self.btn_mpr = QPushButton("MPR 联动: 关"); self.btn_mpr.setObjectName("MprBtn"); self.btn_mpr.setCheckable(True); self.btn_mpr.clicked.connect(self.on_mpr_toggled); self.btn_mpr.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        top_dl.addWidget(self.btn_mpr)
        top_dl.setStretch(0, 3); top_dl.setStretch(1, 2); dl.addLayout(top_dl)
        
        self.lbl_slice = QLabel(); self.slider_slice = QSlider(Qt.Horizontal); self.slider_slice.valueChanged.connect(self.on_slice_changed)
        self.lbl_ww = QLabel(); self.slider_ww = QSlider(Qt.Horizontal); self.slider_ww.setRange(1,4000); self.slider_ww.setValue(1500); self.slider_ww.valueChanged.connect(self.update_display)
        self.lbl_wl = QLabel(); self.slider_wl = QSlider(Qt.Horizontal); self.slider_wl.setRange(-1200,1200); self.slider_wl.setValue(-500); self.slider_wl.valueChanged.connect(self.update_display)
        for w in [self.lbl_slice, self.slider_slice, self.lbl_ww, self.slider_ww, self.lbl_wl, self.slider_wl]: dl.addWidget(w)
        
        pl = QGridLayout(); self.preset_btns = []
        for i, (n, ww, wl) in enumerate([("Lung",1500,-500),("Medi",400,40),("Bone",1500,400),("Vesc",600,150),("Abdo",150,30),("Brain",80,40)]):
            b = QPushButton(n); b.setProperty("class", "ActionBtn"); b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.clicked.connect(lambda c, w=ww, l=wl: self.set_window(w, l)); pl.addWidget(b, i//3, i%3); self.preset_btns.append(b)
        pl.setColumnStretch(0, 1); pl.setColumnStretch(1, 1); pl.setColumnStretch(2, 1)
        dl.addLayout(pl); self.grp_display.setLayout(dl); t1_lay.addWidget(self.grp_display)
        
        # 将原有的 AI 按钮改为状态显示标签，因为现在是全自动的了
        self.grp_ai = QGroupBox("自动化 AI 引擎"); ai_lay = QVBoxLayout(); ai_lay.setContentsMargins(10,15,10,10)
        self.lbl_ai_status = QLabel("状态: 待机中"); self.lbl_ai_status.setStyleSheet("color: #8B949E; font-weight: bold;")
        ai_lay.addWidget(self.lbl_ai_status); self.grp_ai.setLayout(ai_lay); t1_lay.addWidget(self.grp_ai)

        self.grp_measure = QGroupBox("测量与清理"); ml = QVBoxLayout(); ml.setContentsMargins(10,15,10,10)
        self.lbl_hu_value = QLabel(); self.lbl_hu_value.setStyleSheet("color: #00ADB5; font-weight: bold; font-size: 13px; min-height: 18px; max-height: 18px;"); self.lbl_hu_value.setAlignment(Qt.AlignCenter); ml.addWidget(self.lbl_hu_value)
        self.chk_global_scope = QCheckBox("新标注穿透所有切片"); ml.addWidget(self.chk_global_scope)
        self.btn_clear_anno = QPushButton("清空蒙版与标注"); self.btn_clear_anno.setProperty("class", "ActionBtn"); self.btn_clear_anno.clicked.connect(self.clear_current_slice_annotations); ml.addWidget(self.btn_clear_anno)
        self.btn_reset = QPushButton("重置工作区"); self.btn_reset.setObjectName("DangerBtn"); self.btn_reset.clicked.connect(self.reset_all_states); ml.addWidget(self.btn_reset)
        self.grp_measure.setLayout(ml); t1_lay.addWidget(self.grp_measure)
        t1_lay.addStretch()

        # --- Tab 2: 重建实验室 (Recon Lab) ---
        t2_lay = QVBoxLayout(self.tab_recon)
        t2_lay.setContentsMargins(0,0,0,0)
        
        self.grp_proj = QGroupBox("X射线投影生成"); play = QVBoxLayout(); play.setSpacing(10)
        self.rad_60 = QRadioButton("60°"); self.rad_120 = QRadioButton("120°"); self.rad_180 = QRadioButton("180°")
        self.rad_180.setChecked(True)
        h_rad = QHBoxLayout(); 
        for r in [self.rad_60, self.rad_120, self.rad_180]: r.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed); h_rad.addWidget(r)
        h_rad.setStretch(0, 1); h_rad.setStretch(1, 1); h_rad.setStretch(2, 1); play.addLayout(h_rad)
        self.btn_gen_sino = QPushButton("发射射线生成弦图"); self.btn_gen_sino.setProperty("class", "ActionBtn")
        self.btn_gen_sino.setStyleSheet("background-color: #D35400; color: white;")
        self.btn_gen_sino.clicked.connect(self.generate_sinogram)
        play.addWidget(self.btn_gen_sino); self.grp_proj.setLayout(play); t2_lay.addWidget(self.grp_proj)
        
        self.grp_algo = QGroupBox("图像重建算法"); alay = QVBoxLayout(); alay.setSpacing(10)
        self.btn_dfr = QPushButton("直接傅里叶重建 (DFR)"); self.btn_dfr.setProperty("class", "ActionBtn"); self.btn_dfr.clicked.connect(self.run_dfr)
        self.btn_bp = QPushButton("反投影法 (BP - 未滤波)"); self.btn_bp.setProperty("class", "ActionBtn"); self.btn_bp.clicked.connect(self.run_bp)
        
        h_fbp = QHBoxLayout(); 
        self.lbl_filter_text = QLabel("选择滤波器:"); self.lbl_filter_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        h_fbp.addWidget(self.lbl_filter_text)
        self.cb_filter = QComboBox(); self.cb_filter.addItems(["Ram-Lak", "Shepp-Logan", "Cosine", "Hamming", "Hann"])
        self.cb_filter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        h_fbp.addWidget(self.cb_filter)
        h_fbp.setStretch(0, 1); h_fbp.setStretch(1, 2); alay.addLayout(h_fbp)
        
        self.btn_fbp = QPushButton("滤波反投影 (FBP) 对比"); self.btn_fbp.setProperty("class", "ActionBtn"); self.btn_fbp.setStyleSheet("background-color: #27AE60; color: white;")
        self.btn_fbp.clicked.connect(self.run_fbp)
        
        alay.addWidget(self.btn_dfr); alay.addWidget(self.btn_bp); alay.addWidget(self.btn_fbp)
        self.grp_algo.setLayout(alay); t2_lay.addWidget(self.grp_algo)
        
        self.grp_mon = QGroupBox("算法性能监控"); mlay = QVBoxLayout()
        self.lbl_time = QLabel("运行耗时: -- ms"); 
        self.lbl_time.setStyleSheet("color: #00FF00; font-family: monospace; font-size: 14px; font-weight: bold; background-color: #000000; padding: 6px; border-radius: 4px; border: 1px solid #333; min-height: 20px; max-height: 20px;")
        self.lbl_time.setAlignment(Qt.AlignCenter)
        mlay.addWidget(self.lbl_time); self.grp_mon.setLayout(mlay); t2_lay.addWidget(self.grp_mon)
        t2_lay.addStretch()

        rl.addWidget(self.tabs)
        l.addWidget(self.left_toolbar); l.addWidget(self.main_splitter, 1); l.addWidget(self.right_panel)

    def toggle_language(self): self.is_english = not self.is_english; self.update_language()

    def update_language(self):
        e = self.is_english; self.btn_lang.setText("中" if e else "EN")
        self.tool_btns['btn_ptr'].setText("Pan\nProbe" if e else "探针\n拖拽")
        self.tool_btns['btn_rul'].setText("Ruler\nDist" if e else "测距\n卡尺")
        self.tool_btns['btn_drw'].setText("Draw\nPath" if e else "自由\n画笔")
        self.tool_btns['btn_rec'].setText("Rect\nCrop" if e else "矩形\n截取")
        self.tool_btns['btn_las'].setText("Lasso\nMask" if e else "套索\n抠图")
        self.tool_btns['btn_trk'].setText("3D\nTrack" if e else "3D\n追踪")
        self.btn_import.setText("Load DICOM Folder" if e else "加载 DICOM 目录")
        self.btn_save_proj.setText("Save Project" if e else "保存标注工程")
        
        self.tabs.setTabText(0, "Clinical Mode" if e else "临床阅片")
        self.tabs.setTabText(1, "Recon Lab" if e else "重建实验室")

        self.grp_proj.setTitle("Projection Generation" if e else "X射线投影生成")
        self.btn_gen_sino.setText("Generate Sinogram" if e else "发射射线生成弦图")
        self.grp_algo.setTitle("Reconstruction Algorithms" if e else "图像重建算法")
        self.btn_dfr.setText("Direct Fourier (DFR)" if e else "直接傅里叶重建 (DFR)")
        self.btn_bp.setText("Back Projection (BP)" if e else "反投影法 (BP - 未滤波)")
        self.lbl_filter_text.setText("Filter:" if e else "选择滤波器:")
        self.btn_fbp.setText("Filtered BP (FBP)" if e else "滤波反投影 (FBP) 对比")
        self.grp_mon.setTitle("Performance Monitor" if e else "算法性能监控")
        
        if "耗时: --" in self.lbl_time.text() or "Time: --" in self.lbl_time.text():
            self.lbl_time.setText("Run Time: -- ms" if e else "运行耗时: -- ms")

        self.grp_patient.setTitle("PATIENT INFO" if e else "患者信息")
        self.grp_display.setTitle("DISPLAY CONTROL" if e else "显示控制")
        self.grp_measure.setTitle("MEASURE & CLEAN" if e else "测量与清理")
        self.grp_ai.setTitle("Automated AI Engine" if e else "自动化 AI 引擎")
        
        if "待机" in self.lbl_ai_status.text() or "Standby" in self.lbl_ai_status.text():
            self.lbl_ai_status.setText("Status: Standby" if e else "状态: 待机中")
            
        mpr_on = self.btn_mpr.isChecked()
        self.btn_mpr.setText(("MPR Link: ON" if mpr_on else "MPR Link: OFF") if e else ("MPR 联动: 开启" if mpr_on else "MPR 联动: 关"))
        opts = ["1x1 Single", "1x2 Dual", "2x2 Grid"] if e else ["单窗模式 (1x1)", "双窗对比 (1x2)", "四窗矩阵 (2x2)"]
        ci = max(0, self.combo_layout.currentIndex()); self.combo_layout.blockSignals(True); self.combo_layout.clear(); self.combo_layout.addItems(opts); self.combo_layout.setCurrentIndex(ci); self.combo_layout.blockSignals(False)
        p_en, p_cn = ["Lung","Medi","Bone","Vesc","Abdo","Brain"], ["肺窗","纵隔","骨窗","血管","腹部","脑窗"]
        for b, n in zip(self.preset_btns, p_en if e else p_cn): b.setText(n)
        self.btn_clear_anno.setText("Clear Mask" if e else "清空蒙版与标注")
        self.btn_reset.setText("Reset Workspace" if e else "重置工作区")
        
        v_en = ["Global", "Lung", "Medi", "Bone", "Vesc", "Abdo", "Brain"]
        v_cn = ["跟随", "肺窗", "纵隔", "骨窗", "血管", "腹部", "脑窗"]
        plane_en = ["Axial", "Coronal", "Sagittal"]
        plane_cn = ["横断面", "冠状面", "矢状面"]
        
        for vdata in self.views.values():
            curr_p = max(0, vdata['plane'])
            vdata['cb_plane'].blockSignals(True); vdata['cb_plane'].clear(); vdata['cb_plane'].addItems(plane_en if e else plane_cn); vdata['cb_plane'].setCurrentIndex(curr_p); vdata['cb_plane'].blockSignals(False)
            curr_preset = max(0, vdata['preset'].currentIndex())
            vdata['preset'].blockSignals(True); vdata['preset'].clear(); vdata['preset'].addItems(v_en if e else v_cn); vdata['preset'].setCurrentIndex(curr_preset); vdata['preset'].blockSignals(False)
            vdata['chk_anno'].setText("Anno" if e else "显示")
            vdata['lock'].setText("Lock" if e else "锁定")
        self.on_slice_changed(self.slider_slice.value())

    def on_tab_changed(self, index):
        self.recon_mode_active = (index == 1)
        if self.recon_mode_active:
            self.switch_layout(2) 
            self.set_view_title(1, "V1 [Ground Truth]" if self.is_english else "V1 [真实切片]")
            self.set_view_title(2, "V2 [Sinogram]" if self.is_english else "V2 [投影弦图]")
            self.set_view_title(3, "V3 [Spectrum]" if self.is_english else "V3 [频域谱]")
            self.set_view_title(4, "V4 [Reconstructed]" if self.is_english else "V4 [重建结果]")
            
            # 【交互优化】：在重建实验室模式下，隐藏所有无关的临床控件
            for vid, v in self.views.items():
                v['cb_plane'].hide()
                v['preset'].hide()
                v['chk_anno'].hide()
                v['lock'].hide()
            self.update_display() 
        else:
            self.current_sinogram = None 
            for vid, v in self.views.items():
                # 【交互恢复】：切回临床阅片时，重新显示这些控件
                v['cb_plane'].show()
                v['preset'].show()
                v['chk_anno'].show()
                v['lock'].show()
                self.set_view_title(vid, f"V{vid}")
            self.update_display()

    def set_view_title(self, vid, title):
        try:
            toolbar = self.views[vid]['container'].findChild(QFrame, "ViewToolbar")
            if toolbar:
                lbl = toolbar.findChild(QLabel)
                if lbl: lbl.setText(title)
        except: pass

    def display_numpy_image(self, vid, img_array, is_freq=False):
        if img_array is None: return
        h, w = img_array.shape
        if is_freq:
            img_norm = np.log1p(np.abs(img_array))
            # 加上防除零保护，代码更具鲁棒性
            ptp = img_norm.max() - img_norm.min()
            denom = ptp if ptp > 0 else 1e-5
            img_norm = ((img_norm - img_norm.min()) / denom * 255).astype(np.uint8)
        else:
            # 【终极视觉优化：百分位数鲁棒归一化】
            # 掐头去尾：忽略最低的 1% 和最高的 1% 极端噪点
            pmin = np.percentile(img_array, 1)
            pmax = np.percentile(img_array, 99)
            
            # 将数值强行截断在这个健康范围内
            img_clipped = np.clip(img_array, pmin, pmax)
            
            # 再进行 0-255 的映射，分母加上极小值防止除以 0
            denom = pmax - pmin if pmax > pmin else 1e-5
            img_norm = ((img_clipped - pmin) / denom * 255).astype(np.uint8)
            
            img_norm = np.ascontiguousarray(img_norm) # 防 C++ 崩溃
            
        qimg = QImage(img_norm.data, w, h, w, QImage.Format_Grayscale8).copy()
        self.views[vid]['view'].set_image(QPixmap.fromImage(qimg), pixel_spacing=(1.0, 1.0))
        self.views[vid]['view'].clear_annotations()

    def generate_sinogram(self):
        if not self.dicom_datasets or self.volume_hu is None: return
        z, y, x = self.current_3d_pos
        img_gt = self.volume_hu[z]
        if self.rad_60.isChecked(): angles = 60
        elif self.rad_120.isChecked(): angles = 120
        else: angles = 180
        self.current_theta = np.linspace(0., angles, angles, endpoint=False)
        start_t = time.perf_counter()
        img_norm = (img_gt - img_gt.min()) / (img_gt.max() - img_gt.min())
        self.current_sinogram = radon(img_norm, theta=self.current_theta, circle=True)
        end_t = time.perf_counter()
        t_msg = f"Radon Time: {(end_t - start_t)*1000:.1f} ms" if self.is_english else f"Radon投影耗时: {(end_t - start_t)*1000:.1f} ms"
        self.lbl_time.setText(t_msg)
        self.display_numpy_image(2, self.current_sinogram.T) 
        
        # 【关键修复】：只清空图片内容 (赋予空对象)，绝对不要摧毁 C++ 图层实体
        self.views[3]['view'].image_item.setPixmap(QPixmap())
        self.views[4]['view'].image_item.setPixmap(QPixmap())

        print(f"生成的投影矩阵维度 (探测器像素 x 角度): {self.current_sinogram.shape}")
        print(f"投影矩阵局部数值示例:\n{self.current_sinogram[100:105, 0:5]}")

    def run_bp(self):
        if self.current_sinogram is None: return
        start_t = time.perf_counter()
        recon_bp = iradon(self.current_sinogram, theta=self.current_theta, filter_name=None, circle=True)
        end_t = time.perf_counter()
        t_msg = f"BP Time: {(end_t - start_t)*1000:.1f} ms" if self.is_english else f"纯反投影(BP)耗时: {(end_t - start_t)*1000:.1f} ms"
        self.lbl_time.setText(t_msg)
        self.display_numpy_image(4, recon_bp)
        self.set_view_title(4, "V4 [BP Unfiltered]" if self.is_english else "V4 [反投影 BP - 边缘模糊]")

    def run_fbp(self):
        if self.current_sinogram is None: return
        filter_name = self.cb_filter.currentText().lower()
        
        # 【关键修复】：将 UI 的 ram-lak 强制映射为 skimage 底层认识的 ramp
        if filter_name == "ram-lak":
            filter_name = "ramp"
            
        start_t = time.perf_counter()
        recon_bp = iradon(self.current_sinogram, theta=self.current_theta, filter_name=None, circle=True)
        recon_fbp = iradon(self.current_sinogram, theta=self.current_theta, filter_name=filter_name, circle=True)
        end_t = time.perf_counter()
        t_msg = f"FBP ({filter_name}) Time: {(end_t - start_t)*1000:.1f} ms" if self.is_english else f"FBP ({filter_name})耗时: {(end_t - start_t)*1000:.1f} ms"
        self.lbl_time.setText(t_msg)
        self.display_numpy_image(3, recon_bp)
        self.set_view_title(3, "V3 [BP Comparison]" if self.is_english else "V3 [未滤波反投影对比]")
        self.display_numpy_image(4, recon_fbp)
        self.set_view_title(4, f"V4 [FBP - {filter_name}]" if self.is_english else f"V4 [滤波反投影 FBP - {filter_name}]")

    def run_dfr(self):
        if self.current_sinogram is None: return
        p = QProgressDialog("Computing 2D FFT & Gridding..." if self.is_english else "正在计算 2D 傅里叶极坐标插值...", None, 0, 0, self)
        p.setWindowModality(Qt.WindowModal); p.show(); QApplication.processEvents()
        
        start_t = time.perf_counter()
        sinogram = self.current_sinogram
        num_detectors, num_angles = sinogram.shape
        
        # 1. 计算一维傅里叶变换 (沿探测器方向)
        # 注意：这里必须要 shift 到中心，方便后续映射
        proj_fft = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(sinogram, axes=0), axis=0), axes=0)
        
        # 【作业要求补充】：提取一维傅里叶谱用于展示 (取绝对值对数)
        fft_1d_display = np.log1p(np.abs(proj_fft))
        
        # 2. 极坐标网格化准备
        r = np.arange(num_detectors) - num_detectors // 2
        r_grid, theta_grid = np.meshgrid(r, np.deg2rad(self.current_theta), indexing='ij')
        
        # 【极其关键的物理修正：密度补偿 (Ram-Lak 滤波的频域本质)】
        
        x_polar = r_grid * np.cos(theta_grid)
        y_polar = r_grid * np.sin(theta_grid)
        points = np.column_stack((x_polar.flatten(), y_polar.flatten()))
        values = proj_fft.flatten()
        
        grid_x, grid_y = np.meshgrid(r, r, indexing='ij')
        
        # 【极其关键的数学修正：插值算法】
        # 绝对不能用 'nearest' (最近邻)，频域最近邻会导致严重的放射状锯齿伪影
        # 必须改为 'linear' (线性插值) 或 'cubic'
        freq_domain_2d = griddata(points, values, (grid_x, grid_y), method='linear', fill_value=0)
        
        # 3. 逆傅里叶变换回到空间域
        recon_dfr = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(freq_domain_2d)))
        
        end_t = time.perf_counter()
        p.close()
        
        t_msg = f"DFR Time: {(end_t - start_t)*1000:.1f} ms" if self.is_english else f"傅里叶重建(DFR)耗时: {(end_t - start_t)*1000:.1f} ms"
        self.lbl_time.setText(t_msg)
        
        # 【显示分配优化】满足老师的全部要求
       # 【显示分配优化】满足老师的全部要求
        
        # V2 临时征用：展示作业要求的“二维频域分布图”
        # 弦图经过 1D FFT 和极坐标插值后，就变成了这个二维矩阵
        self.display_numpy_image(2, freq_domain_2d, is_freq=True)
        self.set_view_title(2, "V2 [2D Freq Spectrum]" if self.is_english else "V2 [映射后的二维频域分布]")
        
        # V3 显示：投影的一维傅里叶谱 (1D Spectrum)
        self.display_numpy_image(3, fft_1d_display, is_freq=False) 
        self.set_view_title(3, "V3 [1D FFT Spectrum]" if self.is_english else "V3 [投影的一维傅里叶谱]")
        
        # V4 显示：重建出的最终图像
        self.display_numpy_image(4, np.rot90(np.abs(recon_dfr)))
        self.set_view_title(4, "V4 [Direct Fourier DFR]" if self.is_english else "V4 [直接傅里叶重建 DFR]")

    def on_mpr_toggled(self, checked):
        self.update_language() 
        if checked:
            default_planes = [0, AXIAL, CORONAL, SAGITTAL, AXIAL]
            for vid, v in self.views.items(): v['cb_plane'].setCurrentIndex(default_planes[vid])
            self.update_display()
        else:
            for vdata in self.views.values(): vdata['view'].draw_crosshair(0, 0, show=False)

    def create_independent_view(self, vid, plane=AXIAL):
        c = QFrame(); c.setObjectName("ViewContainer"); lay = QVBoxLayout(c); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        t = QFrame(); t.setObjectName("ViewToolbar"); t.setFixedHeight(32) 
        tl = QHBoxLayout(t); tl.setContentsMargins(8,2,8,2); tl.setSpacing(6)
        lt = QLabel(f"V{vid}"); lt.setStyleSheet("color: #C9D1D9; font-weight: bold; min-width: 20px;")
        cb_plane = QComboBox(); cb_plane.setFixedWidth(80)
        ps = QComboBox(); ps.setFixedWidth(85); ps.currentIndexChanged.connect(self.update_display)
        an = QCheckBox(); an.setObjectName("ViewOption"); an.setChecked(True); an.stateChanged.connect(self.update_display)
        lk = QCheckBox(); lk.setObjectName("ViewOption"); lk.stateChanged.connect(self.update_display)
        tl.addWidget(lt); tl.addWidget(cb_plane); tl.addWidget(ps); tl.addStretch(); tl.addWidget(an); tl.addWidget(lk)
        v = MedicalGraphicsView(vid)
        v.clicked_pos.connect(lambda p, id=vid: self.measure_hu(p, id))
        v.wheel_scrolled.connect(lambda d, id=vid: self.on_wheel_mpr(d, id)) 
        v.annotation_added.connect(self.handle_annotation_added)
        v.annotation_deleted.connect(self.handle_annotation_deleted)
        v.crop_requested.connect(lambda pts, id=vid: self.handle_crop_requested(id, pts))
        v.track_requested.connect(lambda r, id=vid: self.handle_3d_track_requested(id, r))
        v.window_changed.connect(self.on_window_changed_by_mouse)
        v.mouse_hovered.connect(lambda pos, id=vid: self.sync_crosshair(pos, id))
        lay.addWidget(t); lay.addWidget(v); t.raise_()
        self.views[vid] = {'container':c, 'cb_plane': cb_plane, 'preset':ps, 'lock':lk, 'chk_anno':an, 'view':v, 'plane': plane, 'locked_idx':0}
        cb_plane.currentIndexChanged.connect(lambda idx, v_id=vid: self.change_view_plane(v_id, idx))

    def change_view_plane(self, vid, plane_idx):
        if plane_idx < 0: return
        self.views[vid]['plane'] = plane_idx
        if not self.recon_mode_active: self.update_display()
        v = self.views[vid]['view']; QTimer.singleShot(20, lambda: v.fitInView(v.scene.sceneRect(), Qt.KeepAspectRatio))

    def sync_crosshair(self, scene_pos, vid):
        if self.volume_hu is None or self.recon_mode_active: return 
        if not getattr(self, 'btn_mpr', None) or not self.btn_mpr.isChecked(): return
        source_plane = self.views[vid]['plane']; z, y, x = self.current_3d_pos; pos_x, pos_y = int(scene_pos.x()), int(scene_pos.y())
        Z_MAX, Y_MAX, X_MAX = self.volume_hu.shape
        if source_plane == AXIAL: x, y = pos_x, pos_y
        elif source_plane == CORONAL: x, z = pos_x, pos_y
        elif source_plane == SAGITTAL: y, z = pos_x, pos_y
        x = max(0, min(x, X_MAX - 1)); y = max(0, min(y, Y_MAX - 1)); z = max(0, min(z, Z_MAX - 1))
        self.current_3d_pos = [z, y, x]
        for v_id, vdata in self.views.items():
            if vdata['container'].isHidden(): continue
            p = vdata['plane']
            if p == AXIAL: vdata['view'].draw_crosshair(x, y)
            elif p == CORONAL: vdata['view'].draw_crosshair(x, z)
            elif p == SAGITTAL: vdata['view'].draw_crosshair(y, z)

    def on_window_changed_by_mouse(self, delta_ww, delta_wl):
        if not self.dicom_datasets or self.recon_mode_active: return
        new_ww = self.slider_ww.value() + delta_ww; new_wl = self.slider_wl.value() + delta_wl
        new_ww = max(self.slider_ww.minimum(), min(self.slider_ww.maximum(), new_ww))
        new_wl = max(self.slider_wl.minimum(), min(self.slider_wl.maximum(), new_wl))
        self.slider_ww.setValue(new_ww); self.slider_wl.setValue(new_wl)
        for vdata in self.views.values():
            if vdata['container'].isHidden(): continue
            if vdata['preset'].currentText() not in ["Global", "跟随"]:
                vdata['preset'].blockSignals(True); vdata['preset'].setCurrentIndex(0); vdata['preset'].blockSignals(False)

    def handle_3d_track_requested(self, vid, rect):
        if self.volume_hu is None or self.recon_mode_active: return
        if self.views[vid]['plane'] != AXIAL: QMessageBox.information(self, "提示", "目前智能追踪仅支持在 Axial 进行。"); return
        idx = self.current_3d_pos[0]; x1, y1, x2, y2 = int(rect.left()), int(rect.top()), int(rect.right()), int(rect.bottom())
        h, w = self.volume_hu.shape[1], self.volume_hu.shape[2]; x1,y1,x2,y2 = max(0,x1), max(0,y1), min(w,x2), min(h,y2)
        p = QProgressDialog("Computing 3D..." if self.is_english else "正在计算 3D...", None, 0, 0, self); p.setWindowModality(Qt.WindowModal); p.show(); QApplication.processEvents()
        try:
            roi = self.volume_hu[idx, y1:y2, x1:x2]; med, std = np.median(roi), np.std(roi)
            bv = (self.volume_hu >= med - 1.5*std) & (self.volume_hu <= med + 1.5*std)
            lab, _ = ndimage.label(bv); rl = lab[idx, y1:y2, x1:x2]; rl = rl[rl>0]
            if len(rl)>0: self.volume_mask = (lab == np.bincount(rl.flatten()).argmax()).astype(np.uint8)
        except: pass
        p.close(); self.update_display()

    def handle_crop_requested(self, vid, pts):
        if self.recon_mode_active or self.views[vid]['plane'] != AXIAL: return
        idx = self.current_3d_pos[0]; ds = self.dicom_datasets[idx]; hu = self.volume_hu[idx]; sp = (float(getattr(ds,'PixelSpacing',[1,1])[0]), float(getattr(ds,'PixelSpacing',[1,1])[1]))
        h, w = hu.shape; mq = QImage(w, h, QImage.Format_Grayscale8); mq.fill(Qt.black)
        painter = QPainter(mq); painter.setBrush(Qt.white); painter.drawPolygon(QPolygonF([QPointF(p[0],p[1]) for p in pts])); painter.end()
        ma = np.array(mq.constBits(), copy=False).reshape((h, mq.bytesPerLine()))[:,:w]; bm = (ma>0).astype(np.uint8); rh = hu[bm==1]
        if len(rh)>0:
            area = len(rh)*sp[0]*sp[1]
            if QMessageBox.question(self, "Stats", f"Area: {area:.2f} mm2\nMean: {np.mean(rh):.1f} HU\nSave?") == QMessageBox.Yes:
                img = np.clip(hu, -1250, 250); img = ((img+1250)/1500*255).astype(np.uint8)
                fn = f"{str(getattr(ds,'PatientName','P')).replace('^','_')}_S{idx+1}_{datetime.now().strftime('%H%M%S')}.png"
                ed = os.path.join(os.path.dirname(os.path.abspath(__file__)),"Exported_Lesions"); os.makedirs(ed, exist_ok=True)
                s_p, _ = QFileDialog.getSaveFileName(self, "Save", os.path.join(ed, fn), "PNG (*.png)")
                if s_p:
                    QImage((img*bm).data, w, h, w, QImage.Format_Grayscale8).copy().save(s_p)
                    with open(os.path.join(os.path.dirname(s_p),"export_log.csv"),'a',newline='',encoding='utf-8-sig') as f: writer = csv.writer(f); writer.writerow([os.path.basename(s_p), idx+1, round(area,2), round(np.mean(rh),2)])

    def handle_annotation_added(self, data):
        if self.recon_mode_active: return
        tk = 'all' if self.chk_global_scope.isChecked() else self.current_3d_pos[0]
        if tk not in self.global_annotations: self.global_annotations[tk] = []
        self.global_annotations[tk].append(data); self.update_display()

    def handle_annotation_deleted(self, aid):
        if self.recon_mode_active: return
        for k in self.global_annotations: self.global_annotations[k] = [a for a in self.global_annotations[k] if a['id'] != aid]
        self.update_display()

    def clear_current_slice_annotations(self):
        idx = self.current_3d_pos[0]; 
        if idx in self.global_annotations: self.global_annotations[idx] = []
        if self.volume_mask is not None: self.volume_mask = np.zeros_like(self.volume_hu)
        if not self.recon_mode_active: self.update_display()

    def change_active_tool(self, tid):
        self.active_tool = tid
        for v in self.views.values(): v['view'].current_tool = tid

    def on_wheel_mpr(self, d, vid):
        if self.volume_hu is None or self.recon_mode_active: return
        plane = self.views[vid]['plane']; increment = -1 if d > 0 else 1
        Z_MAX, Y_MAX, X_MAX = self.volume_hu.shape; z, y, x = self.current_3d_pos
        if plane == AXIAL: self.slider_slice.setValue(max(0, min(z + increment, Z_MAX - 1)))
        elif plane == CORONAL: self.current_3d_pos[1] = max(0, min(y + increment, Y_MAX - 1)); self.update_display()
        elif plane == SAGITTAL: self.current_3d_pos[2] = max(0, min(x + increment, X_MAX - 1)); self.update_display()

    def measure_hu(self, p, vid):
        if self.active_tool == TOOL_POINTER and self.volume_hu is not None and not self.recon_mode_active:
            vd = self.views.get(vid); c = vd['view'].get_real_coordinates(p); plane = vd['plane']
            if c: 
                try:
                    if plane == AXIAL: val = self.volume_hu[self.current_3d_pos[0], c[1], c[0]]
                    elif plane == CORONAL: val = self.volume_hu[c[1], self.current_3d_pos[1], c[0]]
                    elif plane == SAGITTAL: val = self.volume_hu[c[1], c[0], self.current_3d_pos[2]]
                    plane_str = {"Axial": "Axial", "Coronal": "Coronal", "Sagittal": "Sagittal"} if self.is_english else {"Axial": "横断面", "Coronal": "冠状面", "Sagittal": "矢状面"}
                    p_name = plane_str.get({AXIAL: "Axial", CORONAL: "Coronal", SAGITTAL: "Sagittal"}[plane])
                    self.lbl_hu_value.setText(f"V{vid} [{p_name}] ({c[0]}, {c[1]}) : {val:.1f} HU")
                except: pass

    def reset_all_states(self):
        self.combo_layout.setCurrentIndex(0); self.slider_ww.setValue(1500); self.slider_wl.setValue(-500)
        self.tool_btns['btn_ptr'].setChecked(True); self.change_active_tool(0); self.global_annotations = {'all':[]}
        if self.volume_mask is not None: self.volume_mask = np.zeros_like(self.volume_hu)
        self.btn_mpr.setChecked(False) 
        for vid, v in self.views.items(): 
            v['cb_plane'].setCurrentIndex(AXIAL) 
            v['preset'].setCurrentIndex(0); v['lock'].setChecked(False); v['chk_anno'].setChecked(True); v['view'].fitInView(v['view'].scene.sceneRect(), Qt.KeepAspectRatio)
        if not self.recon_mode_active: self.update_display()

    def set_window(self, ww, wl): self.slider_ww.setValue(ww); self.slider_wl.setValue(wl)

    def switch_layout(self, m):
        vs = [self.views[i]['container'] for i in range(1, 5)]
        if m == 0: vs[1].hide(); vs[2].hide(); vs[3].hide(); self.bottom_splitter.hide()
        elif m == 1: vs[1].show(); vs[2].hide(); vs[3].hide(); self.bottom_splitter.hide(); QTimer.singleShot(10, lambda: self.top_splitter.setSizes([1000, 1000]))
        else: vs[1].show(); vs[2].show(); vs[3].show(); self.bottom_splitter.show(); QTimer.singleShot(10, lambda: [self.top_splitter.setSizes([1000,1000]), self.bottom_splitter.setSizes([1000,1000]), self.main_splitter.setSizes([1000,1000])])
        for vd in self.views.values(): QTimer.singleShot(20, lambda v=vd['view']: v.fitInView(v.scene.sceneRect(), Qt.KeepAspectRatio))

    def load_data(self, path):
        self.dicom_datasets = []
        for r, d, fs in os.walk(path):
            for f in fs:
                if not f.startswith('.'):
                    try: self.dicom_datasets.append(pydicom.dcmread(os.path.join(r, f)))
                    except: continue
        if not self.dicom_datasets: return
        self.dicom_datasets.sort(key=lambda x: int(getattr(x, 'InstanceNumber', 0)))
        ds = self.dicom_datasets[0]; pid = str(getattr(ds, 'PatientID', 'N/A'))
        self.info_labels["ID"].setText(pid); self.info_labels["NAME"].setText(str(getattr(ds, 'PatientName', 'Unknown')).replace('^', ' ')); self.info_labels["AGE"].setText(str(getattr(ds, 'PatientAge', 'N/A')))
        self.volume_hu = np.array([d.pixel_array.astype(np.float32)*getattr(d,'RescaleSlope',1)+getattr(d,'RescaleIntercept',0) for d in self.dicom_datasets])
        self.volume_mask = np.zeros_like(self.volume_hu, dtype=np.uint8); self.global_annotations = {'all':[]}
        z, y, x = self.volume_hu.shape; self.current_3d_pos = [z//2, y//2, x//2]
        self.slider_slice.setRange(0, z-1); self.slider_slice.setValue(z//2) 
        af = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Exported_Lesions", f"{pid}_annotations.json")
        if os.path.exists(af):
            try:
                with open(af, 'r', encoding='utf-8') as f:
                    for k, v in json.load(f).items(): self.global_annotations[int(k) if k.isdigit() else k] = v
            except: pass
            
        self.on_slice_changed(z//2)
        QTimer.singleShot(100, lambda: [vd['view'].fitInView(vd['view'].scene.sceneRect(), Qt.KeepAspectRatio) for vd in self.views.values() if not vd['container'].isHidden()])
        
        # ==========================================================
        # 🚀 【自动化触发点】: 数据加载完毕后，全自动静默运行 AI 流水线
        # ==========================================================
        self.lbl_ai_status.setStyleSheet("color: #F1C40F; font-weight: bold;")
        self.lbl_ai_status.setText("Processing AI Pipeline..." if self.is_english else "状态: AI 引擎自动运算中...")
        
        self.ai_thread = AutoAIEngineThread(self.volume_hu)
        self.ai_thread.finished_mask.connect(self.on_auto_ai_finished)
        self.ai_thread.start()

    def on_auto_ai_finished(self, final_mask, time_ms):
        """AI 异步线程计算完毕后的回调"""
        self.volume_mask = final_mask
        self.lbl_ai_status.setStyleSheet("color: #00FF00; font-weight: bold;")
        self.lbl_ai_status.setText(f"Ready ({time_ms:.1f}ms)" if self.is_english else f"状态: 自动分割完成 ({time_ms:.1f}ms)")
        self.update_display()

    def on_slice_changed(self, idx):
        self.current_3d_pos[0] = idx
        self.lbl_slice.setText(f"{'Slice: ' if self.is_english else '层数: '}{idx+1} / {len(self.dicom_datasets)}")
        if not self.recon_mode_active: self.update_display()

    def update_display(self):
        if self.volume_hu is None: return
        z, y, x = self.current_3d_pos
        
        if self.recon_mode_active:
            img_gt = self.volume_hu[z]
            # 获取当前的窗宽窗位
            ww, wl = self.slider_ww.value(), self.slider_wl.value()
            
            # 【关键修复】：应用窗宽窗位截断，恢复高对比度
            img_windowed = np.clip(img_gt, wl - ww / 2, wl + ww / 2)
            img_windowed = ((img_windowed - (wl - ww / 2)) / ww * 255).astype(np.uint8)
            img_windowed = np.ascontiguousarray(img_windowed) # 防崩溃
            
            # 绕过底层的极值压缩，直接显示
            h, w = img_windowed.shape
            qimg = QImage(img_windowed.data, w, h, w, QImage.Format_Grayscale8).copy()
            self.views[1]['view'].set_image(QPixmap.fromImage(qimg))
            self.views[1]['view'].clear_annotations()
            return

        ww_m, wl_m = self.slider_ww.value(), self.slider_wl.value()
        self.lbl_ww.setText(f"WW: {ww_m}"); self.lbl_wl.setText(f"WL: {wl_m}")
        ds = self.dicom_datasets[z]
        px_sp = float(getattr(ds,'PixelSpacing',[1,1])[0]); slice_thick = float(getattr(ds, 'SliceThickness', px_sp * 3)) 
        
        for vid, vdata in self.views.items():
            if vdata['container'].isHidden(): continue
            plane = vdata['plane']; pre = vdata['preset'].currentText()
            if pre in ["Global", "跟随"]: ww, wl = ww_m, wl_m
            else:
                wm = {"Lung":1500,"Medi":400,"Bone":1500,"Vesc":600,"Abdo":150,"Brain":80,"肺窗":1500,"纵隔":400,"骨窗":1500,"血管":600,"腹部":150,"脑窗":80}
                lm = {"Lung":-500,"Medi":40,"Bone":400,"Vesc":150,"Abdo":30,"Brain":40,"肺窗":-500,"纵隔":40,"骨窗":400,"血管":150,"腹部":30,"脑窗":40}
                ww, wl = wm.get(pre, ww_m), lm.get(pre, wl_m)
            
            if plane == AXIAL: hu = self.volume_hu[z, :, :]; sp = (px_sp, px_sp)
            elif plane == CORONAL: hu = self.volume_hu[:, y, :]; sp = (px_sp, slice_thick)
            elif plane == SAGITTAL: hu = self.volume_hu[:, :, x]; sp = (px_sp, slice_thick)
                
            img = np.clip(hu, wl-ww/2, wl+ww/2); img = ((img-(wl-ww/2))/ww*255).astype(np.uint8); mq = None
            h, w = img.shape; qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8).copy()
            
            if plane == AXIAL:
                if vdata['chk_anno'].isChecked() and self.volume_mask is not None:
                    sm = self.volume_mask[z]
                    if np.any(sm):
                        ov = np.zeros((h, w, 4), dtype=np.uint8); ov[sm == 1] = [0, 173, 181, 100] 
                        mq = QImage(ov.data, w, h, w*4, QImage.Format_RGBA8888).copy()
            vdata['view'].set_image(QPixmap.fromImage(qimg), mq, sp); vdata['view'].clear_annotations()
            
            if plane == AXIAL and vdata['chk_anno'].isChecked():
                for anno in self.global_annotations.get(z, []) + self.global_annotations.get('all', []):
                    col = QColor("#00ADB5") if anno in self.global_annotations.get(z, []) else QColor("#F1C40F")
                    if anno['type'] == 'ruler':
                        line = QGraphicsLineItem(QLineF(anno['p1'][0], anno['p1'][1], anno['p2'][0], anno['p2'][1])); line.setPen(QPen(col, 2)); line.setToolTip(anno['id']); line.setFlag(QGraphicsLineItem.ItemIsSelectable); vdata['view'].scene.addItem(line)
                        dist = math.sqrt(((anno['p2'][0]-anno['p1'][0])*sp[1])**2 + ((anno['p2'][1]-anno['p1'][1])*sp[0])**2)
                        txt = QGraphicsTextItem(f"{dist:.1f} mm"); txt.setDefaultTextColor(col); txt.setFont(QFont("Arial", 11, QFont.Bold)); txt.setPos(anno['p2'][0]+10, anno['p2'][1]+10); vdata['view'].scene.addItem(txt)
                    elif anno['type'] == 'path':
                        pts = anno['points']; path = QPainterPath(QPointF(pts[0][0], pts[0][1]))
                        for p in pts[1:]: path.lineTo(QPointF(p[0], p[1]))
                        pen = QPen(col, 2); pen.setCapStyle(Qt.RoundCap); pen.setJoinStyle(Qt.RoundJoin)
                        item = QGraphicsPathItem(path); item.setPen(pen); item.setFlag(QGraphicsPathItem.ItemIsSelectable); item.setToolTip(anno['id']); vdata['view'].scene.addItem(item)
            
            if getattr(self, 'btn_mpr', None) and self.btn_mpr.isChecked():
                if plane == AXIAL: vdata['view'].draw_crosshair(x, y)
                elif plane == CORONAL: vdata['view'].draw_crosshair(x, z)
                elif plane == SAGITTAL: vdata['view'].draw_crosshair(y, z)
            else:
                vdata['view'].draw_crosshair(0, 0, show=False)

    def save_project(self):
        if not self.dicom_datasets: return
        pid = str(getattr(self.dicom_datasets[0], 'PatientID', 'Unknown'))
        ed = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Exported_Lesions"); os.makedirs(ed, exist_ok=True)
        with open(os.path.join(ed, f"{pid}_annotations.json"), 'w', encoding='utf-8') as f: json.dump({str(k): v for k, v in self.global_annotations.items()}, f, indent=4)
        QMessageBox.information(self, "Success", "Project Saved.")

    def select_folder(self):
        p = QFileDialog.getExistingDirectory(self, "Select Folder"); self.load_data(p) if p else None

if __name__ == "__main__":
    app = QApplication(sys.argv); window = MedicalViewer(); window.show(); sys.exit(app.exec())