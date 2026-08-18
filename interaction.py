# =============================================================================
# 交互 Mixin：Cine 电影播放 + MPR 联动 / 导航
# 负责：MPR 十字线联动（平面切换、悬停同步、滚轮换层/换面、HU 探针、光标 HUD）
#       与 Cine 电影播放（往返连播、调速、启停）。
#
# 设计：以 Mixin 形式并入 MedicalViewer。方法经 self 访问主窗口控件与状态
#       （self.views / self.slider_slice / self.cine_timer / self.current_3d_pos 等）
#       及 update_display 等留在 main 的方法。
#       注意：keyPressEvent 是 QMainWindow 的重写方法，因 MRO 中 QMainWindow 先于
#       Mixin，若放到 Mixin 会被遮蔽，故留在 MedicalViewer 本体。
# =============================================================================

from PySide6.QtCore import Qt, QTimer

import mpr_geometry
from constants import AXIAL, CORONAL, SAGITTAL, TOOL_POINTER


class InteractionMixin:
    """Cine 播放与 MPR 联动 / 导航交互方法集合，混入 MedicalViewer。"""

    def on_mpr_toggled(self, checked):
        self.update_language()
        if checked:
            default_planes = [0, AXIAL, CORONAL, SAGITTAL, AXIAL]
            for vid, v in self.views.items(): v['cb_plane'].setCurrentIndex(default_planes[vid])
            self.update_display()
        else:
            for vdata in self.views.values(): vdata['view'].draw_crosshair(0, 0, show=False)

    def change_view_plane(self, vid, plane_idx):
        """切换某个视图的成像平面（横断/冠状/矢状）。
        切换后延迟 20ms 做 fitInView，等待新图像渲染完成后再适配缩放，
        避免基于旧图像尺寸计算比例。
        """
        if plane_idx < 0:
            return  # 下拉框清空重填时会触发 index=-1，需要过滤
        self.views[vid]['plane'] = plane_idx
        if not self.recon_mode_active:
            self.update_display()
        v = self.views[vid]['view']
        QTimer.singleShot(20, lambda: v.fitInView(v.scene.sceneRect(), Qt.KeepAspectRatio))

    def sync_crosshair(self, scene_pos, vid):
        """MPR 联动：当用户在任意视图中移动鼠标时，同步更新所有视图的十字准线位置。

        坐标映射规则（三平面共用同一个 3D 光标 [z, y, x]）：
          - Axial 视图   → 鼠标 (px, py) 对应 3D 的 (x=px, y=py)，z 不变
          - Coronal 视图 → 鼠标 (px, py) 对应 3D 的 (x=px, z=py)，y 不变
          - Sagittal 视图 → 鼠标 (px, py) 对应 3D 的 (y=px, z=py)，x 不变
        """
        if self.volume_hu is None or self.recon_mode_active or self.compare_mode_active:
            return
        source_plane = self.views[vid]['plane']
        pos_x, pos_y = int(scene_pos.x()), int(scene_pos.y())
        # 悬停像素 → 完整 3D 体素（非 source 平面轴沿用当前光标），并裁剪到体积范围
        z, y, x = mpr_geometry.hover_to_voxel(source_plane, pos_x, pos_y,
                                              tuple(self.current_3d_pos), self.volume_hu.shape)
        self._update_hud(z, y, x)  # HUD 实时更新，不依赖 MPR 联动开关
        if not self.btn_mpr.isChecked():
            return
        self.current_3d_pos = [z, y, x]
        # 同步切片滑条（blockSignals 防止递归触发 on_slice_changed），保持三者一致
        if self.slider_slice.value() != z:
            self.slider_slice.blockSignals(True)
            self.slider_slice.setValue(z)
            self.slider_slice.blockSignals(False)
            self.lbl_slice.setText(f"{'Slice: ' if self.is_english else '层数: '}{z + 1} / {len(self.dicom_datasets)}")
        # 刷新所有平面图像到新光标位置（真正的 MPR 联动，而非仅移动十字线）
        self.update_display()
        # 十字线须在 update_display 之后重画（set_image 会重置线段坐标）
        for vdata in self.views.values():
            if vdata['container'].isHidden():
                continue
            cx, cy = mpr_geometry.voxel_to_crosshair(vdata['plane'], z, y, x)
            vdata['view'].draw_crosshair(cx, cy)

    def _update_hud(self, z, y, x):
        """更新光标 HUD：显示 (x,y,z) 坐标、该体素 HU 值、以及所在器官（若有分割）。"""
        hu = float(self.volume_hu[z, y, x])
        txt = f"({x}, {y}, {z})  {hu:.0f} HU"
        if self.volume_mask is not None:
            lid = int(self.volume_mask[z, y, x])
            if lid != 0:
                name = self.organ_names.get(lid, ("", ""))[1 if self.is_english else 0]
                if name:
                    txt += f"  ·  {name}"
        self.lbl_hud.setText(txt)


    def toggle_cine(self):
        """开始/停止 Cine 电影播放（自动连续翻片，到末层循环回第一层）。"""
        if self.volume_hu is None or self.recon_mode_active:
            return
        if self.cine_timer.isActive():
            self._stop_cine()
        else:
            self.cine_timer.start(int(self.cb_cine_speed.currentData()))
            self.btn_cine.setText("⏸ Pause" if self.is_english else "⏸ 暂停")

    def _on_cine_speed_changed(self):
        """播放中改速度即时生效。"""
        if self.cine_timer.isActive():
            self.cine_timer.start(int(self.cb_cine_speed.currentData()))

    def _cine_step(self):
        """Cine 定时器回调：往返(bounce)翻片——到顶/到底反向，避免跳变回环。"""
        if self.volume_hu is None or self.recon_mode_active:
            self._stop_cine(); return
        mx = self.slider_slice.maximum()
        nz = self.slider_slice.value() + self._cine_dir
        if nz > mx:
            nz, self._cine_dir = mx - 1, -1
        elif nz < 0:
            nz, self._cine_dir = 1, 1
        self.slider_slice.setValue(max(0, min(nz, mx)))

    def _stop_cine(self):
        """停止 Cine 播放并复位按钮文案。"""
        self.cine_timer.stop()
        self.btn_cine.setText("▶ Play" if self.is_english else "▶ 播放")


    def on_wheel_mpr(self, d, vid):
        if self.volume_hu is None or self.recon_mode_active: return
        increment = -1 if d > 0 else 1
        Z_MAX, Y_MAX, X_MAX = self.volume_hu.shape; z, y, x = self.current_3d_pos
        if self.compare_mode_active:   # 对比模式：滚轮始终换主序列切片，V2 按比例联动
            self.slider_slice.setValue(max(0, min(z + increment, Z_MAX - 1)))
            return
        plane = self.views[vid]['plane']
        if plane == AXIAL: self.slider_slice.setValue(max(0, min(z + increment, Z_MAX - 1)))
        elif plane == CORONAL: self.current_3d_pos[1] = max(0, min(y + increment, Y_MAX - 1)); self.update_display()
        elif plane == SAGITTAL: self.current_3d_pos[2] = max(0, min(x + increment, X_MAX - 1)); self.update_display()

    def measure_hu(self, p, vid):
        """探针工具：读取鼠标处体素的 HU 值并显示。

        【读不出来时必须清空，不能留着上一次的值】旧实现把整段包在 try/except: pass 里，
        坐标越界或 plane 取到三者之外时静默失败，标签于是**继续显示上一次的读数**——
        连坐标都是旧的。实测：鼠标移到体积外后，标签仍写着 "(10, 10) : 4426.0 HU"，
        看上去完全像一次有效读数，用户无从分辨它其实对应别的位置。对 HU 这种会被
        直接用于判读的数值，陈旧显示比空白危险得多。
        """
        if not (self.active_tool == TOOL_POINTER and self.volume_hu is not None
                and not self.recon_mode_active and not self.compare_mode_active):
            return
        vd = self.views.get(vid)
        if vd is None: return
        c = vd['view'].get_real_coordinates(p); plane = vd['plane']
        # 三平面各自的 (z,y,x) 索引；plane 不在三者之内时得 None，与越界同样处理
        idx = None if not c else {
            AXIAL: (self.current_3d_pos[0], c[1], c[0]),
            CORONAL: (c[1], self.current_3d_pos[1], c[0]),
            SAGITTAL: (c[1], c[0], self.current_3d_pos[2]),
        }.get(plane)
        if idx is None or not all(0 <= i < n for i, n in zip(idx, self.volume_hu.shape, strict=True)):
            self.lbl_hu_value.setText("")   # 读不出就清空，绝不留旧值冒充当前读数
            return
        names = ({AXIAL: "Axial", CORONAL: "Coronal", SAGITTAL: "Sagittal"} if self.is_english
                 else {AXIAL: "横断面", CORONAL: "冠状面", SAGITTAL: "矢状面"})
        self.lbl_hu_value.setText(
            f"V{vid} [{names[plane]}] ({c[0]}, {c[1]}) : {float(self.volume_hu[idx]):.1f} HU")

