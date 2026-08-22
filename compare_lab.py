# =============================================================================
# 双序列随访对比 Mixin
# 负责：加载既往序列并排显示，按 ImagePositionPatient 解剖配准、切片/窗位联动。
#
# 设计：以 Mixin 形式并入 MedicalViewer（class MedicalViewer(QMainWindow,
#       ReconLabMixin, CompareMixin)）。方法通过 self 访问主窗口的 UI 控件与
#       状态（self.views / self.slider_* / self.compare_* / self.dicom_datasets
#       等）及留在 main.py 的共享方法（_read_dicom_dir / _dcm_float /
#       set_view_title / update_display）。
#       对比状态（compare_mode_active / compare_volume / compare_datasets /
#       _primary_zpos / _compare_zpos）在 MedicalViewer.__init__ 中初始化；
#       update_display 里对 compare_mode_active 的分派也留在 main。
# =============================================================================

import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFileDialog, QMessageBox

import followup
import mpr_geometry
import registration


class CompareMixin:
    """双序列随访对比相关方法集合，混入 MedicalViewer。"""

    def toggle_compare(self):
        """加载对比（既往）序列进入对比模式；已在对比模式则退出。"""
        if self.compare_mode_active:
            self._exit_compare_mode()
            return
        if self.volume_hu is None or self.recon_mode_active:
            QMessageBox.information(self, "Info", "Load a primary series first (clinical mode)."
                                    if self.is_english else "请先在临床阅片模式加载主序列。")
            return
        path = QFileDialog.getExistingDirectory(
            self, "Select comparison series" if self.is_english else "选择对比序列目录")
        if not path:
            return
        vol, dsets = self._read_compare_dir(path)
        if vol is None:
            QMessageBox.warning(self, "Failed", "Cannot read a DICOM series from that folder."
                                if self.is_english else "无法从该目录读取 DICOM 序列。")
            return
        self.compare_volume, self.compare_datasets = vol, dsets
        self._enter_compare_mode()

    def _read_compare_dir(self, path):
        """读取对比序列，返回 (volume_hu, datasets)。复用主读取逻辑但不破坏当前 self.dicom_datasets。"""
        saved = self.dicom_datasets
        try:
            if not self._read_dicom_dir(path):
                return None, []
            dsets = self.dicom_datasets
            vol = np.array([
                d.pixel_array.astype(np.float32) * self._dcm_float(d, 'RescaleSlope', 1.0) +
                self._dcm_float(d, 'RescaleIntercept', 0.0) for d in dsets])
            return vol, dsets
        except Exception as e:
            print(f"读取对比序列失败: {e}")
            return None, []
        finally:
            self.dicom_datasets = saved   # 恢复主序列，绝不让对比读取污染主数据

    @staticmethod
    def _zpos_array(datasets):
        """取各切片的解剖 z 坐标 ImagePositionPatient[2]；任一缺失或非有限则返回 None。

        【NaN 必须和缺失同等对待】float(nan) 不抛异常，所以畸形 DICOM 里一个空的
        ImagePositionPatient[2] 会安静地进到数组里；随后 nearest_slice 的 np.argmin
        遇 NaN 会返回那个 NaN 的下标，于是【每一个】主序列层都映射到同一层既往切片，
        而标题栏仍打「· 配准」，Δ / 绝对差 / RMSE 全部基于错误配对算出——看起来对、
        数值全错。main.py 的 _slice_spacing 用同一个数组时是显式 np.isfinite 过滤的，
        说明这条风险早就被认识到了，只是 nearest_slice 这条路没兜住。
        返回 None 会让调用方回退到按索引比例配准，那是可见且可解释的降级。
        """
        try:
            a = np.array([float(d.ImagePositionPatient[2]) for d in datasets])
        except Exception:
            return None
        return a if a.size and np.isfinite(a).all() else None

    def _enter_compare_mode(self):
        """进入对比模式：强制双窗、关闭 MPR、切换按钮文案。"""
        self.compare_mode_active = True
        self._primary_zpos = self._zpos_array(self.dicom_datasets)
        self._compare_zpos = self._zpos_array(self.compare_datasets)
        self._pre_compare_layout = self.combo_layout.currentIndex()
        self.btn_mpr.setChecked(False)
        for vd in self.views.values():
            vd['view'].draw_crosshair(0, 0, show=False)
        self.combo_layout.setCurrentIndex(1)   # 1x2 双窗
        self.btn_compare.setText("Exit Compare" if self.is_english else "退出对比")
        self.chk_register.setEnabled(True)      # 配准只在对比模式下有意义
        self.update_display()

    def _exit_compare_mode(self):
        """退出对比模式：释放对比序列、还原布局与按钮。"""
        self.compare_mode_active = False
        self.compare_volume = None
        self.compare_datasets = []
        self._primary_zpos = self._compare_zpos = None
        self.btn_compare.setText("Load Comparison" if self.is_english else "加载对比序列")
        # 勾选状态刻意保留（下次进入对比仍是用户上次的偏好），只收回可操作性
        self.chk_register.setEnabled(False)
        self.combo_layout.setCurrentIndex(self._pre_compare_layout)
        self.update_display()

    def _render_compare(self):
        """对比模式渲染：V1=当前序列，V2=既往序列（按比例映射切片），共享窗位。"""
        z = self.current_3d_pos[0]
        ww, wl = self.slider_ww.value(), self.slider_wl.value()
        self.lbl_ww.setText(f"WW: {ww}"); self.lbl_wl.setText(f"WL: {wl}")
        Z1 = self.volume_hu.shape[0]
        self._show_windowed(1, self.volume_hu[z], ww, wl)
        self.set_view_title(1, f"V1 [Current {z + 1}/{Z1}]" if self.is_english else f"V1 [当前 {z + 1}/{Z1}]")
        Z2 = self.compare_volume.shape[0]
        # 优先按解剖 z 坐标配准（同一解剖层面），缺位置信息才回退到索引比例
        if self._primary_zpos is not None and self._compare_zpos is not None and len(self._primary_zpos) > z:
            z2 = mpr_geometry.nearest_slice(self._compare_zpos, self._primary_zpos[z])
            reg = "配准" if not self.is_english else "reg"
        else:
            z2 = min(Z2 - 1, max(0, int(round(z / max(1, Z1 - 1) * (Z2 - 1)))))
            reg = "比例" if not self.is_english else "ratio"
        prior = self.compare_volume[z2]
        rtag = ''
        # 平面内刚性配准：勾选后先把既往层对齐到当前层，再显示与算差异。
        # 只在同尺寸时尝试；registration 内部带安全阀，NCC 不升则不采用（applied=False），
        # 此时 rtag 明确标出"未采用"，避免用户以为已配准。
        if getattr(self, 'chk_register', None) is not None and self.chk_register.isChecked():
            if prior.shape == self.volume_hu[z].shape:
                r = registration.register_rigid(self.volume_hu[z], prior)
                if r['applied']:
                    prior = registration.apply_rigid(prior, r['angle_deg'], r['shift_yx'])
                    dy, dx = r['shift_yx']
                    rtag = (f" · reg {r['angle_deg']:+.1f}° ({dy:+d},{dx:+d}) NCC {r['ncc_before']:.2f}→{r['ncc_after']:.2f}"
                            if self.is_english else
                            f" · 配准 {r['angle_deg']:+.1f}° ({dy:+d},{dx:+d}) NCC {r['ncc_before']:.2f}→{r['ncc_after']:.2f}")
                else:
                    rtag = " · reg not applied" if self.is_english else " · 配准未采用"
            else:
                rtag = " · reg n/a (size)" if self.is_english else " · 配准不适用（尺寸不同）"
        self._show_windowed(2, prior, ww, wl)
        # 标题带既往检查日期（脱敏时隐去——检查日期属可识别信息）
        date = '' if self.anonymize else (str(getattr(self.compare_datasets[0], 'StudyDate', '')) if self.compare_datasets else '')
        dtag = f" {date[:4]}-{date[4:6]}-{date[6:8]}" if len(date) == 8 else ''
        stat = self._compare_stat_text(self.volume_hu[z], prior)
        self.set_view_title(2, (f"V2 [Prior{dtag} {z2 + 1}/{Z2} · {reg}]{rtag}{stat}" if self.is_english
                                else f"V2 [既往{dtag} {z2 + 1}/{Z2} · {reg}]{rtag}{stat}"))

    def _compare_stat_text(self, cur, prev):
        """本层 HU 差异定量，返回可直接拼进 V2 标题的一段文字（无差异可比时返回提示）。

        计算全在无 Qt 的 followup 模块，本方法只做取值与格式化。
        差值图另在 V3 渲染——但对比模式默认是 1×2 双窗，V3 处于隐藏状态，
        故仅当用户切到四窗（V3 可见）时才渲染，避免为看不见的视图做无用计算。

        【勿夸大】未勾选「配准」时只做了 z 轴层面对齐，平面内未校正，差值中混有体位
        与呼吸相位差异；勾选后已做平面内刚性配准（平移+旋转），但仍无形变配准，
        呼吸导致的器官形变不会被校正。两种情况都只宜作定性参考，不是临床意义上的
        病灶变化量——故标题按当前状态如实标注是哪一种。
        """
        e = self.is_english
        registered = (getattr(self, 'chk_register', None) is not None
                      and self.chk_register.isChecked())
        scope = ("rigid-aligned, no deformable" if registered else "z-aligned only") if e else \
                ("已刚性配准，无形变校正" if registered else "仅z轴对齐")
        ok, why = followup.can_compare(cur, prev)
        if not ok:
            return f" · Δ n/a ({why})" if e else f" · Δ 不可比（{why}）"
        s = followup.compare_slices(cur, prev)
        # 差值图叠加到 V3——仅在它确实可见时才做（默认双窗布局下 V3 隐藏）
        if 3 in self.views and not self.views[3]['container'].isHidden():
            d = followup.diff_map(cur, prev)
            self._show_windowed(3, cur, self.slider_ww.value(), self.slider_wl.value())
            rgba = np.ascontiguousarray(followup.diff_to_rgba(d))
            h, w = d.shape
            mq = QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
            v3 = self.views[3]['view']
            v3.set_image(v3.image_item.pixmap(), mq)
            self.set_view_title(3, "V3 [Difference map · warm = denser now]" if e
                                else "V3 [差值图 · 暖色=当前更致密]")
        return (f" · Δ{s['mean_diff']:+.0f} MAE {s['mae']:.0f} RMSE {s['rmse']:.0f} HU ({scope})" if e
                else f" · Δ{s['mean_diff']:+.0f} 绝对差 {s['mae']:.0f} RMSE {s['rmse']:.0f} HU（{scope}）")

    def _show_windowed(self, vid, hu2d, ww, wl):
        """把 2D HU 切片按窗位映射为灰度显示到指定视图（对比模式用，不叠加蒙版/标注/四角信息）。"""
        img = np.clip(hu2d, wl - ww / 2, wl + ww / 2)
        img = ((img - (wl - ww / 2)) / ww * 255).astype(np.uint8)
        if self.chk_invert.isChecked():
            img = 255 - img
        img = np.ascontiguousarray(img)
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8).copy()
        v = self.views[vid]['view']
        v.set_image(QPixmap.fromImage(qimg))
        v.set_overlay({}, {})
        v.clear_annotations()
