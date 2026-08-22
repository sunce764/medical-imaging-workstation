# =============================================================================
# AI 推理引擎模块
# 负责：异步后台 AI 多器官分割推理（organs.onnx，25 类）
# 设计：纯 Python daemon 线程，避免继承 QThread 的析构崩溃风险；
#       完成/进度经 Qt 信号（QueuedConnection）投递回主线程更新 UI
# =============================================================================

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable

import numpy as np
import shiboken6  # 随 PySide6 一同安装（PySide6_Essentials 硬依赖 shiboken6==6.11.0），非新增依赖
from PySide6.QtCore import QObject, Signal
from scipy import ndimage

import segmentation
from constants import MODEL_PATH

# organs.onnx = TotalSegmentator v2（nnU-Net v2），训练 spacing 为 1.5mm 各向同性。
# 推理前重采样到该值是 nnU-Net 的推理契约，不是可选优化：实测偏离一倍即掉 13% Dice
# （experiments/seg_spacing.py）。
TARGET_SPACING = 1.5
# 重采样后的体素数上限，防止对**粗** spacing 放大时 OOM（已知 61M 体素约对应 8.8GB 峰值，
# 取 70M 留余量）。实际很少触发：重采样到固定 1.5mm 之后，体素数只由扫描 FOV 决定
# ——胸腹 CT 约 400mm³ 恒定落在 19M 上下，与原始 spacing 无关。这正是重采样的一个额外
# 好处：推理的内存与耗时不再随扫描协议波动，变成可预期的常量。能超限的是全身长范围扫描。
_MAX_RESAMPLED_VOXELS = 70_000_000


def _fit_shape(arr: np.ndarray, shape: tuple) -> np.ndarray:
    """把 zoom 结果对齐到目标 shape：多出来的裁掉，少的用边缘值补。

    zoom 的输出尺寸由 round(n*factor) 决定，与来回两次缩放的目标可能差 1 个体素。
    差一格若不处理，回调里的 `final_mask.shape != volume_hu.shape` 判定会直接丢弃
    整次推理结果——一百秒白跑，且界面只是安静地什么都不显示。
    """
    if arr.shape == tuple(shape):
        return arr
    out = np.zeros(shape, dtype=arr.dtype)
    sl = tuple(slice(0, min(a, b)) for a, b in zip(arr.shape, shape, strict=True))
    out[sl] = arr[sl]
    # 尾部若有缺口，用最后一个有效切片沿各轴补齐，避免边界出现整层空洞
    for ax, (a, b) in enumerate(zip(arr.shape, shape, strict=True)):
        if a < b:
            idx = [slice(None)] * len(shape)
            src = list(idx); src[ax] = slice(a - 1, a)
            for k in range(a, b):
                dst = list(idx); dst[ax] = slice(k, k + 1)
                out[tuple(dst)] = out[tuple(src)]
    return out


class _AISignals(QObject):
    """跨线程回调载体。在主线程创建，子线程 emit 时 Qt 以 QueuedConnection 自动投递到
    主线程事件循环——这是从 threading.Thread 安全更新 Qt UI 的正确方式。
    （不能用 QTimer.singleShot：它依附调用它的子线程，而子线程无 Qt 事件循环，回调不 fire。）"""
    finished = Signal(object, float)   # (label_map, elapsed_ms)
    progress = Signal(int, int)        # (done_slices, total_slices)
    failed = Signal(str)               # 推理彻底失败（含兜底路径），载荷为原因摘要

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


# 模块级 InferenceSession 缓存：模型加载（含 119MB 外部权重）有固定开销，
# 按 model_path 缓存复用，避免每次推理（尤其快速切换数据）重复加载。
# onnxruntime 的 session.run 是线程安全的，可被多个后台线程共享。
_SESSION_CACHE = {}


def _get_session(model_path: str):
    sess = _SESSION_CACHE.get(model_path)
    if sess is None:
        so = ort.SessionOptions()
        so.enable_cpu_mem_arena = False  # 关内存池，显著降低分块推理的峰值内存
        sess = ort.InferenceSession(model_path, sess_options=so,
                                    providers=['CPUExecutionProvider'])
        _SESSION_CACHE[model_path] = sess
    return sess


class AutoAIEngineThread:
    """在后台线程中执行 AI 多器官分割推理，完成后通过 Qt 信号安全回调主线程。"""

    def __init__(self, volume_hu: np.ndarray,
                 callback: Callable[[np.ndarray, float], None],
                 model_path: str = MODEL_PATH,
                 progress_callback: Callable[[int, int], None] | None = None,
                 failed_callback: Callable[[str], None] | None = None,
                 spacing: tuple[float, float, float] | None = None) -> None:
        # volume_hu: 完整的 3D HU 值体素数组，shape=(Z, H, W)，float32
        # callback: 推理完成后调用，签名为 callback(label_map, elapsed_ms)
        #           label_map 为 uint8 多类标签图（0=背景，1-24=器官类别）
        # model_path: ONNX 模型文件绝对路径，默认 constants.MODEL_PATH（models/organs.onnx）
        # progress_callback: 可选，签名 progress_callback(done_slices, total_slices)，
        #                    在滑窗推理过程中经 Qt 信号（QueuedConnection）投递到主线程，
        #                    用于更新进度显示
        self.volume_hu = volume_hu
        self.callback = callback
        self.model_path = model_path
        self.progress_callback = progress_callback
        self.failed_callback = failed_callback
        self._thread = None
        # 体素物理间距 (z, y, x)，单位 mm，轴序与 volume_hu 一致。None 表示未知，
        # 此时不做 spacing 重采样——插值必须基于真实物理尺寸，猜一个只会更糟。
        self.spacing = tuple(float(s) for s in spacing) if spacing is not None else None
        # 实际发生过重采样时记为 (原 shape, 送入模型的 shape)，供 UI 如实告知用户
        self.resampled_from = None
        self.used_fallback = False   # 是否退到了数学降级（供界面如实标注，见 _run_body）
        # 逐体素置信度（softmax 最大类概率，量化为 uint8 的 0-255 对应 0-1）。
        # 走实例属性而非扩展 finished 信号：不改动已有的信号契约与回调签名。
        # Qt 队列连接的投递自带内存屏障，emit 前的写入对主线程槽函数可见。
        # 数学降级路径没有概率输出，此时保持 None——宁可不显示，也不编一个数字。
        self.confidence = None
        # 信号对象在此（主线程）创建，其槽即在主线程执行；子线程 emit 自动排队投递。
        # 【刻意不设 parent，勿"优化"】实测（PySide6 6.11.0，15×15 对照）：不设 parent 时
        # 宿主 widget 销毁后信号源仍存活（isValid=True），emit 正常，0/15 异常；一旦设
        # parent=viewer，Qt 会在宿主析构时连带删除本对象，后台线程的 emit 必抛
        # RuntimeError('Signal source has been deleted') —— 15/15 必现，且把 Python 层
        # 异常升级为 C++ 跨线程 use-after-free。生存期已由后台线程栈帧持有本引擎保证。
        self._signals = _AISignals()
        self._signals.finished.connect(lambda m, t: self.callback(m, t))
        if progress_callback is not None:
            self._signals.progress.connect(lambda d, t: self.progress_callback(d, t))
        if failed_callback is not None:
            self._signals.failed.connect(lambda why: self.failed_callback(why))
        # 单次推理约 8.8GB 内存、~100s 且不可中途中断 onnxruntime.run。快速切换数据时
        # 若不作废旧线程，多个推理会并发叠加导致内存翻倍甚至 OOM。cancel() 置位后，
        # 滑窗循环在下一个 z 块边界提前退出并放弃回调，及时释放内存。
        self._cancelled = False

    def start(self) -> None:
        # daemon=True 确保主窗口关闭后不会因后台线程阻塞进程退出
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        """作废本次推理：滑窗循环会在下一个 z 块边界停止，且不再触发完成回调。"""
        self._cancelled = True

    def isRunning(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _emit_safe(self, signal, *args) -> bool:
        """向主线程投递信号；若信号源已被拆卸则返回 False（不抛异常）。

        QApplication 拆卸 / 进程退出时，_AISignals 的 C++ 侧可能已被销毁，此后任何
        emit 都抛 RuntimeError('Signal source has been deleted')。那是退出期竞态，
        不是推理失败——不加守卫的话它会被 _run 的 except 吞成「ONNX 推理失败」（日志说假话），
        或从 finished.emit 处直接冲出线程打印 traceback。
        投递机制仍是 Qt 信号（QueuedConnection），此处只加一道前置存活校验。
        """
        if not shiboken6.isValid(self._signals):
            return False
        try:
            signal.emit(*args)
            return True
        except RuntimeError:
            return False   # isValid 与 emit 之间宿主被拆卸的 TOCTOU 窗口，同样按拆卸处理

    def _run(self) -> None:
        """推理主体的顶层守卫。真正的工作在 _run_body。

        为何需要这一层：ONNX 分支自带 try，但**兜底的数学降级路径没有**——兜底本身没有
        兜底。实测（构造 segment_lungs_fallback 抛异常）：异常直接冲出 _run，线程死掉，
        Python 只在终端打印「Exception in thread」，而界面上 _ai_state 永远停在 'running'、
        状态栏永远显示「AI 引擎自动运算中…」——用户在等一个永远不会来的结果，且毫无提示。

        失败时刻意**不** emit finished(空 mask)：那会让界面显示「检出 0 个器官」，把
        「失败了」谎报成「成功了但没找到东西」。日志与界面都必须说实话。
        """
        try:
            self._run_body()
        except Exception as e:                      # noqa: BLE001 — 顶层守卫，必须兜住一切
            if self._cancelled:
                return    # 拆卸/切数据期间的连带异常，不是真失败，静默退出
            print(f"AI 分割彻底失败（含兜底路径）: {type(e).__name__}: {e}")
            self._emit_safe(self._signals.failed, f"{type(e).__name__}: {e}")

    def _plan_resample(self):
        """决定是否重采样到训练 spacing，返回 (zoom 因子, 重采样后 shape) 或 None。

        organs.onnx 是 nnU-Net v2，其推理契约的第一步就是把体积重采样到训练 spacing
        （1.5mm 各向同性）。本引擎长期跳过这一步，代价已实测（experiments/seg_spacing.py）：
        spacing 变为 2 倍时平均 Dice 从 0.922 掉到 0.799，且小器官先垮、并非单调。

        三种情况不做重采样，各有实测理由：
          1. **不知道 spacing**（调用方没传，或 DICOM 缺 tag）——插值必须基于真实物理
             尺寸，猜一个只会更糟；
          2. **已经足够接近 1.5mm**（各轴偏差 < 5%）——重采样本身带插值损失，
             为了消除 5% 的失配去引入一次插值不划算；
          3. **重采样后体素数超过上限，【且】比原始还大**——对**粗** spacing（如层厚
             5mm 的临床序列）重采样是**放大**，z 方向可涨 3 倍以上，会直接 OOM。
             此时宁可维持失配也不能把应用跑崩，并在返回值里让调用方知道跳过了。

        第 3 条的「且比原始还大」不可省。此前的判据只看重采样后的体素数、不与原始比较，
        于是当重采样是**缩小**（细 spacing → 1.5mm）而缩小后仍超上限时，也会跳过——
        送进 ONNX 的反而是更大的原始体积，保护措施制造了它本要防的那次 OOM。
        另外第 1、2 条在算 shape 之前就返回，本身不经过任何体积检查，那两条路径下
        原始体积多大都不会被发现（1.5mm 各向同性的全身扫描恰好走第 2 条），故在
        函数开头统一算一次原始体素数并在超限时告警——那种情形重采样帮不上忙，
        能做的只有让它可见，而不是继续沉默。
        """
        orig_n = int(np.prod(self.volume_hu.shape))
        if orig_n > _MAX_RESAMPLED_VOXELS:
            print(f"AI: 原始体积已达 {orig_n/1e6:.0f}M 体素，超出内存上限 "
                  f"{_MAX_RESAMPLED_VOXELS/1e6:.0f}M；重采样与否都无法回避，推理可能失败")
        sp = self.spacing
        if sp is None or len(sp) != 3 or not all(np.isfinite(s) and s > 0 for s in sp):
            return None
        f = tuple(float(s) / TARGET_SPACING for s in sp)
        if all(abs(x - 1.0) < 0.05 for x in f):
            return None
        shape = tuple(max(1, int(round(n * x))) for n, x in zip(self.volume_hu.shape, f, strict=True))
        new_n = int(np.prod(shape))
        if new_n > _MAX_RESAMPLED_VOXELS and new_n > orig_n:
            print(f"AI: spacing {sp} 重采样后达 {new_n/1e6:.0f}M 体素（原始 {orig_n/1e6:.0f}M），"
                  f"放大且超出内存上限，跳过重采样（准确度将受 spacing 失配影响）")
            return None
        return f, shape

    def _run_body(self) -> None:
        """推理主体，运行在后台线程，严禁在此处操作任何 Qt 对象（非线程安全）。"""
        start_t = time.perf_counter()

        # HU 值归一化到 [0, 1]：肺窗范围 -1000~400 HU 覆盖空气到软组织
        # 超出范围的 HU 值（骨骼 > 400）通过 clip 截断，防止影响网络输入分布
        norm_vol = np.clip(self.volume_hu, -1000, 400)
        norm_vol = (norm_vol - (-1000)) / (400 - (-1000))
        norm_vol = norm_vol.astype(np.float32)

        final_mask = None
        orig_shape = norm_vol.shape
        plan = self._plan_resample()
        if plan is not None:
            f, _ = plan
            # order=1：图像用线性插值。归一化后的体积是连续量，最近邻会产生阶梯伪影。
            norm_vol = ndimage.zoom(norm_vol, f, order=1, prefilter=False).astype(np.float32)
            self.resampled_from = (orig_shape, norm_vol.shape)

        # === 路径1：真实 ONNX 多器官分割推理 ===
        if HAS_ONNX and os.path.exists(self.model_path):
            try:
                final_mask = self._run_onnx_multiorgan(norm_vol)
                if final_mask is not None and plan is not None:
                    # 标签与置信度都必须回到原网格，否则与 volume_hu 对不上。
                    # order=0 最近邻：标签是离散的，插值会造出不存在的类别；
                    # 置信度同样走最近邻，保证与它所描述的那个标签严格同源。
                    back = [o / s for o, s in zip(orig_shape, final_mask.shape, strict=True)]
                    final_mask = ndimage.zoom(final_mask, back, order=0, prefilter=False)
                    final_mask = _fit_shape(final_mask, orig_shape)
                    if self.confidence is not None:
                        cf = ndimage.zoom(self.confidence, back, order=0, prefilter=False)
                        self.confidence = _fit_shape(cf, orig_shape)
            except RuntimeError as e:
                # Qt 生命周期竞态，不是模型问题：进度 emit 在 QApplication 拆卸后会抛
                # RuntimeError('Signal source has been deleted')。实测 onnxruntime 1.23.2 的
                # 14 个异常类型全部直接继承 Exception、无一继承 RuntimeError，故此分支
                # 不会误吞真正的推理失败。日志必须说实话：这不是「ONNX 推理失败」。
                print(f"AI 推理在应用拆卸期间中断（非模型故障）: {e}")
                self._cancelled = True   # 宿主已在拆卸，无需再做数学降级，让线程尽早退出
            except Exception as e:
                print(f"ONNX 推理失败，降级为数学算法: {e}")

        # 推理被作废（用户已切换到新数据）：直接退出，不回调、不做数学降级，释放内存
        if self._cancelled:
            return

        # === 路径2：纯数学算法降级（无模型文件或推理失败时自动启用）===
        # 逻辑已抽到 segmentation.segment_lungs_fallback（无 Qt，可独立单测）：
        # 阈值取低密度空气 → 3D 连通域 → 剔除体表边界相连的体外空气 → 剩余内部空气取
        # 最大(及≥其5%的次大)连通域为双肺。
        if final_mask is None:
            # 【降级必须可见】走到这里意味着 ONNX 没有产出结果（权重缺失、session 建不起来、
            # 推理抛异常）。不标记的话，界面与「检出 N 个器官」的正常成功文案完全一样，
            # 用户无从知道拿到的是连通域算法的粗略结果而非 25 类模型输出。
            self.used_fallback = True
            self.resampled_from = None   # 降级走原网格，重采样信息属于那次失败的 ONNX 尝试
            final_mask = segmentation.segment_lungs_fallback(self.volume_hu)

        if self._cancelled:
            return  # 数学降级期间也可能被作废，退出前再确认一次
        final_mask = final_mask.astype(np.uint8)  # 统一为 uint8 标签图，供调色板 LUT 索引
        end_t = time.perf_counter()
        # 经信号跨线程投递到主线程（QueuedConnection），安全更新 Qt UI。
        # 走 _emit_safe：拆卸期若信号源已销毁，此处原本会抛 RuntimeError 冲出 _run，
        # 由 threading 打印「Exception in thread」traceback（此 emit 不在任何 try 内）。
        self._emit_safe(self._signals.finished, final_mask, (end_t - start_t) * 1000)

    def _run_onnx_multiorgan(self, norm_vol: np.ndarray) -> np.ndarray | None:
        """ONNX 多器官分割推理，返回 uint8 标签图（0=背景，1-24=器官类别）。

        关键约束（均由对 organs.onnx 的实测确定）：
          - 输入 (1,1,D,H,W)，每个空间维必须 pad 到 32 的倍数；
          - 必须整幅 xy 送入，做中心裁剪会破坏全局上下文导致器官被误判为背景；
          - 输出 (1,25,D,H,W) 为 logits，取 argmax(axis=1) 得类别，而非阈值二值化；
          - 整卷一次推理输出约 6.7GB 会 OOM，故沿 z 分块滑窗（DZ=32）+ 关闭
            CPU 内存池，把峰值压到约 8.8GB。
        """
        Z, H, W = norm_vol.shape
        session = _get_session(self.model_path)  # 复用缓存的会话，避免重复加载模型
        input_name = session.get_inputs()[0].name

        ph, pw = (-H) % 32, (-W) % 32  # xy 方向对齐到 32 的倍数所需的填充
        seg = np.zeros((Z, H, W), dtype=np.uint8)
        # 置信度量化到 uint8：float32 存整卷要 244MB，uint8 只要 61MB，
        # 而「模型有多确信」的显示精度远用不到 float32
        conf = np.zeros((Z, H, W), dtype=np.uint8)
        DZ = 32
        # 【末块回移，别用空气把它填满】原先 z0 直接按 range(0, Z, DZ) 递增，末块只剩
        # Z % 32 层真实数据，其余由 pad(mode='constant') 补 0——而 HU 归一化后 0 就是
        # 空气(-1000)。实测 233 层 @1.25mm 重采样到 194 层时，末块只有 2 层真实数据、
        # 30 层合成空气，等于让模型在一个几乎全空的 slab 里判断那 2 层。nnU-Net 的滑窗
        # 做法是把末窗回移到 [Z-DZ, Z) 再融合；这里同样回移，重叠部分由后一块的 argmax
        # 覆盖。代价为零，且只影响末块——其余块的 z0 不变。
        # （块间仍无重叠、每 32 层有一道硬接缝，那是另一件事：改它要引入 logit 融合，
        #   实测收益 +0.0133 Dice 但 +0.65GB 峰值，见 experiments 的 F 节，未采用。）
        starts = [min(z0, max(0, Z - DZ)) for z0 in range(0, Z, DZ)]
        for z0 in starts:
            if self._cancelled:
                return None  # 已作废：停止推理，让 _run 放弃回调并释放内存
            z1 = min(z0 + DZ, Z)
            blk = norm_vol[z0:z1]
            pd = (-blk.shape[0]) % 32  # z 方向（尤其末块）对齐到 32
            if pd or ph or pw:
                blk = np.pad(blk, ((0, pd), (0, ph), (0, pw)), mode='constant')
            out = session.run(None, {input_name: blk[np.newaxis, np.newaxis].astype(np.float32)})[0][0]
            lab = out.argmax(0).astype(np.uint8)  # (D',H',W')
            seg[z0:z1] = lab[:z1 - z0, :H, :W]     # 裁掉 pad 部分写回原尺寸
            # 逐体素置信度 = softmax 最大类概率。减去 max 再 exp 是数值稳定写法，
            # 且减完之后最大项恒为 exp(0)=1，故 max-prob 直接等于 1/Σexp(x-max)。
            # 两步都对 out 原地做，不额外分配 (25,D,H,W)（单块 839MB，推理峰值已 8.8GB）。
            # 实测代价：整卷约 +3s（基线 ~100s）。曾以为 top1-top2 的 np.partition 更省，
            # 实测反而慢 5.5 倍——它沿 axis=0 跨步重排，内存访问模式远差于顺序的 in-place exp。
            out -= out.max(0, keepdims=True)
            np.exp(out, out=out)
            cf = 1.0 / out.sum(0)                  # (D',H',W') ∈ (0,1]
            # 量化时下限钳到 1：0 被留作「此体素无模型置信度」的哨兵（手动追踪、
            # 画笔编辑过的体素）。25 类 softmax 的最大类概率下限是 1/25=0.04，
            # 量化后为 10，模型本身永远产不出 0，故这个哨兵不会与真实值混淆。
            conf[z0:z1] = np.clip(cf[:z1 - z0, :H, :W] * 255.0, 1, 255).astype(np.uint8)
            del out, lab, cf
            if self.progress_callback is not None:
                # 经信号跨线程投递到主线程更新进度显示（子线程禁止直接操作 Qt）
                self._emit_safe(self._signals.progress, z1, Z)
        self.confidence = conf   # 仅 ONNX 路径产出；兜底路径不设，保持 None
        return seg
