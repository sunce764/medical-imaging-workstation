#!/usr/bin/env python3
"""为本地 RIDER 副本补写 explicit ``RescaleType=HU``，使其满足产品的单位证明契约。

**为什么需要这一步**：本地 RIDER 副本每层的 ``ImageType`` 是
``DERIVED\\SECONDARY\\PROCESSED`` 且缺失 ``RescaleType``。按 DICOM PS3.3 C.8.2，
省略 Rescale Type 只有对 ``ORIGINAL`` 的 classic CT 才可推定为 HU，因此
``dicom_geometry._slice_has_standard_hu`` 正确地 fail closed，整卷退为
viewer-only（关闭 CT preset / HU 定量 / AI / 3D / 随访）。

**本脚本做什么**：先用物理锚点验证该序列的数值确为标准 HU，验证通过后写出一份副本，
副本中仅新增 ``RescaleType=HU`` 这一个 tag。

**本脚本明确不做什么**：
  * 不修改 ``ImageType`` —— 该序列确实是 DERIVED，谎称 ORIGINAL 才是造假。
    DICOM 允许 DERIVED 图像显式声明 ``RescaleType=HU``，两者不矛盾。
  * 不触碰 ``PixelData`` —— 不解压、不重编码，逐字节原样搬运并在写出后校验。
  * 不修改源目录 —— 源为只读输入，输出目录必须与源不同。

**副本与源的完整差异面**（写出后逐文件断言，多一处即报错）：
  * 新增 ``(0028,1054) RescaleType = HU``，仅此一个。
  * 丢失 retired 的 Group Length ``(gggg,0000)`` 元素。这不是本脚本的选择：
    pydicom 依 DICOM PS3.5 §7.2 一律不写出它们（``filewriter.py`` 中
    ``if tag.element == 0 and tag.group > 6: continue``）。丢弃也是唯一正确的处理
    —— 新增元素后 group 0028 的字节长度已变，保留旧的 ``(0028,0000)`` 反而自相矛盾。
    Group Length 不携带任何临床或几何语义，读取方不依赖它。
  * 除以上两类外，不允许任何 tag 新增、删除或改值。

依据：空气峰 -1025 HU、软组织峰 -5 HU、去 padding 后值域 -1024..3071
（即 12-bit CT 标准区间），slope=1 / intercept=-1024 全序列一致。
"""
import argparse
import os
import sys

import numpy as np
import pydicom

# 验证判据的容许区间。取值依据 CT 的物理锚点：空气 ≈ -1000 HU、水 ≈ 0 HU。
# 区间放宽到足以容纳重建核差异与体外空气被 clip 到存储下界的情况，
# 但仍窄到足以否决单位不是 HU 的序列（如 raw storage value 或线性衰减系数）。
AIR_PEAK_RANGE = (-1100.0, -900.0)
SOFT_PEAK_RANGE = (-100.0, 150.0)
HU_VALUE_RANGE = (-1100.0, 3200.0)
SAMPLE_STRIDE = 10  # 验证阶段的抽样步长，233 层取 24 层足够定位直方图峰
RESCALE_TYPE_TAG = pydicom.tag.Tag(0x0028, 0x1054)  # (0028,1054) RescaleType


def _iter_files(root):
    """按稳定顺序列出目录下的全部普通文件。"""
    out = []
    for dirpath, _, names in os.walk(root):
        for name in sorted(names):
            path = os.path.join(dirpath, name)
            if os.path.isfile(path):
                out.append(path)
    return sorted(out)


def _read_scale(ds):
    """取出 slope/intercept，非有限或 slope=0 一律视为不可用。"""
    try:
        slope = float(ds.RescaleSlope)
        intercept = float(ds.RescaleIntercept)
    except (AttributeError, TypeError, ValueError):
        return None
    if not (np.isfinite(slope) and np.isfinite(intercept)) or slope == 0:
        return None
    return slope, intercept


def verify_is_hu(files):
    """用物理锚点证明该序列的数值确为标准 HU。返回 (ok, 诊断行列表)。"""
    notes = []

    # ① 全序列必须共享同一组有限、非零的 slope/intercept，否则是 mixed-unit series。
    scales = set()
    for path in files:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        if getattr(ds, "Modality", None) != "CT":
            return False, [f"拒绝：{path} 的 Modality 不是 CT"]
        scale = _read_scale(ds)
        if scale is None:
            return False, [f"拒绝：{path} 缺少可用的 RescaleSlope/Intercept"]
        scales.add(scale)
    if len(scales) != 1:
        return False, [f"拒绝：序列内 slope/intercept 不唯一，共 {len(scales)} 组"]
    slope, intercept = scales.pop()
    notes.append(f"slope={slope:g} intercept={intercept:g}（全 {len(files)} 层一致）")

    # ② 已显式声明单位的序列不该走这个脚本——要么已经是 HU，要么明确不是 HU。
    for path in files:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        if getattr(ds, "RescaleType", None) not in (None, ""):
            return False, [f"拒绝：{path} 已有 RescaleType={ds.RescaleType!r}，本脚本不覆盖既有声明"]

    # ③ 多能量 CT 的单位契约本产品未实现，不在本脚本适用范围内。
    for path in files:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        multi = getattr(ds, "MultienergyCTAcquisition", None)
        if multi not in (None, "", "NO"):
            return False, [f"拒绝：{path} 为 multi-energy（{multi!r}），单位契约未实现"]

    # ④ 直方图物理锚点：抽样重建 HU 分布，检查空气峰与软组织峰的位置。
    sample = files[::SAMPLE_STRIDE]
    chunks = []
    for path in sample:
        ds = pydicom.dcmread(path)
        raw = ds.pixel_array.astype(np.int32)
        pad = getattr(ds, "PixelPaddingValue", None)
        mask = raw != int(pad) if pad is not None else np.ones(raw.shape, bool)
        chunks.append(raw[mask] * slope + intercept)
    hu = np.concatenate(chunks)
    lo, hi = float(hu.min()), float(hu.max())
    notes.append(f"抽样 {len(sample)} 层，去 padding 后值域 {lo:.0f}..{hi:.0f} HU")
    if not (HU_VALUE_RANGE[0] <= lo and hi <= HU_VALUE_RANGE[1]):
        return False, notes + [f"拒绝：值域超出合理 CT HU 区间 {HU_VALUE_RANGE}"]

    counts, edges = np.histogram(hu, bins=np.arange(-1200, 1600, 10))
    centers = (edges[:-1] + edges[1:]) / 2.0
    air = float(centers[np.argmax(np.where(centers < -500, counts, 0))])
    soft = float(centers[np.argmax(np.where((centers > -300) & (centers < 300), counts, 0))])
    notes.append(f"空气峰 {air:.0f} HU，软组织峰 {soft:.0f} HU")
    if not AIR_PEAK_RANGE[0] <= air <= AIR_PEAK_RANGE[1]:
        return False, notes + [f"拒绝：空气峰不在 {AIR_PEAK_RANGE}"]
    if not SOFT_PEAK_RANGE[0] <= soft <= SOFT_PEAK_RANGE[1]:
        return False, notes + [f"拒绝：软组织峰不在 {SOFT_PEAK_RANGE}"]
    return True, notes


def _is_group_length(tag):
    """retired 的 Group Length 元素：(gggg,0000)，group > 6。"""
    return tag.element == 0 and tag.group > 6


def assert_only_expected_diff(src_ds, dst_ds, path):
    """断言副本相对源只有预期差异，其余任何改动都视为副本被污染。"""
    ka = {e.tag: str(e.value) for e in src_ds}
    kb = {e.tag: str(e.value) for e in dst_ds}
    added = set(kb) - set(ka)
    removed = set(ka) - set(kb)
    changed = {t for t in set(ka) & set(kb) if ka[t] != kb[t]}

    if added != {RESCALE_TYPE_TAG}:
        raise RuntimeError(f"{path}: 新增 tag 不止 RescaleType，实际为 {sorted(added)}")
    unexpected_removed = {t for t in removed if not _is_group_length(t)}
    if unexpected_removed:
        raise RuntimeError(f"{path}: 非 Group Length 的 tag 被删除：{sorted(unexpected_removed)}")
    if changed:
        raise RuntimeError(f"{path}: 已有 tag 的值被改动：{sorted(changed)}")
    return len(removed)


def write_copy(files, src_root, dst_root):
    """写出副本：仅新增 RescaleType=HU，并逐文件校验 PixelData 未被改动。"""
    written = 0
    grouplen_dropped = 0
    for path in files:
        src_ds = pydicom.dcmread(path)      # 保留一份未修改的源，用于写出后做差异比对
        ds = pydicom.dcmread(path)          # 读入但**不访问** pixel_array，避免解压重编码
        original_bytes = ds.PixelData        # 原始像素 bytes，用于写出后校验
        ds.RescaleType = "HU"

        rel = os.path.relpath(path, src_root)
        out = os.path.join(dst_root, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        ds.save_as(out)

        # 回读校验：像素必须逐字节相同，且 ImageType 未被动过。
        back = pydicom.dcmread(out)
        if back.PixelData != original_bytes:
            raise RuntimeError(f"PixelData 在写出后发生变化：{out}")
        if tuple(back.ImageType) != tuple(ds.ImageType):
            raise RuntimeError(f"ImageType 被意外修改：{out}")
        if back.RescaleType != "HU":
            raise RuntimeError(f"RescaleType 未正确写入：{out}")
        # 全 tag 差异面校验：除新增 RescaleType 与丢弃 Group Length 外不得有任何改动。
        grouplen_dropped += assert_only_expected_diff(src_ds, back, out)
        written += 1
    return written, grouplen_dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="肺癌", help="源 DICOM 目录（只读，不会被修改）")
    ap.add_argument("--dst", required=True, help="输出目录，必须与 --src 不同")
    ap.add_argument("--verify-only", action="store_true", help="只跑验证，不写任何文件")
    a = ap.parse_args()

    src = os.path.abspath(a.src)
    dst = os.path.abspath(a.dst)
    if not os.path.isdir(src):
        sys.exit(f"源目录不存在：{src}")
    # 防止误把源目录当输出——源是神圣路径，绝不可被覆写。
    if dst == src or dst.startswith(src + os.sep):
        sys.exit("拒绝：输出目录不得等于或位于源目录内")

    files = _iter_files(src)
    if not files:
        sys.exit(f"源目录内没有文件：{src}")
    print(f"源：{src}（{len(files)} 个文件）")

    ok, notes = verify_is_hu(files)
    for line in notes:
        print("  " + line)
    if not ok:
        sys.exit("验证未通过，未写出任何文件。")
    print("验证通过：该序列数值确为标准 HU。")

    if a.verify_only:
        print("--verify-only，未写出文件。")
        return

    n, dropped = write_copy(files, src, dst)
    print(f"已写出 {n} 个文件到：{dst}")
    print("差异面已逐文件断言：只新增 RescaleType=HU，"
          f"并丢弃 {dropped} 个 retired Group Length 元素（pydicom 依 PS3.5 §7.2）。")
    print("PixelData 逐字节一致，ImageType 未改动，其余 tag 无新增/删除/改值。")


if __name__ == "__main__":
    main()
