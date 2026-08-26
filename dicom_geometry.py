"""Pure DICOM patient-space geometry checks used by the product load path.

The workstation stores classic single-frame slices as a NumPy ``(slice, row,
column)`` array.  That array is only an anatomical axial volume when the DICOM
orientation and positions prove it; array shape alone is not evidence.
"""

import hashlib
import json
from dataclasses import dataclass

import numpy as np

DIRECTION_NORM_ATOL = 1e-4
ORTHOGONALITY_ATOL = 1e-6
ORIENTATION_CONSISTENCY_ATOL = 1e-4
CANONICAL_ORIENTATION_ATOL = 1e-4
SPACING_RTOL = 1e-3
SPACING_ATOL_MM = 1e-4
POSITION_ATOL_MM = 1e-3
CANONICAL_AXIAL_IOP = np.array((1.0, 0.0, 0.0, 0.0, 1.0, 0.0))


@dataclass(frozen=True)
class SeriesGeometry:
    """Independently usable capabilities derived from one DICOM series."""

    hu_calibrated: bool
    canonical_orientation: bool
    inplane_spacing_valid: bool
    uniform_z_geometry_valid: bool
    sort_indices: tuple[int, ...] | None
    projected_positions_mm: tuple[float, ...] | None
    slice_spacing_mm: float | None


def _finite_vector(dataset, name, length):
    try:
        value = np.asarray(getattr(dataset, name), dtype=float)
    except (AttributeError, TypeError, ValueError):
        return None
    if value.shape != (length,) or not np.all(np.isfinite(value)):
        return None
    return value


def _finite_scalar(dataset, name):
    try:
        value = float(getattr(dataset, name))
    except (AttributeError, TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _normalized_terms(dataset, name):
    """Return upper-case DICOM text values without depending on pydicom types."""
    try:
        raw = getattr(dataset, name)
    except AttributeError:
        return ()
    if raw is None:
        return ()
    if isinstance(raw, str):
        values = raw.split("\\")
    else:
        try:
            values = list(raw)
        except TypeError:
            values = [raw]
    return tuple(term for term in (str(value).strip().upper() for value in values) if term)


def _slice_has_standard_hu(dataset):
    """Prove that one supported classic CT slice maps stored values to standard HU.

    DICOM PS3.3 C.8.2 permits an omitted Rescale Type to imply HU only for an
    ORIGINAL, non-LOCALIZER, non-multi-energy CT image.  An explicit HU type is
    also accepted.  This workstation does not implement the multi-energy
    contract, so any value other than absent/NO fails closed even though some
    multi-energy images can themselves represent HU.
    """
    slope = _finite_scalar(dataset, "RescaleSlope")
    intercept = _finite_scalar(dataset, "RescaleIntercept")
    if slope is None or slope == 0 or intercept is None:
        return False

    multi_energy = _normalized_terms(dataset, "MultienergyCTAcquisition")
    if multi_energy not in ((), ("NO",)):
        return False

    rescale_type = _normalized_terms(dataset, "RescaleType")
    if rescale_type:
        return rescale_type == ("HU",)

    image_type = _normalized_terms(dataset, "ImageType")
    return bool(
        len(image_type) >= 3
        and image_type[0] == "ORIGINAL"
        and image_type[2] != "LOCALIZER"
        and "LOCALIZER" not in image_type
    )


def _is_orthonormal_iop(iop):
    """Validate one slice's two DICOM direction cosines with explicit tolerances."""
    row, column = iop[:3], iop[3:]
    normal = np.cross(row, column)
    return bool(
        np.isclose(np.linalg.norm(row), 1.0, rtol=0.0, atol=DIRECTION_NORM_ATOL)
        and np.isclose(np.linalg.norm(column), 1.0,
                       rtol=0.0, atol=DIRECTION_NORM_ATOL)
        and np.isclose(float(np.dot(row, column)), 0.0,
                       rtol=0.0, atol=ORTHOGONALITY_ATOL)
        and np.isclose(np.linalg.norm(normal), 1.0,
                       rtol=0.0, atol=DIRECTION_NORM_ATOL)
    )


def patient_coordinate(ipp, iop, pixel_spacing, row, column):
    """Map array ``(row, column)`` to DICOM patient LPS millimetres."""
    ipp = np.asarray(ipp, dtype=float)
    iop = np.asarray(iop, dtype=float)
    spacing = np.asarray(pixel_spacing, dtype=float)
    if ipp.shape != (3,) or iop.shape != (6,) or spacing.shape != (2,):
        raise ValueError("IPP/IOP/PixelSpacing must have lengths 3/6/2")
    if not np.all(np.isfinite(np.concatenate((ipp, iop, spacing)))) or np.any(spacing <= 0):
        raise ValueError("DICOM geometry values must be finite and spacing must be positive")
    return ipp + float(column) * spacing[1] * iop[:3] + float(row) * spacing[0] * iop[3:]


def _lps_direction_label(vector):
    axis = int(np.argmax(np.abs(vector)))
    value = float(vector[axis])
    return (("L", "R"), ("P", "A"), ("S", "I"))[axis][value < 0]


def voxel_plane_edge_labels(iop):
    """Derive displayed source-plane edges from IOP in patient LPS."""
    iop = np.asarray(iop, dtype=float)
    if iop.shape != (6,) or not np.all(np.isfinite(iop)):
        raise ValueError("IOP must contain six finite values")
    right = _lps_direction_label(iop[:3])
    bottom = _lps_direction_label(iop[3:])
    opposite = {"L": "R", "R": "L", "P": "A", "A": "P", "S": "I", "I": "S"}
    return {"top": opposite[bottom], "bottom": bottom,
            "left": opposite[right], "right": right}


def _normalized_floats(values):
    return [format(float(value), ".12g") for value in values]


def series_fingerprint(datasets, volume_shape):
    """Bind persisted slice-indexed data to deterministic geometry and order.

    An empty string means the current datasets cannot provide enough stable
    identity/geometry fields for safe automatic restoration.
    """
    if not datasets or len(volume_shape) != 3 or int(volume_shape[0]) != len(datasets):
        return ""
    slices = []
    seen_sops = set()
    for ds in datasets:
        sop = str(getattr(ds, "SOPInstanceUID", "") or "")
        iop = _finite_vector(ds, "ImageOrientationPatient", 6)
        ipp = _finite_vector(ds, "ImagePositionPatient", 3)
        pixel_spacing = _finite_vector(ds, "PixelSpacing", 2)
        if not sop or sop in seen_sops or iop is None or ipp is None or pixel_spacing is None:
            return ""
        seen_sops.add(sop)
        normal = np.cross(iop[:3], iop[3:])
        projected = float(np.dot(ipp, normal))
        if not np.isfinite(projected):
            return ""
        slices.append({
            "sop_instance_uid": sop,
            "iop": _normalized_floats(iop),
            "ipp": _normalized_floats(ipp),
            "projected_position_mm": format(projected, ".12g"),
            "pixel_spacing_mm": _normalized_floats(pixel_spacing),
        })
    payload = {
        "schema": "dicom-geometry-order-v1",
        "series_instance_uid": str(getattr(datasets[0], "SeriesInstanceUID", "") or ""),
        "volume_shape": [int(v) for v in volume_shape],
        "ordered_slices": slices,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def analyze_series(datasets) -> SeriesGeometry:
    """Return the patient-space capabilities proven by ``datasets``.

    Sorting uses ``dot(ImagePositionPatient, slice_normal)`` rather than the
    third patient coordinate, so a valid sagittal/coronal stack remains ordered
    correctly without being misrepresented as canonical axial.
    """
    if not datasets:
        return SeriesGeometry(False, False, False, False, None, None, None)

    # Series 不允许混合单位：每一张 decode 后保留的 slice 都必须独立证明为标准 HU。
    hu_calibrated = all(_slice_has_standard_hu(ds) for ds in datasets)

    pixel_spacings = [_finite_vector(ds, "PixelSpacing", 2) for ds in datasets]
    inplane_spacing_valid = all(
        spacing is not None and np.all(spacing > 0) for spacing in pixel_spacings
    )
    if inplane_spacing_valid:
        ref_spacing = pixel_spacings[0]
        inplane_spacing_valid = all(
            np.allclose(spacing, ref_spacing, rtol=SPACING_RTOL, atol=SPACING_ATOL_MM)
            for spacing in pixel_spacings[1:]
        )

    orientations = [_finite_vector(ds, "ImageOrientationPatient", 6) for ds in datasets]
    # 每片先独立证明 direction cosines 有限、单位长度且正交，再检查整列方向一致。
    # 只验证首片会放过“首片合法、后片轻微非正交”的畸形 series。
    orientation_valid = all(
        iop is not None and _is_orthonormal_iop(iop) for iop in orientations
    )
    normal = None
    if orientation_valid:
        ref_iop = orientations[0]
        row, column = ref_iop[:3], ref_iop[3:]
        normal = np.cross(row, column)
        orientation_valid = all(
            np.allclose(iop, ref_iop, rtol=0.0, atol=ORIENTATION_CONSISTENCY_ATOL)
            for iop in orientations[1:]
        )

    canonical_orientation = bool(
        orientation_valid
        and np.allclose(orientations[0], CANONICAL_AXIAL_IOP,
                        rtol=0.0, atol=CANONICAL_ORIENTATION_ATOL)
    )

    positions = [_finite_vector(ds, "ImagePositionPatient", 3) for ds in datasets]
    positions_valid = orientation_valid and all(position is not None for position in positions)
    projected = None
    order = None
    spacing = None
    uniform_z_geometry_valid = False
    if positions_valid:
        projected_array = np.array([float(np.dot(position, normal)) for position in positions])
        order_array = np.argsort(projected_array, kind="stable")
        sorted_positions = projected_array[order_array]
        projected = tuple(float(v) for v in sorted_positions)
        order = tuple(int(v) for v in order_array)
        if len(sorted_positions) >= 2:
            gaps = np.diff(sorted_positions)
            if np.all(gaps > SPACING_ATOL_MM):
                spacing = float(np.median(gaps))
                sorted_origins = np.asarray(positions)[order_array]
                deltas = np.diff(sorted_origins, axis=0)
                inplane_residual = deltas - gaps[:, None] * normal[None, :]
                uniform_z_geometry_valid = bool(
                    np.allclose(gaps, spacing, rtol=SPACING_RTOL, atol=SPACING_ATOL_MM)
                    and np.all(np.linalg.norm(inplane_residual, axis=1) <= POSITION_ATOL_MM)
                )

    return SeriesGeometry(
        hu_calibrated=hu_calibrated,
        canonical_orientation=canonical_orientation,
        inplane_spacing_valid=inplane_spacing_valid,
        uniform_z_geometry_valid=uniform_z_geometry_valid,
        sort_indices=order,
        projected_positions_mm=projected,
        slice_spacing_mm=spacing,
    )
