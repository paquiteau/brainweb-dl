"""Collection of function to create realistic MRI data from brainweb segmentations."""
from __future__ import annotations
import logging
import os
from typing import Literal

import nibabel as nib
import numpy as np
from ._brainweb import (
    _load_tissue_map,
    get_brainweb1,
    get_brainweb1_seg,
    get_brainweb20,
    get_brainweb20_T1,
    Contrast,
    Segmentation,
    BrainWebTissueMap,
    BrainWebDirType,
)

logger = logging.getLogger("brainweb_dl")

SCIPY_AVAILABLE = True
try:
    import scipy as sp
except ImportError:
    SCIPY_AVAILABLE = False


def _get_mri_sub0(
    contrast: Contrast | Segmentation,
    brainweb_dir: BrainWebDirType = None,
    res: int = 1,
    noise: int = 0,
    field_value: int = 0,
    force: bool = False,
    tissue_map: os.PathLike = BrainWebTissueMap.v1.value,
    rng: int | np.random.Generator | None = None,
) -> np.ndarray:
    if contrast in [Contrast.T1, Contrast.T2, Contrast.PD]:
        filename = get_brainweb1(
            contrast,
            res=res,
            noise=noise,
            field_value=field_value,
            brainweb_dir=brainweb_dir,
            force=force,
        )

        data = nib.Nifti2Image.from_filename(filename).get_fdata()

    elif Contrast(contrast) is Contrast.T2s:
        logger.warning(
            "Brainweb 1 does not have T2s data. The provided values are empirical."
        )
        filename = get_brainweb1_seg(
            Segmentation.FUZZY,
            force=force,
            brainweb_dir=brainweb_dir,
        )

        data = _apply_contrast(filename, tissue_map, contrast, rng)
    elif contrast in Segmentation:
        filename = get_brainweb1_seg(
            Segmentation(contrast), force=force, brainweb_dir=brainweb_dir
        )
        data_ = nib.Nifti2Image.from_filename(filename)
        data = np.asanyarray(data_.dataobj, dtype=np.uint16)
        if contrast == Segmentation.FUZZY:
            data = data.astype(np.float32) / 4095
    else:
        raise ValueError(f"Unknown contrast {contrast}")

    return data


def _get_mri_sub20(
    contrast: Contrast | Segmentation,
    sub_id: int,
    brainweb_dir: BrainWebDirType = None,
    force: bool = False,
    tissue_map: os.PathLike = BrainWebTissueMap.v2.value,
    rng: int | np.random.Generator | None = None,
) -> np.ndarray:
    if contrast == Contrast.T1:
        filename = get_brainweb20_T1(sub_id, brainweb_dir=brainweb_dir, force=force)
        data = nib.Nifti2Image.from_filename(filename).get_fdata()
    elif contrast in Segmentation:
        filename = get_brainweb20(sub_id, segmentation=Segmentation(contrast))
        data_ = nib.Nifti2Image.from_filename(filename).dataobj
        data = np.asanyarray(data_, dtype=np.uint16)
        if Segmentation(contrast) == Segmentation.FUZZY:
            data = data.astype(np.float32) / 4095
    else:
        filename = get_brainweb20(sub_id, segmentation=Segmentation.FUZZY)
        tissue_map = tissue_map or BrainWebTissueMap.v2.value
        data = _apply_contrast(filename, tissue_map, Contrast(contrast), rng)

    return data


def get_mri(
    sub_id: int,
    contrast: Contrast | Segmentation = Contrast.T1,
    shape: tuple[int, int, int] | None = None,
    brainweb_dir: BrainWebDirType = None,
    res: int = 1,
    noise: int = 0,
    field_value: int = 0,
    force: bool = False,
    extension: Literal[".nii", ".nii.gz", ".npy", None] = ".nii.gz",
    tissue_map: os.PathLike | None = None,
    rng: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Get MRI data from a brainweb fuzzy segmentation.

    Parameters
    ----------
    sub_id : int
        Subject ID.
    contrast : {"T1", "T2", "T2*"}
        Contrast to use.
    shape : tuple[int, int, int], optional
        Shape of the MRI data. If None, the original shape is used.
    rng : int | np.random.Generator, optional
        Random number generator.
    dir : str, optional
        Brainweb download directory.
    res : int, optional
        Resolution of the data, only use for subject 0.
    noise : int, optional
        Noise level of the data., only use for subject 0.
    field_value : int, optional, only use for subject 0.
        Field value of the data.
    kwargs : dict, optional
        Additional arguments to pass to the brainweb functions.

    Returns
    -------
    np.ndarray
        MRI data.
    """
    if sub_id == 0:
        data = _get_mri_sub0(
            contrast,
            brainweb_dir=brainweb_dir,
            res=res,
            noise=noise,
            field_value=field_value,
            force=force,
            tissue_map=tissue_map or BrainWebTissueMap.v1.value,
            rng=rng,
        )
    else:
        data = _get_mri_sub20(
            contrast,
            sub_id,
            brainweb_dir=brainweb_dir,
            force=force,
            tissue_map=tissue_map or BrainWebTissueMap.v2.value,
        )

    zoom: tuple[float, ...]
    if shape is not None and shape != data.shape and SCIPY_AVAILABLE:
        if isinstance(shape, float):
            zoom = shape
            zoom = (zoom,) * 3
        elif -1 in shape:
            if np.prod(data.shape) <= 0:
                raise ValueError(
                    "The zoom factor must have two implicit dimension (-1)"
                    "in its defintion (ex. `(-1,-1, 64)` )."
                )
            ref_ax = [i for i, v in enumerate(shape) if v > 0][0]
            zoom = (shape[ref_ax] / data.shape[ref_ax],) * 3
        else:
            zoom = np.array(shape) / np.array(data.shape[:3])

        if Segmentation(contrast) == Segmentation.FUZZY:
            # Don't rescale the tissue dimension.
            zoom = (*zoom, 1.0)
        # rescale the data
        data_rescaled = sp.ndimage.zoom(data, zoom=zoom)
        # clip the data to the original range.
        data_rescaled = np.clip(data_rescaled, data.min(), data.max())
        return data_rescaled
    else:
        return data


def _apply_contrast(
    file_fuzzy: os.PathLike,
    tissue_map: os.PathLike,
    contrast: str,
    rng: int | np.random.Generator | None,
) -> np.ndarray:
    """Apply contrast to the data.

    Parameters
    ----------
    file_fuzzy : str
        Path to the fuzzy segmentation.
    tissue_map : str
        Path to the tissue map.
    rng : int | np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        MRI data with the applied contrast.
    """
    rng = np.random.default_rng(rng)

    tissues = _load_tissue_map(tissue_map)
    data = nib.Nifti2Image.from_filename(file_fuzzy).get_fdata(dtype=np.float32)
    data /= 4095  # Data was encode in 12 bits
    ret_data = np.zeros(data.shape[:-1], dtype=np.float32)
    contrast_mean = []
    contrast_std = []
    for t in tissues:
        contrast_mean.append(float(t[f"{contrast} (ms)"]))
        try:
            std_val = float(t[f"{contrast} Std (ms)"])
        except KeyError:
            std_val = 0
        contrast_std.append(std_val)

    for tlabel in range(1, len(tissues)):
        mask = data[..., tlabel] > 0
        ret_data[mask] += data[mask, tlabel] * rng.normal(
            contrast_mean[tlabel], contrast_std[tlabel] / 5, np.sum(mask, dtype=int)
        )
    return ret_data
