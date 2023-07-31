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
)

logger = logging.getLogger("brainweb_dl")

SCIPY_AVAILABLE = True
try:
    import scipy as sp
except ImportError:
    SCIPY_AVAILABLE = False


def get_mri(
    sub_id: int,
    contrast: Literal["T1", "T2", "T2*"],
    shape: tuple[int, int, int] = None,
    rng: int | np.random.Generator = None,
    brainweb_dir: os.PathLike = None,
    res: int = 1,
    noise: int = 0,
    field_value: int = 0,
    force: bool = False,
    extension: Literal["nii", "nii.gz", "npy", None] = ".nii.gz",
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
        if contrast in ["T1", "T2", "PD"]:
            filename = get_brainweb1(
                contrast,
                res=res,
                noise=noise,
                field_value=field_value,
                brainweb_dir=brainweb_dir,
                force=force,
            )

            data = nib.load(filename)
            data = data.get_fdata()

        elif contrast == "T2*":
            logger.warning(
                "Brainweb 1 does not have T2* data. The values are going to be empirical."
            )
            filename = get_brainweb1_seg(
                "fuzzy",
                force=force,
                brainweb_dir=brainweb_dir,
            )

            data = _apply_contrast(filename, 1, contrast, rng)
        elif contrast in ["fuzzy", "crisp"]:
            filename = get_brainweb1_seg(
                contrast, force=force, brainweb_dir=brainweb_dir
            )
            data = nib.load(filename)
            data = np.asanyarray(data.dataobj, dtype=np.uint16)
            if contrast == "fuzzy":
                data = data.astype(np.float32) / 4095
        else:
            raise ValueError(f"Unknown contrast {contrast}")
    else:
        if contrast == "T1":
            filename = get_brainweb20_T1(sub_id)
            data = nib.load(filename).get_fdata()
        else:
            filename = get_brainweb20(sub_id, segmentation="fuzzy")
            data = _apply_contrast(filename, 20, contrast, rng)

    if shape is not None and shape != data.shape:
        if isinstance(shape, float):
            zoom = shape
            zoom = (zoom,) * 3
        elif -1 in shape:
            if np.prod(data.shape) <= 0:
                raise ValueError(
                    "The zoom factor should only have two -1 in its definition"
                    "(ex. `(-1,-1, 64)` )."
                )
            ref_ax = [i for i, v in enumerate(shape) if v > 0][0]
            zoom = shape[ref_ax] / data.shape[ref_ax]
            zoom = (zoom,) * 3
        else:
            zoom = np.array(shape) / np.array(data.shape)

        if contrast == "fuzzy":
            # Don't rescale the tissue dimension.
            zoom = (*zoom, 1)
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
    rng: int | np.random.Generator,
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
    data = nib.load(file_fuzzy).get_fdata(dtype=np.float32)
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
            contrast_mean[tlabel], contrast_std[tlabel] / 5, np.sum(mask)
        )
    return ret_data
