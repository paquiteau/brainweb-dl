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
        if contrast != "T2*":
            filename = get_brainweb1(
                contrast,
                res=res,
                noise=noise,
                field_value=field_value,
                brainweb_dir=brainweb_dir,
                force=force,
            )
            data = nib.load(filename).get_fdata()
            return data

        logger.warning(
            "Brainweb 1 does not have T2* data. The values are going to be empirical."
        )
        filename = get_brainweb1_seg(
            "fuzzy",
            force=force,
            brainweb_dir=brainweb_dir,
        )

        return _apply_contrast(filename, 1, contrast, rng)
    if contrast == "T1":
        filename = get_brainweb20_T1(sub_id)
        data = nib.load(filename).get_fdata()
    else:
        filename = get_brainweb20(sub_id, segmentation="fuzzy")
        data = _apply_contrast(filename, 20, contrast, rng)

    if shape is not None and shape != data.shape:
        # rescale the data
        data_rescaled = sp.ndimage.zoom(data, np.array(shape) / np.array(data.shape))
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
