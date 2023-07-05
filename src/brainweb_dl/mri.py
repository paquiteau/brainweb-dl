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


    Returns
    -------
    np.ndarray
        MRI data.
    """
    if sub_id == 0:
        if contrast != "T2*":
            filename = get_brainweb1(
                contrast,
                res=1,
                noise=0,
                field_value=0,
                brainweb_dir=brainweb_dir,
            )
            data = nib.load(filename).get_fdata()
            return data

        logger.warning(
            "Brainweb 1 does not have T2* data. The values are going to be empirical."
        )
        filename = get_brainweb1("fuzzy", res=1, noise=0, field_value=0)

        return _apply_contrast(filename, 1, contrast, rng)
    if contrast != "T1":
        filename = get_brainweb20(sub_id, segmentation="fuzzy")
        return _apply_contrast(filename, 20, contrast, rng)
    filename = get_brainweb20_T1(sub_id)

    data = nib.load(filename).get_fdata()
    if shape != data.shape and SCIPY_AVAILABLE:
        # rescale the data
        data_rescaled = sp.ndimage.zoom(data, np.array(data.shape) / np.array(shape))
        return data_rescaled
    elif shape != data.shape and not SCIPY_AVAILABLE:
        raise RuntimeError("scipy is required to rescale the data.")
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
    contrast_mean = [int(t[f"{contrast} (ms)"]) for t in tissues]
    contrast_std = [int(t[f"{contrast} Std (ms)"]) for t in tissues]
    for tlabel in range(1, len(tissues)):
        mask = data[..., tlabel] > 0
        ret_data[mask] += data[mask, tlabel] * rng.normal(
            contrast_mean[tlabel], contrast_std[tlabel] / 5, np.sum(mask)
        )
    return ret_data
