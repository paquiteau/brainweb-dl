"""Collection of function to create realistic MRI data from brainweb segmentations."""
import logging
import os
from typing import Literal

import nibabel as nib
import numpy as np
from _brainweb import _load_tissue_map, get_brainweb1, get_brainweb20

logger = logging.getLogger("brainweb_dl")


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

    filename = get_brainweb20(sub_id, segmentation="fuzzy")
    return _apply_contrast(filename, 20, contrast, rng)


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
    t2s = np.array([t[f"{contrast} (ms)"] for t in tissues])
    t2s_std = np.array([t[f"{contrast} Std (ms)"] for t in tissues])

    for tlabel in len(tissues[1:]):
        mask = data[..., tlabel] > 0
        ret_data[mask] = data[..., tlabel] * rng.normal(
            t2s[tlabel], t2s_std[tlabel], np.sum(mask)
        )
    return ret_data
