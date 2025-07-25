"""Collection of function to create realistic MRI data from brainweb segmentations."""

from __future__ import annotations

import logging
import os

import numpy as np
import scipy as sp
from nibabel import nifti1 as nifti
from numpy.typing import NDArray

from ._brainweb import (
    BIG_RES_MM,
    STD_RES_MM,
    BrainWebDirType,
    BrainWebTissueMap,
    Contrast,
    Segmentation,
    _load_tissue_map,
    get_brainweb1,
    get_brainweb1_seg,
    get_brainweb20,
    get_brainweb20_T1,
)

logger = logging.getLogger("brainweb_dl")
GenericPath = os.PathLike[str] | str


def _get_mri_sub0(
    contrast: Contrast | Segmentation,
    brainweb_dir: BrainWebDirType = None,
    res: int = 1,
    noise: int = 0,
    field_value: int = 0,
    force: bool = False,
    tissue_map: GenericPath = BrainWebTissueMap.v1,
    rng: int | np.random.Generator | None = None,
) -> tuple[NDArray, NDArray]:
    if contrast in [Contrast.T1, Contrast.T2, Contrast.PD]:
        filename = get_brainweb1(
            Contrast(contrast),
            res=res,
            noise=noise,
            field_value=field_value,
            brainweb_dir=brainweb_dir,
            force=force,
        )

        nft = nifti.Nifti1Image.from_filename(filename)
        affine = np.asarray(nft.affine)
        return nft.get_fdata(), affine

    if contrast is Contrast.T2S:
        logger.warning(
            "Brainweb 1 does not have T2s data. The provided values are empirical."
        )
        filename = get_brainweb1_seg(
            Segmentation.FUZZY,
            force=force,
            brainweb_dir=brainweb_dir,
        )
        nft = nifti.Nifti1Image.from_filename(filename)
        affine = np.asarray(nft.affine)
        return _apply_contrast(filename, tissue_map, contrast, rng), affine
    # Segmenation data.
    filename = get_brainweb1_seg(
        Segmentation(contrast), force=force, brainweb_dir=brainweb_dir
    )
    nft = nifti.Nifti1Image.from_filename(filename)
    data = np.asanyarray(nft.dataobj, dtype=np.uint16)
    affine = np.asarray(nft.affine)
    if contrast is Segmentation.FUZZY:
        data = data.astype(np.float32) / 4095.0  # type: ignore
    return data, affine


def _get_mri_sub20(
    contrast: Contrast | Segmentation,
    sub_id: int | str,
    brainweb_dir: BrainWebDirType = None,
    force: bool = False,
    tissue_map: GenericPath = BrainWebTissueMap.v2,
    rng: int | np.random.Generator | None = None,
) -> tuple[NDArray, NDArray]:
    if contrast is Contrast.T1:
        filename = get_brainweb20_T1(sub_id, brainweb_dir=brainweb_dir, force=force)
        nft = nifti.Nifti1Image.from_filename(filename)
        data, affine = np.asarray(nft.get_fdata()), np.asarray(nft.affine)
    elif contrast in Segmentation:
        filename = get_brainweb20(
            sub_id, segmentation=Segmentation(contrast), force=force
        )
        nft = nifti.Nifti1Image.from_filename(filename)
        data = np.asanyarray(nft.dataobj, dtype=np.uint16)
        affine = np.asarray(nft.affine)
        if contrast is Segmentation.FUZZY:
            data = data.astype(np.float32) / 4095
    else:
        filename = get_brainweb20(sub_id, segmentation=Segmentation.FUZZY, force=force)
        tissue_map = tissue_map or BrainWebTissueMap.v2
        data = _apply_contrast(filename, tissue_map, Contrast(contrast), rng)
        affine = np.asarray(nifti.Nifti1Image.from_filename(filename).affine)
    return (data, affine)


def get_mri(
    sub_id: int,
    contrast: Contrast | Segmentation = Contrast.T1,
    bbox: tuple[float | None, ...] | None = None,
    shape: tuple[int, int, int] | None = None,
    brainweb_dir: BrainWebDirType = None,
    output_res: float | tuple[float, float, float] | None = None,
    download_res: int = 1,
    noise: int = 0,
    field_value: int = 0,
    force: bool = False,
    with_affine: bool = False,
    tissue_map: GenericPath | None = None,
    rng: int | np.random.Generator | None = None,
) -> tuple[NDArray, NDArray] | NDArray:
    """Get MRI data from a brainweb fuzzy segmentation.

    Parameters
    ----------
    sub_id : int
        Subject ID.
    contrast : {"T1", "T2", "T2*"}
        Contrast to use.
    bbox : tuple[float, float, float, float, float, float], optional
        Bounding box of the data, specified as [xmin, xmax, ymin, ymax, zmin, zmax]
        with values in [0, 1].
        The data is cropped to the bounding box.
    output_res: float | tuple[float, float, float] optional, default None
        Resolution of the output data, the data will be rescale to the given resolution.
    rng : int | np.random.Generator, optional
        Random number generator.
    dir : str, optional
        Brainweb download directory.
    download_res : int, optional
        Resolution of the data, only use for subject 0.
    noise : int, optional
        Noise level of the data., only use for subject 0.
    field_value : int, optional, only use for subject 0.
        Field value of the data.
    with_affine : bool, optional
        Return the affine matrix with the data.

    kwargs : dict, optional
        Additional arguments to pass to the brainweb functions.

    Returns
    -------
    np.ndarray
        MRI data.
    """
    try:
        contrast = Contrast(contrast)
    except ValueError:
        try:
            contrast = Segmentation(contrast)
        except ValueError as e:
            raise ValueError(f"Unknown contrast {contrast}") from e
    logger.debug(f"Get MRI data for subject {sub_id} and contrast {contrast}")
    if sub_id == 0 or sub_id == "0":
        data, affine = _get_mri_sub0(
            contrast,
            brainweb_dir=brainweb_dir,
            res=download_res,
            noise=noise,
            field_value=field_value,
            force=force,
            tissue_map=tissue_map or BrainWebTissueMap.v1,
            rng=rng,
        )
    else:
        data, affine = _get_mri_sub20(
            contrast,
            sub_id,
            brainweb_dir=brainweb_dir,
            force=force,
            tissue_map=tissue_map or BrainWebTissueMap.v2,
        )

    if bbox is not None:
        logger.debug(f"Apply bounding box {bbox} to the data")
        data = _crop_data(data, bbox)
        # FIXME: changing the bbox updates the affine matrix !

    zoom: tuple[float | NDArray, ...] | None = None
    if shape is not None and output_res is None:  # rescale the data with shape
        if isinstance(shape, float):
            zoom = shape
            zoom = (zoom,) * 3
        elif -1 in shape:
            if np.prod(shape) <= 0:
                raise ValueError(
                    "The zoom factor must have two implicit dimension (-1)"
                    "in its definition (ex. `(-1,-1, 64)` )."
                )
            ref_ax = [i for i, v in enumerate(shape) if v > 0][0]
            zoom = (shape[ref_ax] / data.shape[ref_ax],) * 3
        else:
            zoom = tuple(np.array(shape) / np.array(data.shape[:3]))

    elif output_res is not None and shape is None:  # rescale the data with res
        if isinstance(output_res, float):
            output_res_ = (output_res,) * 3
        else:
            output_res_ = output_res
        base_res = BIG_RES_MM  #
        if sub_id == 0 or sub_id != 0 and contrast == Contrast.T1:
            base_res = STD_RES_MM
        zoom = tuple(np.array(base_res) / np.array(output_res_))

    elif output_res is not None and shape is not None:
        raise ValueError("output_res and shape cannot be set at the same time")

    if zoom is not None:
        logger.debug(f"Rescale the data with zoom {zoom}")
        if contrast is Segmentation.FUZZY:
            # Don't rescale the tissue dimension.
            zoom = (*zoom, 1.0)

        data_rescaled = sp.ndimage.zoom(data, zoom=zoom)
        # clip the data to the original range.
        data_rescaled = np.clip(data_rescaled, data.min(), data.max())
        logger.debug(f"Data shape after rescaling: {data_rescaled.shape}")
        data = data_rescaled
    # FIXME: zoom changes the affine matrix.
    else:
        logger.debug(f"Return data with shape {data.shape}")
    if with_affine:
        return data, affine
    return data


def _crop_data(data: np.ndarray, bbox: tuple[float | None, ...]) -> np.ndarray:
    """Crop the 3D data to the bounding box bbox."""
    slicer = [slice(None)] * len(data.shape)
    if len(data.shape) == 4:
        # add a fourth dimension for the segmentation.
        bbox = (*bbox, None, None)
    for i, s in enumerate(data.shape):
        slicer[i] = slice(
            int(bbox[2 * i] * s) if bbox[2 * i] is not None else 0,  # type: ignore
            int(bbox[2 * i + 1] * s) if bbox[2 * i + 1] is not None else s,  # type: ignore
        )
    return data[tuple(slicer)]


def _apply_contrast(
    file_fuzzy: GenericPath,
    tissue_map: GenericPath,
    contrast: Contrast,
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
    rng_ = np.random.default_rng(rng)

    tissues = _load_tissue_map(tissue_map)
    data = nifti.Nifti1Image.from_filename(file_fuzzy).get_fdata(dtype=np.float32)
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
        ret_data[mask] += data[mask, tlabel] * rng_.normal(
            contrast_mean[tlabel], contrast_std[tlabel] / 5, np.sum(mask, dtype=int)
        )
    return ret_data
