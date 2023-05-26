"""
Utils for dowloading brainweb dataset.

The brainweb phantoms consists of a set of realistic simulated brain MR images
that can be used for validation of segmentation, registration or reconstruction
algorithms.
There is two datasets available:
 - Brainweb1": Consist of 2 phantoms
   -
There is a single Anatomical T1, T2 and PD volume. The volumes are 181x217x181
at 1mm isotropic resolution.
References
----------
- Brainweb: https://brainweb.bic.mni.mcgill.ca/brainweb/
- Original Python interface: https://github.com/casperdcl/brainweb/blob/master/brainweb/utils.py
"""
import logging
import os
import csv
from importlib.resources import files
from pathlib import Path
from typing import Literal
import io
import gzip

import nibabel as nib
import numpy as np
import requests
from numpy.typing import DTypeLike
from tqdm.auto import tqdm

logger = logging.getLogger("brainweb")

# +fmt: off
SUB_ID = [4, 5, 6, 18, 20, 38, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
# +fmt: on
BRAINWEB_VALUES = {1: "brainweb1_tissues.csv", 20: "brainweb20_tissues.csv"}

ALLOWED_NOISE_LEVEL = [0, 1, 3, 5, 7, 9]
ALLOWED_RF = [0, 20, 40]
ALLOWED_RES = [1, 3, 5, 7, 9]


BASE_URL = "http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1/"

BIG_RES = (362, 434, 362)
STD_RES = (181, 217, 181)

AFFINE = np.array(
    [
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ]
)

AFFINE = np.eye(4)

def download_brainweb20_multiple(
    subject: int | list | Literal["all"],
    dir: os.PathLike = ".brainweb",
    force: bool = False,
    segmentation: Literal["crisp", "fuzzy"] = "crisp",
) -> list[os.PathLike]:
    """Download sample or all brainweb subjects.

    Parameters
    ----------
    subject : int | list | Literal["all"]
        subject id or list of subject id to download.
        If "all", download all subjects.
    dir : os.PathLike
        directory to download the data.
    force : bool
        force download even if the file already exists.

    Returns
    -------
    list[os.PathLike]
        list of downloaded files.
    """
    if subject == "all":
        subject = SUB_ID
    elif isinstance(subject, int):
        subject = [SUB_ID[subject]]
    elif not isinstance(subject, list):
        subject = [SUB_ID[s] for s in subject]
        raise ValueError("subject must be int, list or 'all'")
    f = []
    for s in tqdm(subject):
        filename = download_brainweb20(s, dir, force, segmentation)
        f.append(filename)
    return f


def download_brainweb20(
    s: int,
    dir: os.PathLike,
    force: bool = False,
    segmentation: Literal["crisp", "fuzzy"] = "crisp",
) -> os.PathLike:
    """Download one subject of brainweb dataset.

    Parameters
    ----------
    s : int
        subject id.
    dir : os.PathLike
        directory to download the data.
    force : bool
        force download even if the file already exists.
    segmentation: "crisp" | "fuzzy"
        segmentation type.

    Returns
    -------
    os.PathLike
        Path to downloaded file.
    """
    dir = Path(dir)
    path = dir / f"brainweb_s{s:02d}_{segmentation}.nii.gz"
    if path.exists() and not force:
        return path

    if segmentation == "crisp":
        download_command = f"subject{s:02d}_{segmentation}"
        _request_get_brainweb(download_command, path, shape=BIG_RES, dtype=np.uint16)
    elif segmentation == "fuzzy":
        # Download all the fuzzy segmentation and create a 4D volume.
        path = Path(dir) / f"brainweb_s{s:02d}_fuzzy.nii.gz"
        # The case of fuzzy segmentation is a bit special.
        # We download all the fuzzy segmentation and create a 4D volume.
        # The 4th dimension is the segmentation type.
        if path.exists() and not force:
            return path
        tissue_map = _load_tissue_map(20)
        data = np.zeros((*BIG_RES, len(tissue_map)))
        for i, row in tqdm(
            enumerate(tissue_map),
            desc="Downloading fuzzy segmentation",
            position=0,
        ):
            name = f"subject{s:02d}_{row['ID']}"
            data[..., i] = _request_get_brainweb(
                name,
                dir / f"{name}.nii.gz",
                dtype=np.uint16,
                shape=BIG_RES,
                obj_mode=True,
            )
        nib.save(nib.Nifti1Image(data, affine=AFFINE), path)
        return path

    else:
        raise ValueError("segmentation must be 'crisp' or 'fuzzy'")
    return path


def download_brainweb1(
    type: Literal["T1", "T2", "PD", "crisp", "fuzzy"] = "T2",
    res: int = 1,
    noise: int = 0,
    field_value: int = 0,
    dir: os.PathLike = ".brainweb",
    force: bool = False,
) -> os.PathLike | np.ndarray:
    """Download the Brainweb1 phantom as a nifti file.

    Parameters
    ----------
    res : int
        Resolution of the phantom. Must be in {1, 3, 5, 7}
    noise : int
        Percent of noise level in the phantom. Must be in
        {0, 1, 3, 5, 7, 9}
    field_value : int
        RF field value in the phantom. Must be in {0, 20, 40}
    dir : os.PathLike
        directory to download the data.
    force : bool
        force download even if the file already exists.

    Returns
    -------
    os.PathLike
        Path to downloaded file.

    Notes
    -----
    This is a simple interface to the form available at:
    https://brainweb.bic.mni.mcgill.ca/brainweb/selection_normal.html
    """
    dir = Path(dir)

    if res not in ALLOWED_RES:
        raise ValueError(f"Resolution must be in {ALLOWED_RES}")
    if noise not in ALLOWED_NOISE_LEVEL:
        raise ValueError(f"Noise level must be in {ALLOWED_NOISE_LEVEL}")
    if field_value not in ALLOWED_RF:
        raise ValueError(f"RF field value must be in {ALLOWED_RF}")

    if type == "fuzzy":
        # The case of fuzzy segmentation is a bit special.
        # We download all the fuzzy segmentation and create a 4D volume.
        # The 4th dimension is the segmentation type.
        fname = f"phantom_{res:.1f}mm_normal_fuzzy.nii.gz"
        path = dir / fname
        if path.exists() and not force:
            return path
        tissue_map = _load_tissue_map(1)
        data = np.zeros((*STD_RES, len(tissue_map)))
        for i, row in tqdm(
            enumerate(tissue_map),
            desc="Downloading tissues",
            total=len(tissue_map),
            position=1,
            leave=False,
        ):
            name = f"phantom_{res:.1f}mm_normal_{row['ID']}"
            data[..., i] = _request_get_brainweb(
                name,
                path=dir / f"{name}.nii.gz",
                dtype=np.uint16,
                shape=STD_RES,
                obj_mode=True,
            )
        # Create the 4D volume.
        nib.save(nib.Nifti1Image(data, affine=AFFINE), path)
        return path
    elif type == "crisp":
        download_command = f"phantom_{res:.1f}mm_normal_crisp"
        fname = f"phantom_{res:.1f}mm_normal_crisp.nii"
    elif type in ["T1", "T2", "PD"]:
        # download of contrasted images
        download_command = f"{type}+ICBM+normal+{res}mm+pn{noise}+rf{field_value}"
        fname = f"{type}_ICBM_normal_{res}mm_pn{noise}_rf{field_value}.nii.gz"
    else:
        raise ValueError("type must be in {'T1', 'T2', 'PD', 'crisp', 'fuzzy'}")
    return _request_get_brainweb(
        download_command, dir / fname, force, shape=STD_SHAPE, dtype=np.uint16
    )


def _request_get_brainweb(
    download_command: str,
    path: os.PathLike,
    force: bool = False,
    dtype: DTypeLike = np.float32,
    shape: tuple = STD_RES,
    obj_mode: bool = False,
) -> None:
    """Request to download brainweb dataset.

    Parameters
    ----------
    do_download_alias : str
        Formatted request code to download a volume from brainweb.
    path : os.PathLike
        Path to save the downloaded file.
    force : bool
        Force download even if the file already exists.
    dtype : DTypeLike
        Data type of the downloaded file.
    shape : tuple
        Shape of the downloaded file.
    obj_mode : bool
        If True, return the downloaded data as a numpy array.

    Returns
    -------
    os.PathLike
        Path to downloaded file.

    Raises
    ------
    Exception
        If the download fails.
    """
    if path.exists() and not force:
        return path
    d = requests.get(
        BASE_URL
        + "?"
        + "&".join(
            [
                f"{k}={v}"
                for k, v in {
                    "do_download_alias": download_command,
                    "format_value": "raw_short",
                    "zip_value": "gnuzip",
                    "who_name": "",
                    "who_institution": "",
                    "who_email": "",
                    "download_for_real": "%5BStart+download%21%5D",
                }.items()
                if v
            ]
        ),
        stream=True,
        headers={"Accept-Encoding": None, "Content-Encoding": "gzip"},
    )

    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    # download
    with io.BytesIO() as buffer:
        with tqdm(
            total=float(d.headers.get("Content-length", 0)),
            desc=f"Downloading {download_command}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            leave=True,
            position=1,
        ) as pbar:
            for chunk in d.iter_content(chunk_size=1024):
                if chunk:
                    buffer.write(chunk)
                    pbar.update(len(chunk))
        data = np.frombuffer(gzip.decompress(buffer.getvalue()), dtype=dtype)
    data = data.reshape(shape)
    if obj_mode:
        return data

    nib.save(nib.Nifti1Image(data, AFFINE), path)
    return path


def _load_tissue_map(brainweb_set: Literal[1, 20]) -> list[dict]:
    with open(
        files("brainweb_dl").joinpath(
            BRAINWEB_VALUES[brainweb_set]
        )
    ) as csvfile:
        return list(csv.DictReader(csvfile))
