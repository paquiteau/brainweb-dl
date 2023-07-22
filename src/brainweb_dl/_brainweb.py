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
from __future__ import annotations
import csv
import logging
import os
import sys
import functools

if sys.version_info > (3, 9):
    from importlib.resources import files
else:
    from importlib_resources import files
import gzip
import io
from pathlib import Path
from typing import Literal

from joblib import Parallel, delayed
import nibabel as nib
import numpy as np
import requests
from numpy.typing import DTypeLike
from tqdm.auto import tqdm

logger = logging.getLogger("brainweb")

# +fmt: off
SUB_ID = (4, 5, 6, 18, 20, 38, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54)
# +fmt: on
BRAINWEB_VALUES = {1: "brainweb1_tissues.csv", 20: "brainweb20_tissues.csv"}

ALLOWED_NOISE_LEVEL = (0, 1, 3, 5, 7, 9)
ALLOWED_RF = (0, 20, 40)
ALLOWED_RES = (1, 3, 5, 7, 9)


BASE_URL = "http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1/"

BIG_RES = (362, 434, 362)
STD_RES = (181, 217, 181)
T1_20_RES = (181, 256, 256)


def get_brainweb_dir(brainweb_dir: os.PathLike = None) -> os.PathLike:
    """Get the Brainweb directory.

    Parameters
    ----------
    brainweb_dir : os.PathLike
       brainweb_directory to download the data.

    Returns
    -------
    os.PathLike
        Path to brainweb_dir

    Notes
    -----
    The brainweb is set in the following order:
    - Thebrainweb_directory passed as argument.
    - The environment variable BRAINWEB_DIR.
    - The defaultbrainweb_directory ~/.cache/brainweb.
    """
    if brainweb_dir is None:
        brainweb_dir = os.environ.get("BRAINWEB_DIR", None)
    if brainweb_dir is None:
        brainweb_dir = Path.home() / ".cache" / "brainweb"
    os.makedirs(brainweb_dir, exist_ok=True)
    return Path(brainweb_dir)


def get_brainweb20_multiple(
    subject: int | list | Literal["all"],
    brainweb_dir: os.PathLike = None,
    force: bool = False,
    segmentation: Literal["crisp", "fuzzy"] = "crisp",
) -> list[os.PathLike]:
    """Download sample or all brainweb subjects.

    Parameters
    ----------
    subject : int | list | Literal["all"]
        subject id or list of subject id to download.
        If "all", download all subjects.
    brainweb_dir : os.PathLike
       brainweb_directory to download the data.
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
    if len(subject) > 1:
        f = []
        for s in tqdm(subject, desc="Downloading brainweb20"):
            f.append(get_brainweb20(s, brainweb_dir, force, segmentation))
        return f
    return get_brainweb20(subject[0], brainweb_dir, force, segmentation)


def get_brainweb20(
    s: int,
    brainweb_dir: os.PathLike = None,
    force: bool = False,
    segmentation: Literal["crisp", "fuzzy"] = "crisp",
    extension: Literal["nii.gz", "nii"] = "nii.gz",
) -> os.PathLike:
    """Download one subject of brainweb dataset.

    Parameters
    ----------
    s : int
        subject id.
    brainweb_dir : os.PathLike
       brainweb_directory to download the data.
    force : bool
        force download even if the file already exists.
    segmentation: "crisp" | "fuzzy"
        segmentation type.
    extension: "nii.gz" | "nii"
        extension of the downloaded file.

    Returns
    -------
    os.PathLike
        Path to downloaded file.
    """
    if s not in SUB_ID:
        raise ValueError(f"subject id {s} not in brainweb20 dataset.")
    brainweb_dir = get_brainweb_dir(brainweb_dir)
    path = brainweb_dir / f"brainweb_s{s:02d}_{segmentation}.{extension}"
    if path.exists() and not force:
        return path

    if segmentation == "crisp":
        download_command = f"subject{s:02d}_{segmentation}"
        data = _request_get_brainweb(
            download_command, path, shape=BIG_RES, dtype=np.uint16, obj_mode=True
        )
        data = data >> 4
        data = data.astype(np.uint8)
    elif segmentation == "fuzzy":
        # Download all the fuzzy segmentation and create a 4D volume.
        path = Path(brainweb_dir) / f"brainweb_s{s:02d}_fuzzy.{extension}"
        # The case of fuzzy segmentation is a bit special.
        # We download all the fuzzy segmentation and create a 4D volume.
        # The 4th dimension is the segmentation type.
        if path.exists() and not force:
            return path
        tissue_map = _load_tissue_map(20)
        data = np.zeros((*BIG_RES, len(tissue_map)), dtype=np.uint16)

        # For faster download, let's use joblib.
        def _download_fuzzy(i: int, tissue: str, data: np.ndarray) -> None:
            name = f"subject{s:02d}_{tissue}"
            data[..., i] = _request_get_brainweb(
                name,
                brainweb_dir / f"{name}",  # placeholder value
                dtype=np.uint16,
                shape=BIG_RES,
                obj_mode=True,
            )

        Parallel(n_jobs=-1, backend="threading")(
            delayed(_download_fuzzy)(i, tissue["ID"], data)
            for i, tissue in enumerate(tissue_map)
        )
    else:
        raise ValueError("segmentation must be 'crisp' or 'fuzzy'")
    return save_array(data, path)


def get_brainweb20_T1(
    s: int,
    brainweb_dir: os.PathLike = ".brainweb",
    force: bool = False,
    extension: Literal["nii.gz", "nii"] = "nii.gz",
) -> os.PathLike | np.ndarray:
    """Download the Brainweb20 T1 Phantom.

    Parameters
    ----------
    s : int
        subject id.
    brainweb_dir : os.PathLike
       brainweb_directory to download the data.
    force : bool
        force download even if the file already exists.
    extension: "nii.gz" | "nii"
        extension of the downloaded file.

    Returns
    -------
    os.PathLike
        Path to downloaded file.

    Notes
    -----
    This is a simple interface to the form available at:
    https://brainweb.bic.mni.mcgill.ca/brainweb/selection_normal.html
    """
    brainweb_dir = get_brainweb_dir(brainweb_dir)
    # download of contrasted images
    download_command = f"subject{s:02d}_t1w"
    fname = f"subject{s:02d}_t1w.{extension}"
    return _request_get_brainweb(
        download_command, brainweb_dir / fname, force, shape=T1_20_RES, dtype=np.uint16
    )


def get_brainweb1(
    type: Literal["T1", "T2", "PD", "crisp", "fuzzy"] = "T2",
    res: int = 1,
    noise: int = 0,
    field_value: int = 0,
    brainweb_dir: os.PathLike = ".brainweb",
    force: bool = False,
    extension: Literal["nii.gz", "nii"] = "nii.gz",
) -> os.PathLike | np.ndarray:
    """Download the Brainweb1 phantom as a nifti file.

    Parameters
    ----------
    type : "T1" | "T2" | "PD" | "crisp" | "fuzzy"
        Type of the phantom to download.
    res : int
        Resolution of the phantom. Must be in {1, 3, 5, 7}
    noise : int
        Percent of noise level in the phantom. Must be in
        {0, 1, 3, 5, 7, 9}
    field_value : int
        RF field value in the phantom. Must be in {0, 20, 40}
    brainweb_dir : os.PathLike
       brainweb_directory to download the data.
    force : bool
        force download even if the file already exists.
    extension: "nii.gz" | "nii"
        extension of the downloaded file.

    Returns
    -------
    os.PathLike
        Path to downloaded file.

    Notes
    -----
    This is a simple interface to the form available at:
    https://brainweb.bic.mni.mcgill.ca/brainweb/selection_normal.html
    """
    brainweb_dir = get_brainweb_dir(brainweb_dir)
    shape = STD_RES
    if res not in ALLOWED_RES:
        raise ValueError(f"Resolution must be in {ALLOWED_RES}")
    if noise not in ALLOWED_NOISE_LEVEL:
        raise ValueError(f"Noise level must be in {ALLOWED_NOISE_LEVEL}")
    if field_value not in ALLOWED_RF:
        raise ValueError(f"RF field value must be in {ALLOWED_RF}")

    if type not in ["T1", "T2", "PD"]:
        raise ValueError("type must be in {'T1', 'T2', 'PD'}")
        # download of contrasted images
    download_command = f"{type}+ICBM+normal+{res}mm+pn{noise}+rf{field_value}"
    fname = f"{type}_ICBM_normal_{res}mm_pn{noise}_rf{field_value}.{extension}"
    shape = (int(np.rint(STD_RES[0] / res)), *STD_RES[1:])
    return _request_get_brainweb(
        download_command, brainweb_dir / fname, force, shape=shape, dtype=np.uint16
    )


def get_brainweb1_seg(
    segmentation: Literal["crisp", "fuzzy"] = "crisp",
    extension: Literal["nii.gz", "nii"] = "nii.gz",
    brainweb_dir: os.PathLike = ".brainweb",
    force: bool = False,
) -> os.PathLike:
    """Download the Brainweb1 phantom segmentation as a nifti file."""
    # The case of fuzzy segmentation is a bit special.
    # We download all the fuzzy segmentation and create a 4D volume.
    # The 4th dimension is the segmentation type.
    if segmentation not in ["crisp", "fuzzy"]:
        raise ValueError("type must be in {'crisp', 'fuzzy'}")
    if segmentation == "crisp":
        download_command = "phantom_1.0mm_normal_crisp"
        fname = f"phantom_1.0mm_normal_crisp.{extension}"
        path = brainweb_dir / fname
        if path.exists() and not force:
            return path
        return _request_get_brainweb(
            download_command,
            brainweb_dir / fname,
            force,
            shape=STD_RES,
            dtype=np.uint16,
        )
    brainweb_dir = get_brainweb_dir(brainweb_dir)
    fname = f"phantom_1.0mm_normal_fuzzy.{extension}"
    path = brainweb_dir / fname
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
        name = f"phantom_1.0mm_normal_{row['ID']}"
        data[..., i] = _request_get_brainweb(
            name,
            path=brainweb_dir / f"{name}.{extension}",  # placeholder
            dtype=np.uint16,
            shape=STD_RES,
            obj_mode=True,
        )
    # Create the 4D volume.
    nib.save(nib.Nifti1Image(data, affine=np.eye(4)), path)
    return path


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
    if not obj_mode and path.exists() and not force:
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

    if not obj_mode:
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
    # download
    with io.BytesIO() as buffer, tqdm(
        total=float(d.headers.get("Content-length", 0)),
        desc=f"Downloading {download_command}",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        leave=False,
        position=2,
    ) as pbar:
        for chunk in d.iter_content(chunk_size=1024):
            if chunk:
                buffer.write(chunk)
                pbar.update(len(chunk))
        data = np.frombuffer(gzip.decompress(buffer.getvalue()), dtype=dtype)
    if data.size != np.prod(shape):
        raise ValueError(f"Mismatch between data size and shape {data.size} != {shape}")
    data = data.reshape(shape)
    if obj_mode:
        return data

    return save_array(data, path)


@functools.lru_cache(maxsize=3)
def _load_tissue_map(brainweb_set: Literal[1, 20]) -> list[dict]:
    with open(
        files("brainweb_dl.data").joinpath(BRAINWEB_VALUES[brainweb_set])
    ) as csvfile:
        return list(csv.DictReader(csvfile))


def save_array(data: np.ndarray, path: os.PathLike) -> os.PathLike:
    if path.suffix == ".npy":
        np.save(path, data)
    else:
        nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    return path
