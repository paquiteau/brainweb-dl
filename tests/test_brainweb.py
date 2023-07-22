"""Test for the brainweb module."""
import numpy as np

import pytest
from brainweb_dl import get_mri, get_brainweb20_multiple, get_brainweb1_seg


@pytest.mark.parametrize(
    "sub_id, contrast, res, noise, field_value",
    [
        (0, "T1", 1, 1, 20),
        (0, "T2", 3, 5, 40),
        (0, "PD", 7, 9, 0),
    ],
)
def test_get_mri1(sub_id, contrast, res, noise, field_value, tmp_path):
    """Test retrieval of available data."""
    data = get_mri(
        sub_id,
        contrast,
        brainweb_dir=tmp_path,
        res=res,
        noise=noise,
        field_value=field_value,
    )

    assert data.shape == (int(np.rint(181 / res)), 217, 181)


@pytest.mark.parametrize(
    "kwargs", [{"res": None}, {"noise": None}, {"field_value": None}]
)
def test_get_mri1_unavailable(kwargs, tmp_path):
    """Test retrieval of unavailable data."""
    with pytest.raises(ValueError):
        get_mri(0, "T1", **kwargs, brainweb_dir=tmp_path)


def test_get_mri20T1(tmp_path):
    """Test retrieval of available data."""
    data = get_mri(4, "T1", brainweb_dir=tmp_path, force=True)
    assert data.shape == (181, 256, 256)


@pytest.mark.parametrize("contrast", ["T2*"])
def test_get_mri20_custom(contrast, tmp_path):
    data = get_mri(4, contrast, brainweb_dir=tmp_path, force=True)
    assert data.shape == (362, 434, 362)


@pytest.mark.parametrize("contrast", ["T2*"])
def test_get_mri1_custom(contrast, tmp_path):
    data = get_mri(0, contrast, brainweb_dir=tmp_path, force=True)
    assert data.shape == (181, 217, 181)


@pytest.mark.parametrize("seg", ["fuzzy", "crisp"])
def test_get_seg(seg, tmp_path):
    """Test retrieval of available data."""
    paths = get_brainweb20_multiple(
        [4, 44], segmentation=seg, brainweb_dir=tmp_path, force=True
    )
    print(paths)


@pytest.mark.parametrize("seg", ["fuzzy", "crisp"])
def test_get_seg2(seg, tmp_path):
    """Test retrieval of available data."""
    paths = get_brainweb1_seg(segmentation=seg, brainweb_dir=tmp_path, force=True)
    print(paths)
