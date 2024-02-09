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
def test_get_mri1(sub_id, contrast, res, noise, field_value, bw_dir):
    """Test retrieval of available data."""
    data = get_mri(
        sub_id,
        contrast,
        brainweb_dir=bw_dir,
        download_res=res,
        noise=noise,
        field_value=field_value,
    )

    assert data.shape == (int(np.rint(181 / res)), 217, 181)


@pytest.mark.parametrize(
    "kwargs", [{"download_res": None}, {"noise": None}, {"field_value": None}]
)
def test_get_mri1_unavailable(kwargs, bw_dir):
    """Test retrieval of unavailable data."""
    with pytest.raises(ValueError):
        get_mri(0, "T1", **kwargs, brainweb_dir=bw_dir)


def test_get_mri20T1(bw_dir, force):
    """Test retrieval of available data."""
    data = get_mri(4, "T1", brainweb_dir=bw_dir, force=force)
    assert data.shape == (181, 256, 256)


@pytest.mark.parametrize("contrast", ["T2*"])
def test_get_mri20_custom(contrast, bw_dir, force):
    """Test brainweb v2 with T2* data."""
    data = get_mri(4, contrast, brainweb_dir=bw_dir, force=force)
    assert data.shape == (362, 434, 362)


@pytest.mark.parametrize("contrast", ["T2*"])
def test_get_mri1_custom(contrast, bw_dir, force):
    """Test brainweb v1 with T2* data."""
    data = get_mri(0, contrast, brainweb_dir=bw_dir, force=force)
    assert data.shape == (181, 217, 181)


@pytest.mark.parametrize("seg", ["fuzzy", "crisp"])
def test_get_seg(seg, bw_dir, force):
    """Test retrieval of available data."""
    paths = get_brainweb20_multiple(
        [4, 44], segmentation=seg, brainweb_dir=bw_dir, force=force
    )
    print(paths)


@pytest.mark.parametrize("seg", ["fuzzy", "crisp"])
def test_get_seg2(seg, bw_dir, force):
    """Test retrieval of available data."""
    paths = get_brainweb1_seg(segmentation=seg, brainweb_dir=bw_dir, force=force)
    print(paths)
