"""Test configurations."""

import pytest


def pytest_addoption(parser):
    """Add option to pytest CLI."""
    parser.addoption(
        "--force",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="force mode to test.",
    )


def pytest_generate_tests(metafunc):
    """Extend test with force mode on or off."""
    force_values = {
        0: [False],
        1: [True],
        2: [True, False],
    }
    val = force_values[metafunc.config.getoption("force")]
    print("VAL", val)
    if "force" in metafunc.fixturenames:
        metafunc.parametrize("force", val)


@pytest.fixture(scope="session")
def bw_dir(tmp_path_factory):
    """Create a temporary directory for brainweb files."""
    fn = tmp_path_factory.mktemp("bw_dir")
    return fn
