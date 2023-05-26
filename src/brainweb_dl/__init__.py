from .mri import get_mri

from ._brainweb import get_brainweb1, get_brainweb20, get_brainweb20_multiple
from ._brainweb import STD_RES, BIG_RES

__all__ = [
    "get_mri",
    "get_brainweb1",
    "get_brainweb20",
    "get_brainweb20_multiple",
    "STD_RES",
    "BIG_RES",
]
