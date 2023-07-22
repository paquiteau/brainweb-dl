"""CLI function for the package."""
from __future__ import annotations
import argparse
import logging

import nibabel as nib
import numpy as np

from ._brainweb import SUB_ID, get_brainweb1, get_brainweb1_seg, get_brainweb20_multiple
from .mri import get_mri


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Get data from the brainWeb Dataset",
        epilog="For more information, visit https://github.com/paquiteau/brainweb-dl",
    )
    parser.add_argument(
        "subject",
        type=int,
        help="Subject ID",
        nargs="*",
        choices=[-1, 0, *SUB_ID],
        default=-1,
    )
    parser.add_argument(
        "--contrast",
        type=str,
        help="Contrast to download/create. ",
        choices=["T1", "T2", "T2*", "crisp", "fuzzy"],
    )
    parser.add_argument(
        "--brainweb-dir",
        type=str,
        help="Brainweb directory, overrides the environment variable BRAINWEB_DIR",
    )
    parser.add_argument(
        "--extension",
        type=str,
        help="Output format. Default: nii.gz",
        nargs="?",
        choices=["nii.gz", "nii", "npy"],
        default="nii.gz",
    )
    parser.add_argument("--rng", type=int, help="Random seed", default=None)
    parser.add_argument("--all", action="store_true", help="Download all subjects")

    ns = parser.parse_args()
    if ns.all and ns.subject == -1:
        ns.subject = [*SUB_ID]
    elif ns.subject == -1:
        raise ValueError("Subject ID or --all is required")
    return ns


def main() -> None:
    """CLI interface."""
    ns = parse_args()

    if ns.contrast in ["T1", "T2", "T2*"]:
        array = get_mri(
            ns, ns.contrast, brainweb_dir=ns.brainweb_dir, extension=ns.format
        )
        filename = f"brainweb_{ns.subject}_{ns.contrast}.{ns.format}"
        nib.Nifti1Image(array, np.eye(4)).to_filename(filename)
    elif ns.contrast in ["crisp", "fuzzy"]:
        if ns.subject == 0:
            filename = get_brainweb1_seg(ns.contrast, brainweb_dir=ns.brainweb_dir)
        else:
            filename = get_brainweb20_multiple(
                ns.subject, segmentation=ns.contrast, brainweb_dir=ns.brainweb_dir
            )
    else:
        raise ValueError(f"Unknown contrast {ns.contrast}")
    if isinstance(filename, list):
        if len(filename) == 1:
            print(f"Data saved to {filename[0]}")
        else:
            print(f"{len(filename)} files saved to {filename[0].parent}")
    else:
        print(f"Data saved to {filename}")


if __name__ == "__main__":
    main()
