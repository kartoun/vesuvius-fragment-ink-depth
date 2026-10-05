"""Shared paths and loaders for the 3D ink-label project."""
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
DATA = Path("D:/DBBun/data/vesuvius/Frag1")
LABELS_OUT = Path("D:/DBBun/data/vesuvius/Frag1_3d_labels")
OUT = Path(__file__).parent / "outputs"

# label values in the 3D volumes
NOT_INK, INK, IGNORE = 0, 1, 2


def load_png_mask(name, data=DATA):
    return np.array(Image.open(data / name)) > 0


def load_stack(data=DATA, verbose=True):
    """All surface-volume layers as uint16 (Z, H, W). ~6.7 GB for Frag1."""
    paths = sorted((data / "surface_volume").glob("*.tif"))
    first = tifffile.imread(paths[0])
    stack = np.empty((len(paths), *first.shape), np.uint16)
    stack[0] = first
    for i, p in enumerate(paths[1:], 1):
        stack[i] = tifffile.imread(p)
        if verbose:
            print(f"\rloading layer {i + 1}/{len(paths)}", end="", flush=True)
    if verbose:
        print()
    return stack
