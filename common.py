"""Shared paths and loaders for the 3D ink-label project."""
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
DATA_ROOT = Path("D:/DBBun/data/vesuvius")
OUT = Path(__file__).parent / "outputs"

# label values in the 3D volumes
NOT_INK, INK, IGNORE = 0, 1, 2


def frag_paths(name):
    """(data folder, label/reference folder) for a fragment such as 'Frag1'."""
    return DATA_ROOT / name, DATA_ROOT / f"{name}_3d_labels"


DATA, LABELS_OUT = frag_paths("Frag1")


def load_png_mask(name, data=DATA):
    return np.array(Image.open(data / name)) > 0


class LazyStack:
    """Surface-volume layers as memory-mapped uint16 TIFFs; read row blocks on demand."""

    def __init__(self, data=DATA):
        self.layers = [tifffile.memmap(p, mode="r") for p in sorted((data / "surface_volume").glob("*.tif"))]
        self.shape = (len(self.layers), *self.layers[0].shape)

    def rows(self, r0, r1):
        return np.stack([m[r0:r1] for m in self.layers])


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
