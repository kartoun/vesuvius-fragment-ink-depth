"""Download the public Frag1 (PHercParis2Fr47) exposed-surface data needed for 3D ink labels.

Source: Vesuvius Challenge open data (EduceLab-Scrolls dataset, CC-BY-NC 4.0).
Cite: Parsons et al. (2023) EduceLab-Scrolls, doi:10.48550/arXiv.2304.02084

Downloads ~6.5 GB: 65 surface-volume layers (~99 MB each) plus the aligned
IR photo, hand-traced ink labels and fragment mask. Skips the large .ppm/.ply
files. Re-running resumes: files already complete are skipped.

Usage:  python download_frag1.py [output_dir]
"""
import re
import sys
import urllib.request
from pathlib import Path

BASE = ("https://dl.ash2txt.org/fragments/Frag1/PHercParis2Fr47.volpkg/"
        "working/54keV_exposed_surface/")
SMALL_FILES = ["ir.png", "inklabels.png", "mask.png"]


def list_tifs(url):
    html = urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "replace")
    return sorted(set(re.findall(r'href="(\d+\.tif)"', html)))


def remote_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length", -1))


def fetch(url, dest):
    size = remote_size(url)
    if dest.exists() and dest.stat().st_size == size:
        print(f"  ok (already have) {dest.name}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    if size > 0 and tmp.stat().st_size != size:
        raise IOError(f"size mismatch for {dest.name}")
    tmp.replace(dest)
    print(f"  downloaded {dest.name} ({size / 2**20:.1f} MiB)")


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("D:/DBBun/data/vesuvius/Frag1")
    (out / "surface_volume").mkdir(parents=True, exist_ok=True)

    print("Labels and IR photo:")
    for name in SMALL_FILES:
        fetch(BASE + name, out / name)

    tifs = list_tifs(BASE + "surface_volume/")
    print(f"Surface volume: {len(tifs)} layers")
    for name in tifs:
        fetch(BASE + "surface_volume/" + name, out / "surface_volume" / name)
    print(f"Done. Data in {out.resolve()}")


if __name__ == "__main__":
    main()
