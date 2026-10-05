"""Download the public IR-imaged fragments (Frag1-Frag6) needed for the benchmark.

Source: Vesuvius Challenge open data (EduceLab-Scrolls dataset, CC-BY-NC 4.0).
Cite: Parsons et al. (2023) EduceLab-Scrolls, doi:10.48550/arXiv.2304.02084

Each fragment is saved as <out_dir>/FragN/ with the same layout:
  surface_volume/00.tif ... 64.tif, inklabels.png, mask.png, ir.png (if published)
Skips the large .ppm/.ply files. Re-running resumes: complete files are skipped.

Approximate sizes: Frag1 6.4 GB, Frag2 17.5 GB, Frag3 4.9 GB, Frag4 7.4 GB,
Frag5 4.1 GB, Frag6 6.8 GB. Frag4 has no IR image aligned to the surface volume.

Usage:  python download_fragments.py Frag2 Frag3 ... [--out DIR]
"""
import argparse
import re
import time
import urllib.request
from pathlib import Path

ROOT = "https://dl.ash2txt.org/fragments/"
FRAGMENTS = {
    # name: (surface folder, surface_volume subfolder, {local name: remote name})
    "Frag1": ("Frag1/PHercParis2Fr47.volpkg/working/54keV_exposed_surface/", "surface_volume/",
              {"ir.png": "ir.png", "inklabels.png": "inklabels.png", "mask.png": "mask.png"}),
    "Frag2": ("Frag2/PHercParis2Fr143.volpkg/working/54keV_exposed_surface/", "surface_volume/",
              {"ir.png": "ir.png", "inklabels.png": "inklabels.png", "mask.png": "mask.png"}),
    "Frag3": ("Frag3/PHercParis1Fr34.volpkg/working/54keV_exposed_surface/", "surface_volume/",
              {"ir.png": "ir.png", "inklabels.png": "inklabels.png", "mask.png": "mask.png"}),
    "Frag4": ("Frag4/PHercParis1Fr39.volpkg/working/54keV_exposed_surface/",
              "PHercParis1Fr39_54keV_surface_volume/",
              {"inklabels.png": "PHercParis1Fr39_54keV_inklabels.png",
               "mask.png": "PHercParis1Fr39_54keV_mask.png"}),
    "Frag5": ("Frag5/PHerc1667Cr1Fr3.volpkg/working/PHerc1667Cr01Fr03_70keV_3.24um/surface_processing/",
              "surface_volume/",
              {"ir.png": "ir.png", "inklabels.png": "inklabels.png", "mask.png": "mask.png"}),
    "Frag6": ("Frag6/PHerc51Cr4Fr8.volpkg/working/PHerc0051Cr04Fr08_53keV_3.24um/surface_processing/",
              "surface_volume/",
              {"ir.png": "ir.png", "inklabels.png": "inklabels.png", "mask.png": "mask.png"}),
}


def list_tifs(url):
    html = urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "replace")
    return sorted(set(re.findall(r'href="(\d+\.tif)"', html)))


def remote_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length", -1))


def fetch(url, dest, attempts=5):
    size = remote_size(url)
    if dest.exists() and dest.stat().st_size == size:
        print(f"  ok (already have) {dest.name}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if size > 0 and tmp.stat().st_size != size:
                raise IOError(f"size mismatch for {dest.name}")
            break
        except (OSError, IOError) as e:  # dropped connections, truncated transfers
            if attempt == attempts:
                raise
            print(f"  retry {attempt}/{attempts - 1} for {dest.name}: {e}", flush=True)
            time.sleep(10 * attempt)
    tmp.replace(dest)
    print(f"  downloaded {dest.name} ({size / 2**20:.1f} MiB)", flush=True)


def download(name, out_root):
    base, sv, files = FRAGMENTS[name]
    out = out_root / name
    (out / "surface_volume").mkdir(parents=True, exist_ok=True)
    print(f"== {name}: labels and IR")
    for local, remote in files.items():
        fetch(ROOT + base + remote, out / local)
    tifs = list_tifs(ROOT + base + sv)
    print(f"== {name}: surface volume, {len(tifs)} layers")
    for t in tifs:
        fetch(ROOT + base + sv + t, out / "surface_volume" / t)
    print(f"== {name} done: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fragments", nargs="+", choices=list(FRAGMENTS))
    ap.add_argument("--out", type=Path, default=Path("D:/DBBun/data/vesuvius"))
    a = ap.parse_args()
    for name in a.fragments:
        download(name, a.out)


if __name__ == "__main__":
    main()
