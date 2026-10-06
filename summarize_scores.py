"""Collect score_3d.py results (outputs/scores/*.json) into one Markdown table."""
import json

from common import OUT


def f(v, c=None):
    if v is None:
        return "n/a"
    s = f"{v:.3f}" if isinstance(v, float) else str(v)
    return f"{s} [{c[0]:.3f}, {c[1]:.3f}]" if c and c[0] is not None else s


def main():
    rows = []
    for p in sorted((OUT / "scores").glob("*.json")):
        r = json.loads(p.read_text())
        a, b, c = r["a_ink_map"], r["b_depth"], r["c_ink_vs_surface"]
        frag = r["name"].split("_")[0] if r["name"].startswith("Frag") else "Frag1"
        rows.append(f"| {frag} | {r['name']} | {f(a['auc'], a['auc_ci'])} | {f(a['ap'], a['ap_ci'])} | "
                    f"{f(a['spearman_ir_darkness'])} | {b['argmax_offset_median']:+.0f} | "
                    f"{f(b['share_argmax_within_5'], b['share_argmax_within_5_ci'])} | "
                    f"{f(c['surface_share'], c['surface_share_ci'])} | {f(c['band_auc'], c['band_auc_ci'])} |")
    head = ("| fragment | volume | (a) AUC | (a) AP | (a) rho vs IR | (b) peak offset (median) | (b) peak within +-5 | "
            "(c) surface share (lower = more ink-specific) | (c) band AUC |\n|" + "---|" * 9)
    note = ("\npred_* rows: models trained on Frag1 (step 4); on Frag1 they are scored on the held-out band "
            "(rows 3298-4432), on Frag2-6 on the whole fragment. *_lofo_* rows: models trained on the other five "
            "fragments (step 5), scored on the whole held-out fragment. 95% CIs from a block bootstrap over 256 x 256 px tiles. For label volumes "
            "only (b) is informative: (a) and (c) are near-trivial because every label set derives from the same 2D outline.\n")
    txt = head + "\n" + "\n".join(rows) + "\n" + note
    (OUT / "scores" / "summary.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
