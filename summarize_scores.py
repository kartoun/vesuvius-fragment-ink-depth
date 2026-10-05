"""Collect score_3d.py results (outputs/scores/*.json) into one Markdown table."""
import json

from common import OUT


def f(v, c=None):
    s = f"{v:.3f}" if isinstance(v, float) else str(v)
    return f"{s} [{c[0]:.3f}, {c[1]:.3f}]" if c and c[0] is not None else s


def main():
    rows = []
    for p in sorted((OUT / "scores").glob("*.json")):
        r = json.loads(p.read_text())
        a, b, c = r["a_ink_map"], r["b_depth"], r["c_ink_vs_surface"]
        rows.append(f"| {r['name']} | {f(a['auc'], a['auc_ci'])} | {f(a['ap'], a['ap_ci'])} | "
                    f"{a['spearman_ir_darkness'] if a['spearman_ir_darkness'] is None else round(a['spearman_ir_darkness'], 3)} | {b['argmax_offset_median']:+.0f} | "
                    f"{f(b['share_argmax_within_5'], b['share_argmax_within_5_ci'])} | "
                    f"{f(c['surface_share'], c['surface_share_ci'])} | {f(c['band_auc'], c['band_auc_ci'])} |")
    head = ("| volume | (a) AUC | (a) AP* | (a) rho vs IR | (b) peak offset (median) | (b) peak within +-5 | "
            "(c) surface share (lower = more ink-specific) | (c) band AUC |\n|" + "---|" * 8)
    txt = head + "\n" + "\n".join(rows) + "\n\n*AP on a per-tile class-balanced sample, so not comparable to AP on all pixels.\n"
    (OUT / "scores" / "summary.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
