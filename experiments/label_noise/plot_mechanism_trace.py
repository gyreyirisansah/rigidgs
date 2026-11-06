#!/usr/bin/env python3
"""Turn one or more --enable_mechanism_trace CSVs (from
src/train.py) into a figure answering
"what actually happens to the deliberately-mislabeled Gaussians over
training?" — population count, opacity/suppression of the mismatched
(corrupted-label) subset vs the clean subset, and how much of the mismatch
is resolved by pre-freeze logit self-correction vs post-freeze opacity
suppression.

Usage:
  python3 plot_mechanism_trace.py \
      --traces "fg2bg=label_noise_experiment/traces/kitchen_fg2bg_r020/trace.csv" \
               "bg2fg=label_noise_experiment/traces/kitchen_bg2fg_r020/trace.csv" \
      --logit_freeze_iter 500 \
      --output label_noise_experiment/mechanism_trace_kitchen.png

Each --traces entry is "label=path_to_trace.csv". Produces one figure with
one column per trace, 4 rows:
  1. Gaussian population: total count, and cloned/split/pruned deltas.
  2. Opacity (median, with mean as a fainter line) over iteration, split
     into "still mismatched" vs "correctly labeled" Gaussians.
  3. Fraction of each population below the suppression threshold.
  4. Fraction of currently-mismatched Gaussians whose EFFECTIVE (rendered)
     label already agrees with the true label via pre-freeze logits, vs
     fraction with opacity already suppressed — the two coexisting
     "recovery" mechanisms.
"""
import argparse
import csv as csv_mod

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_trace(path):
    """Reads every numeric column generically (robust to columns added later,
    e.g. the opacity_p10..p90 percentiles) rather than hand-picking a subset."""
    rows = []
    with open(path) as f:
        reader = csv_mod.DictReader(f)
        for r in reader:
            row = {}
            for key, v in r.items():
                if key == "iteration":
                    row[key] = int(v)
                    continue
                if v == "" or v is None:
                    row[key] = None
                    continue
                try:
                    row[key] = float(v)
                except ValueError:
                    row[key] = None
            rows.append(row)
    rows.sort(key=lambda r: r["iteration"])
    return rows


def _col(rows, key):
    return [r.get(key) for r in rows]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces", nargs="+", required=True,
                   help="One or more 'label=path.csv' entries.")
    p.add_argument("--logit_freeze_iter", type=int, default=500,
                   help="Draws a vertical reference line at this iteration "
                        "(the trainer's default LOGIT_FREEZE_ITER).")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    traces = []
    for entry in args.traces:
        label, path = entry.split("=", 1)
        traces.append((label, load_trace(path)))

    n_cols = len(traces)
    fig, axes = plt.subplots(4, n_cols, figsize=(6 * n_cols, 14), squeeze=False)

    for col, (label, rows) in enumerate(traces):
        it = [r["iteration"] for r in rows]

        # Row 0: population
        ax = axes[0][col]
        ax.plot(it, [r["n_total"] for r in rows], color="black", label="total Gaussians")
        ax2 = ax.twinx()
        cloned = [r["n_cloned"] or 0 for r in rows]
        split = [r["n_split"] or 0 for r in rows]
        pruned = [-(r["n_pruned"] or 0) for r in rows]
        ax2.bar(it, cloned, width=40, color="tab:green", alpha=0.5, label="cloned")
        ax2.bar(it, split, width=40, bottom=cloned, color="tab:blue", alpha=0.5, label="split")
        ax2.bar(it, pruned, width=40, color="tab:red", alpha=0.5, label="pruned")
        ax.set_title(f"{label}: Gaussian population")
        ax.set_ylabel("total count")
        ax2.set_ylabel("added / pruned this event")
        ax.axvline(args.logit_freeze_iter, color="gray", linestyle="--", linewidth=1)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

        # Row 1: opacity — median line + p25-p75 band + p10/p90 dotted, for
        # the mismatched (flipped) population specifically, since mean/median
        # alone can hide a bimodal split (some recovering, some crushed).
        ax = axes[1][col]
        has_pcts = any(r.get("mismatch_opacity_p25") is not None for r in rows)
        if has_pcts:
            p10 = [r.get("mismatch_opacity_p10") for r in rows]
            p25 = [r.get("mismatch_opacity_p25") for r in rows]
            p75 = [r.get("mismatch_opacity_p75") for r in rows]
            p90 = [r.get("mismatch_opacity_p90") for r in rows]
            ax.fill_between(it, p25, p75, color="tab:red", alpha=0.25, label="mismatched p25-p75")
            ax.plot(it, p10, color="tab:red", linestyle=":", linewidth=1, label="mismatched p10/p90")
            ax.plot(it, p90, color="tab:red", linestyle=":", linewidth=1)
            m_p25 = [r.get("match_opacity_p25") for r in rows]
            m_p75 = [r.get("match_opacity_p75") for r in rows]
            ax.fill_between(it, m_p25, m_p75, color="tab:green", alpha=0.20, label="correct p25-p75")
        ax.plot(it, [r["mismatch_opacity_median"] for r in rows], color="tab:red",
                label="mismatched-label Gaussians (median opacity)")
        ax.plot(it, [r["match_opacity_median"] for r in rows], color="tab:green",
                label="correctly-labeled Gaussians (median opacity)")
        ax.axvline(args.logit_freeze_iter, color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel("opacity")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7, loc="center right")
        ax.set_title(f"{label}: opacity distribution, mismatched vs correct")

        # Row 2: suppression fraction
        ax = axes[2][col]
        ax.plot(it, [r["mismatch_frac_suppressed"] for r in rows], color="tab:red",
                label="mismatched: frac. opacity < threshold")
        ax.plot(it, [r["match_frac_suppressed"] for r in rows], color="tab:green",
                label="correct: frac. opacity < threshold")
        ax.axvline(args.logit_freeze_iter, color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel("fraction suppressed")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8, loc="center right")
        ax.set_title(f"{label}: suppression fraction")

        # Row 3: two recovery mechanisms
        ax = axes[3][col]
        ax.plot(it, [r["frac_mismatch_self_corrected_by_logits"] for r in rows], color="tab:purple",
                label="frac. mismatched already self-corrected\nby pre-freeze logits (rendered as true label)")
        ax.plot(it, [r["mismatch_frac_suppressed"] for r in rows], color="tab:red", linestyle=":",
                label="frac. mismatched opacity-suppressed\n(repeated from row above)")
        ax.axvline(args.logit_freeze_iter, color="gray", linestyle="--", linewidth=1,
                   label=f"LOGIT_FREEZE_ITER={args.logit_freeze_iter}")
        ax.set_ylabel("fraction of mismatched Gaussians")
        ax.set_xlabel("iteration")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7, loc="center right")
        ax.set_title(f"{label}: two recovery mechanisms")

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
