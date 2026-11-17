#!/usr/bin/env python3
import argparse
import csv
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, TEXT, TEXT_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0"
TYPE_SUFFIXES = {"_f16_f32": "f16/f32", "_f16_f16": "f16/f16"}


def split_case(name):
    for suffix, types in TYPE_SUFFIXES.items():
        if name.endswith(suffix):
            return name[: -len(suffix)], types
    return name, "f32/f32"


def ops(shape):
    batch, m, k, n = (int(v) for v in shape.split("x"))
    return 2.0 * batch * m * k * n


def load(paths, column):
    rows = []
    for path in paths:
        source = pathlib.Path(path).stem
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r["phase"] != "matmul":
                    continue
                ms = float(r[column])
                if math.isnan(ms) or ms <= 0:
                    continue
                base, types = split_case(r["case"])
                series = f"{source}: {r['kernel']}" if len(paths) > 1 else r["kernel"]
                rows.append({"case": base, "category": base.split("_", 1)[0], "types": types, "series": series,
                             "gops": ops(r["shape"]) / (ms * 1e6)})
    return rows


def main():
    ap = argparse.ArgumentParser(description="GOP/s по CSV из gemm_bench и op_bench Adept")
    ap.add_argument("csv", nargs="+", help="CSV в формате op_bench: case,phase,shape,wall_ms,gpu_ms,cpu_ms,kernel")
    ap.add_argument("-o", "--output", default="results.png", help="картинка (по умолчанию results.png)")
    ap.add_argument("--time", choices=["wall", "gpu"], default="wall",
                    help="по какому времени считать GOP/s: wall_ms (по умолчанию) или gpu_ms")
    ap.add_argument("--log", action="store_true", help="логарифмическая шкала по оси Y")
    args = ap.parse_args()

    rows = load(args.csv, f"{args.time}_ms")
    if not rows:
        sys.exit(f"{', '.join(args.csv)}: no matmul rows with {args.time}_ms")

    for r in rows:
        print(f"{r['case']:<24} {r['types']:<8} {r['series']:<32} {r['gops']:10.2f} GOP/s")

    names = list(dict.fromkeys(r["series"] for r in rows))
    if len(names) > len(PALETTE):
        print(f"note: {len(names)} lines, plotting the first {len(PALETTE)}", file=sys.stderr)
        names = names[: len(PALETTE)]
    color = dict(zip(names, PALETTE))

    panels = list(dict.fromkeys((r["category"], r["types"]) for r in rows))
    cols = min(3, len(panels))
    grid_rows = math.ceil(len(panels) / cols)

    plt.rcParams.update({"text.color": TEXT, "axes.labelcolor": TEXT_MUTED,
                         "xtick.color": TEXT_MUTED, "ytick.color": TEXT_MUTED})
    fig, axes = plt.subplots(grid_rows, cols, figsize=(6 * cols, 4.5 * grid_rows), squeeze=False,
                             facecolor=SURFACE)
    flat = [ax for row in axes for ax in row]
    for ax in flat[len(panels):]:
        ax.set_visible(False)
    for ax, (category, types) in zip(flat, panels):
        panel_rows = [r for r in rows if r["category"] == category and r["types"] == types]
        cases = list(dict.fromkeys(r["case"] for r in panel_rows))
        x_of = {c: i for i, c in enumerate(cases)}
        ax.set_facecolor(SURFACE)
        for name in names:
            pts = sorted((x_of[r["case"]], r["gops"]) for r in panel_rows if r["series"] == name)
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=color[name], lw=1.5, marker="o", ms=6, label=name)
        ax.set_title(f"{category}  {types}", loc="left", fontsize=12)
        ax.set_xticks(range(len(cases)), [c[len(category) + 1:] or c for c in cases], rotation=30, ha="right")
        ax.set_ylabel(f"GOP/s по {args.time}_ms")
        if args.log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)

    handles = {}
    for ax in flat[: len(panels)]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    labels = [name for name in names if name in handles]
    legend_rows = (len(labels) + 1) // 2
    fig.legend([handles[label] for label in labels], labels, loc="lower center", ncol=min(2, len(labels)),
               frameon=False)
    fig.tight_layout(rect=(0, 0.06 * legend_rows / grid_rows, 1, 1))
    fig.savefig(args.output, dpi=150)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
