#!/usr/bin/env python3
import argparse
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, TEXT, TEXT_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0"


def layout(r):
    s = ("t" if r["transA"] == "1" else "n") + ("t" if r["transB"] == "1" else "n")
    s += f" b{r['batch']}"
    return s + (" bcastB" if r["broadcastB"] == "1" else "")


def shape(r):
    m, n, k = r["M"], r["N"], r["K"]
    return m if m == n == k else f"{m}x{n}x{k}"


def best_rows(rows):
    best = {}
    for r in rows:
        key = (r["kernel"], r["dtype"], r["M"], r["N"], r["K"], layout(r))
        if key not in best or float(r["gops"]) > float(best[key]["gops"]):
            best[key] = r
    return list(best.values())


def main():
    ap = argparse.ArgumentParser(description="График GOP/s по CSV из gemm_bench --csv")
    ap.add_argument("csv", help="файл из gemm_bench --csv")
    ap.add_argument("-o", "--output", default="results.png", help="картинка (по умолчанию results.png)")
    ap.add_argument("--best", action="store_true",
                    help="сравнение ядер: для каждого ядра и задачи лучший набор параметров")
    args = ap.parse_args()

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{args.csv}: no results")

    vary_layout = len({layout(r) for r in rows}) > 1

    if args.best:
        rows = best_rows(rows)
        for r in rows:
            print(f"{r['kernel']}  {r['dtype']} {shape(r)} {layout(r)}  best: {r['params']}  "
                  f"({float(r['gops']):.2f} GOP/s)")

    def series(r):
        name = r["kernel"] if args.best else f"{r['kernel']} [{r['params']}]"
        return f"{name} {layout(r)}" if vary_layout else name

    names = list(dict.fromkeys(series(r) for r in rows))
    if len(names) > len(PALETTE):
        print(f"note: {len(names)} lines, plotting the first {len(PALETTE)}; narrow the sweep "
              "(--kernel, --params) for the rest", file=sys.stderr)
        names = names[: len(PALETTE)]
    color = dict(zip(names, PALETTE))

    shapes = sorted({(int(r["M"]) * int(r["N"]) * int(r["K"]), shape(r)) for r in rows})
    x_of = {label: i for i, (_, label) in enumerate(shapes)}
    dtypes = list(dict.fromkeys(r["dtype"] for r in rows))

    plt.rcParams.update({"text.color": TEXT, "axes.labelcolor": TEXT_MUTED,
                         "xtick.color": TEXT_MUTED, "ytick.color": TEXT_MUTED})
    fig, axes = plt.subplots(1, len(dtypes), figsize=(6 * len(dtypes), 4.5), squeeze=False,
                             facecolor=SURFACE)
    for ax, dtype in zip(axes[0], dtypes):
        ax.set_facecolor(SURFACE)
        for name in names:
            pts = sorted((x_of[shape(r)], float(r["gops"])) for r in rows
                         if r["dtype"] == dtype and series(r) == name)
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=color[name], lw=1.5, marker="o", ms=6, label=name)
        ax.set_title(dtype, loc="left", fontsize=12)
        ax.set_xticks(range(len(shapes)), [label for _, label in shapes])
        ax.set_xlabel("размер: N или MxNxK")
        ax.set_ylabel("GOP/s, лучшие параметры ядра" if args.best else "GOP/s (по среднему времени)")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(2, len(labels)), frameon=False)
    fig.tight_layout(rect=(0, 0.06 * ((len(labels) + 1) // 2), 1, 1))
    fig.savefig(args.output, dpi=150)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
