#!/usr/bin/env python3
import argparse, importlib, json, math, os, random, re
from typing import Any, Dict, List, Tuple
import numpy as np

import matplotlib
matplotlib.use("Agg")  # write files only
import matplotlib.pyplot as plt

# ------------------------------- loader -------------------------------

def load_evaluate(module_name: str):
    mod = importlib.import_module(module_name)
    if not hasattr(mod, "evaluate"):
        raise AttributeError(f"Module '{module_name}' has no 'evaluate(user_data, answer, targets)'")
    return mod.evaluate

# ------------------------------- helpers ------------------------------

def _call_eval(EVAL, user_data: Dict[str, Any], ans_obj: Dict[str, Any], targets: Dict[str, Any]) -> float:
    """Return only the numeric score."""
    ans = json.dumps(ans_obj, ensure_ascii=False)
    out = EVAL(user_data, ans, targets)
    if isinstance(out, dict):
        return float(out.get("score", 0.0))
    return float(out)

def _call_eval_full(EVAL, user_data: Dict[str, Any], ans_obj: Dict[str, Any], targets: Dict[str, Any]) -> Dict[str, Any]:
    """Return the full dict from evaluate (score, reason, etc.)."""
    ans = json.dumps(ans_obj, ensure_ascii=False)
    out = EVAL(user_data, ans, targets)
    return out if isinstance(out, dict) else {"score": float(out), "is_score_valid": True, "reason": ""}

def _linspace(a: float, b: float, n: int) -> List[float]:
    if n <= 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]

def _extract_rank(reason: str) -> Tuple[int | None, float | None]:
    """Parse 'rank=' and 'rank_factor=' from the reason string (if present)."""
    m_k = re.search(r"rank=(\d+)", reason)
    m_rf = re.search(r"rank_factor=([0-9.]+)", reason)
    k = int(m_k.group(1)) if m_k else None
    rf = float(m_rf.group(1)) if m_rf else None
    return k, rf

# ============================= score_candidate =============================

def score_candidate_suite(EVAL, plots_dir: str):
    user_data = {"task": "score_candidate", "data": {"candidate": {"index": 2}, "confidence_threshold": 0.85}}
    gold_confs = [0.60, 0.85, 0.93, 0.98]
    xs = _linspace(0.0, 1.0, 201)

    print("\n[score_candidate] sweep across GT confidences:")
    for gc in gold_confs:
        targets = {"option_number": 2, "confidence": gc}
        ys_ok, ys_bad = [], []
        for x in xs:
            ys_ok.append(_call_eval(EVAL, user_data, {"option_number": 2, "confidence": x}, targets))
            ys_bad.append(_call_eval(EVAL, user_data, {"option_number": 0, "confidence": x}, targets))
        peak_x = xs[int(np.argmax(ys_ok))]
        print(f"  GT={gc:.2f} -> peak at ~{peak_x:.3f}; wrong@0={ys_bad[0]:.3f}, wrong@1={ys_bad[-1]:.5f}")

        plt.figure()
        plt.plot(xs, ys_ok, label="correct index")
        plt.plot(xs, ys_bad, label="wrong index")
        plt.title(f"score_candidate curves (GT={gc:.2f})")
        plt.xlabel("predicted confidence")
        plt.ylabel("reward")
        plt.grid(True, alpha=0.3)
        plt.legend()
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, f"score_candidate_GT{int(round(gc*100))}.png"), dpi=144, bbox_inches="tight")
        plt.close()

    # Heatmap pred_conf vs gold_conf
    xs = np.linspace(0, 1, 121)
    gcs = np.linspace(0.50, 0.99, 100)
    H = np.zeros((len(gcs), len(xs)))
    for i, gc in enumerate(gcs):
        targets = {"option_number": 2, "confidence": float(gc)}
        for j, x in enumerate(xs):
            H[i, j] = _call_eval(EVAL, user_data, {"option_number": 2, "confidence": float(x)}, targets)
    plt.figure()
    plt.imshow(H, aspect="auto", origin="lower", extent=[xs[0], xs[-1], gcs[0], gcs[-1]])
    plt.colorbar()
    plt.title("score_candidate heatmap (reward vs pred_conf × gold_conf)")
    plt.xlabel("pred_conf")
    plt.ylabel("gold_conf")
    plt.savefig(os.path.join(plots_dir, "score_candidate_heatmap.png"), dpi=144, bbox_inches="tight")
    plt.close()

# ============================= select_chapters =============================

def select_chapters_suite(EVAL, plots_dir: str):
    targets = {
        "chapters": [
            {"chapter": "84 - Machinery", "confidence": 0.92},  # GT
            {"chapter": "85 - Electrical", "confidence": 0.60},
            {"chapter": "90 - Instruments", "confidence": 0.35},
        ]
    }
    user_data = {"task": "select_chapters", "data": {"confidence_threshold": 0.85}}
    xs = _linspace(0.0, 1.0, 201)
    gaps = [-0.20, -0.10, -0.02, +0.02, +0.10, +0.20]

    print("\n[select_chapters] competitor gap curves:")
    overlay = []  # to plot on one axis later: (gap, xs, ys, k_at_peak, rf_at_peak)
    for gap in gaps:
        ys = []
        for x in xs:
            c2 = max(0.0, min(1.0, x + gap))  # single competitor at x + gap
            ans = {
                "chapters": [
                    {"chapter": "84 - Machinery", "confidence": x},
                    {"chapter": "85 - Electrical", "confidence": c2},
                    {"chapter": "90 - Instruments", "confidence": 0.02},
                ]
            }
            ys.append(_call_eval(EVAL, user_data, ans, targets))

        peak_idx = int(np.argmax(ys))
        peak_x = xs[peak_idx]
        c2_peak = max(0.0, min(1.0, peak_x + gap))
        ans_peak = {
            "chapters": [
                {"chapter": "84 - Machinery", "confidence": peak_x},
                {"chapter": "85 - Electrical", "confidence": c2_peak},
                {"chapter": "90 - Instruments", "confidence": 0.02},
            ]
        }
        res_peak = _call_eval_full(EVAL, user_data, ans_peak, targets)
        k_peak, rf_peak = _extract_rank(res_peak.get("reason", ""))
        print(f"  gap={gap:+.2f} -> max={max(ys):.3f} at x={peak_x:.3f}; rank≈{k_peak}; factor≈{(rf_peak if rf_peak is not None else float('nan')):.3f}")
        # per-gap figure
        plt.figure()
        plt.plot(xs, ys)
        plt.title(f"select_chapters (competitor gap={gap:+.2f})")
        plt.xlabel("GT_pred_conf")
        plt.ylabel("reward")
        plt.grid(True, alpha=0.3)
        os.makedirs(plots_dir, exist_ok=True)
        fname = f"select_chapters_gap_{gap:+.2f}.png".replace("+","p").replace("-","m")
        plt.savefig(os.path.join(plots_dir, fname), dpi=144, bbox_inches="tight")
        plt.close()
        # collect for overlay
        overlay.append((gap, xs, ys, k_peak, rf_peak))

    # Overlay all gap curves on one axis with fixed ylim so amplitudes are visible
    plt.figure()
    for gap, xs_, ys_, k_peak, rf_peak in overlay:
        lbl = f"gap={gap:+.2f} (k@peak={k_peak}, rf@peak={rf_peak:.3f})" if (k_peak and rf_peak is not None) else f"gap={gap:+.2f}"
        plt.plot(xs_, ys_, label=lbl)
    plt.ylim(0.0, 1.0)
    plt.title("select_chapters overlay (all gaps)")
    plt.xlabel("GT_pred_conf")
    plt.ylabel("reward")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncols=2)
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, "select_chapters_overlay.png"), dpi=144, bbox_inches="tight")
    plt.close()

    # Add a rank-3 curve (two competitors always ahead) with a representative positive gap
    gap = +0.10
    ys_rank3 = []
    for x in xs:
        comp1 = max(0.0, min(1.0, x + gap))
        comp2 = max(0.0, min(1.0, x + gap + 0.03))
        ans_r3 = {
            "chapters": [
                {"chapter": "84 - Machinery", "confidence": x},
                {"chapter": "85 - Electrical", "confidence": comp1},
                {"chapter": "90 - Instruments", "confidence": comp2},
            ]
        }
        ys_rank3.append(_call_eval(EVAL, user_data, ans_r3, targets))
    plt.figure()
    for gap, xs_, ys_, k_peak, rf_peak in overlay:
        plt.plot(xs_, ys_, alpha=0.5, label=f"gap={gap:+.2f}")
    plt.plot(xs, ys_rank3, linewidth=2.2, label="rank≈3 curve (gap=+0.10, +0.13)")
    plt.ylim(0.0, 1.0)
    plt.title("select_chapters: rank-2 vs rank-3 comparison")
    plt.xlabel("GT_pred_conf")
    plt.ylabel("reward")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncols=2)
    plt.savefig(os.path.join(plots_dir, "select_chapters_rank3_overlay.png"), dpi=144, bbox_inches="tight")
    plt.close()

    # Heatmap over GT_conf × competitor_conf
    X = np.linspace(0, 1, 121)
    C = np.linspace(0, 1, 121)
    Z = np.zeros((len(C), len(X)))
    K = np.zeros_like(Z)  # inferred rank at each cell (1 or 2 here)
    for i, comp in enumerate(C):
        for j, x in enumerate(X):
            ans = {
                "chapters": [
                    {"chapter": "84 - Machinery", "confidence": float(x)},
                    {"chapter": "85 - Electrical", "confidence": float(comp)},
                    {"chapter": "90 - Instruments", "confidence": 0.02},
                ]
            }
            res = _call_eval_full(EVAL, user_data, ans, targets)
            Z[i, j] = float(res.get("score", 0.0))
            k, _ = _extract_rank(res.get("reason", ""))
            K[i, j] = float(k or 0)
    # reward heatmap
    plt.figure()
    plt.imshow(Z, aspect="auto", origin="lower", extent=[X[0], X[-1], C[0], C[-1]])
    plt.colorbar()
    plt.title("select_chapters heatmap (reward vs GT_conf × competitor_conf)")
    plt.xlabel("GT_pred_conf")
    plt.ylabel("competitor_conf")
    plt.savefig(os.path.join(plots_dir, "select_chapters_heatmap.png"), dpi=144, bbox_inches="tight")
    plt.close()
    # rank heatmap (so you can see the k=1 vs k=2 regions)
    plt.figure()
    plt.imshow(K, aspect="auto", origin="lower", extent=[X[0], X[-1], C[0], C[-1]], vmin=1, vmax=3)
    cbar = plt.colorbar()
    cbar.set_label("inferred rank k")
    plt.title("select_chapters inferred rank (k) over GT_conf × competitor_conf")
    plt.xlabel("GT_pred_conf")
    plt.ylabel("competitor_conf")
    plt.savefig(os.path.join(plots_dir, "select_chapters_rank_heatmap.png"), dpi=144, bbox_inches="tight")
    plt.close()

    # GT missing -> 0.0 (by design)
    ans_missing = {
        "chapters": [
            {"chapter": "85 - Electrical", "confidence": 0.60},
            {"chapter": "90 - Instruments", "confidence": 0.35},
            {"chapter": "39 - Plastics", "confidence": 0.05},
        ]
    }
    y = _call_eval(EVAL, user_data, ans_missing, targets)
    print(f"  GT missing reward={y:.4f}")
    plt.figure()
    plt.bar([0], [y])
    plt.title("select_chapters: GT missing")
    plt.ylabel("reward")
    plt.xticks([])
    plt.savefig(os.path.join(plots_dir, "select_chapters_gt_missing.png"), dpi=144, bbox_inches="tight")
    plt.close()

# ============================= select_candidates =============================

def _footrule_only(pred: List[int], gold: List[int]) -> float:
    gold_pos = {v:i for i,v in enumerate(gold)}
    common = [x for x in pred if x in gold_pos]
    n = len(common)
    if n <= 1:
        return 0.0
    ranks_pred = {v:i for i,v in enumerate(common)}
    ranks_gold = {v:i for i,v in enumerate(gold) if v in ranks_pred}
    F = sum(abs(ranks_pred[v]-ranks_gold[v]) for v in ranks_gold)
    Fmax = (n*n)//2
    return 0.0 if Fmax==0 else min(1.0, F/Fmax)

def select_candidates_suite(EVAL, plots_dir: str):
    gold = [0,2,5]
    targets = {"selected_indices": gold}
    user_data = {"task": "select_candidates", "data": {}}

    cases = [
        ("perfect", [0,2,5]),
        ("permute_mid", [2,0,5]),
        ("reverse", [5,2,0]),
        ("subset", [0,2]),
        ("superset+1", [0,2,5,3]),
        ("disjoint3", [1,3,4]),
        ("empty", []),
    ]
    print("\n[select_candidates] canonical cases:")
    print("case          | Jaccard | footrule | reward | pred")
    for name, pred in cases:
        j = len(set(pred) & set(gold))/max(1,len(set(pred)|set(gold)))
        fr = _footrule_only(pred, gold)
        r = _call_eval(EVAL, user_data, {"selected_indices": pred}, targets)
        print(f"{name:12s} | {j:7.3f} | {fr:8.3f} | {r:6.3f} | {pred}")

    # random scatter exploration
    rng_pool = list(range(9))
    Xj, Xo, Y = [], [], []
    for _ in range(2000):
        k = np.random.randint(0,6)
        pred = sorted(np.random.choice(rng_pool, size=k, replace=False).tolist(), key=lambda _: np.random.random())
        r = _call_eval(EVAL, user_data, {"selected_indices": pred}, targets)
        j = len(set(pred) & set(gold))/max(1,len(set(pred)|set(gold)))
        fr = _footrule_only(pred, gold)
        Xj.append(j); Xo.append(fr); Y.append(r)

    # reward vs Jaccard
    plt.figure()
    plt.scatter(Xj, Y, s=9)
    plt.title("select_candidates: reward vs Jaccard (randomized)")
    plt.xlabel("Jaccard")
    plt.ylabel("reward")
    plt.grid(True, alpha=0.3)
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, "select_candidates_scatter_jaccard.png"), dpi=144, bbox_inches="tight")
    plt.close()

    # reward vs order distance
    plt.figure()
    plt.scatter(Xo, Y, s=9)
    plt.title("select_candidates: reward vs order distance (randomized)")
    plt.xlabel("normalized footrule distance")
    plt.ylabel("reward")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, "select_candidates_scatter_order.png"), dpi=144, bbox_inches="tight")
    plt.close()

    # heatmap of mean reward over (Jaccard × order)
    bins_j = np.linspace(0, 1, 21)
    bins_o = np.linspace(0, 1, 21)
    H = np.zeros((len(bins_o)-1, len(bins_j)-1))
    C = np.zeros_like(H)
    for j, o, r in zip(Xj, Xo, Y):
        i = np.digitize(o, bins_o) - 1
        k = np.digitize(j, bins_j) - 1
        if 0 <= i < H.shape[0] and 0 <= k < H.shape[1]:
            H[i,k] += r; C[i,k] += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        A = np.where(C>0, H/C, 0.0)
    plt.figure()
    plt.imshow(A, aspect="auto", origin="lower",
               extent=[bins_j[0], bins_j[-1], bins_o[0], bins_o[-1]])
    plt.colorbar()
    plt.title("select_candidates: mean reward (Jaccard × order bins)")
    plt.xlabel("Jaccard")
    plt.ylabel("normalized footrule distance")
    plt.savefig(os.path.join(plots_dir, "select_candidates_heatmap.png"), dpi=144, bbox_inches="tight")
    plt.close()

# ----------------------------------- main -----------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="reward")
    ap.add_argument("--plots", default="plots")
    args = ap.parse_args()
    EVAL = load_evaluate(args.module)
    score_candidate_suite(EVAL, args.plots)
    select_chapters_suite(EVAL, args.plots)
    select_candidates_suite(EVAL, args.plots)
    print(f"\nSaved plots to: {os.path.abspath(args.plots)}")

if __name__ == "__main__":
    main()
