#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert collected episodes + counterfactual labels into SFT/DPO datasets.

    python finetune/convert.py --dataset dataset/pilot --out dataset/sft

Outputs:
    dataset/sft/train.jsonl / val.jsonl / test.jsonl   (chat format)
    dataset/sft/dpo.jsonl                              (prompt/chosen/rejected)
    dataset/sft/stats.json

Rules:
- A decision with a counterfactual label uses the best labeled action (planner).
- Otherwise a stub decision is used only near T2 or when it is an overtake;
  plain holds outside the window are heavily down-sampled.
- Random-policy episodes are only used where they have labels (their raw actions
  are unsafe by construction).
- Labels whose best action still has a high cost are skipped by default, because
  they usually mean the replay itself was unreliable or every candidate crashed.
"""
import argparse
import glob
import json
import os
import random
from collections import Counter, defaultdict

SYSTEM = (
    "You are a racing tactical planner for a car approaching a corner. "
    "Answer ONLY a JSON object with keys: action (overtake|hold), "
    "extra_inside_m (float, 0..3.5; 0 = racing line, positive = further toward the corner inside), "
    "speed_factor (float, 0.9..1.1), reason (short text). "
    "Never request an inside pass with less than 3.0 m extra offset."
)


def _f(x, n=2):
    try:
        return round(float(x), n)
    except Exception:
        return x


def action_json(a):
    act = a.get("action", "hold")
    # A hold must not carry a lateral request: the executor ignores it, and
    # teaching "hold + 3.5 m" only creates contradictory targets.
    extra = _f(a.get("extra_inside_m", 0.0), 2) if act == "overtake" else 0.0
    sf = _f(a.get("speed_factor", 1.0), 2) if act == "overtake" else 1.0
    return json.dumps({
        "action": act,
        "extra_inside_m": extra,
        "speed_factor": sf,
        "reason": a.get("reason", ""),
    }, ensure_ascii=False, separators=(",", ":"))


def make_reason(state, a):
    gap = _f(state.get("ahead_m", 0.0), 1)
    d = _f(state.get("d_apex_m", 0.0), 0)
    sep = state.get("lateral_sep_m")
    if a.get("action") != "overtake":
        if gap > 10:
            return "already ahead by %.1f m; hold" % gap
        return "gap=%.1f m d_apex=%.0f m; hold keeps the safety margin" % (gap, d)
    if sep is None:
        return "gap=%.1f m d_apex=%.0f m; inside %.1f m" % (
            gap, d, _f(a.get("extra_inside_m", 0.0), 1))
    return "gap=%.1f m d_apex=%.0f m sep=%.1f m; inside %.1f m keeps clearance" % (
        gap, d, abs(float(sep)), _f(a.get("extra_inside_m", 0.0), 1))


def observation(rec, prev_action="hold"):
    s = rec.get("state", {})
    keys = ["ego_v", "lead_v", "ahead_m", "gap_m", "d_apex_m", "kappa", "ttc",
            "actual_off_m", "racing_off_m", "actual_extra_m", "lateral_sep_m",
            "inside_side"]
    obs = {k: (_f(s.get(k), 4) if isinstance(s.get(k), float) else s.get(k)) for k in keys}
    # The tactical problem is partially observed: the same lateral offset can be
    # an attack entry or a normal racing line.  Feed back the previous decision
    # so "continue the attack" is distinguishable from "start one".
    obs["prev_action"] = prev_action
    obs["scenario"] = {
        "track": "Town04", "corner": "T2",
        "lead_policy": rec.get("lead_policy"),
    }
    return json.dumps(obs, ensure_ascii=False, separators=(",", ":"))


def load_labels(root):
    by_key = {}
    path = os.path.join(root, "labels.jsonl")
    if not os.path.exists(path):
        return by_key
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_key[(r["episode_id"], round(float(r["t"]), 2))] = r
    return by_key


def iter_episodes(root):
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        sd = os.path.join(d, "summary.json")
        dd = os.path.join(d, "decisions.jsonl")
        if os.path.isdir(d) and os.path.exists(sd) and os.path.exists(dd):
            with open(sd) as f:
                summary = json.load(f)
            yield d, summary, dd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset/pilot")
    ap.add_argument("--out", default="dataset/sft")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--hold-keep", type=float, default=0.20,
                    help="fraction of non-labeled holds to keep")
    ap.add_argument("--max-cost", type=float, default=500.0,
                    help="skip labeled states whose best action still costs more")
    ap.add_argument("--min-center", type=float, default=2.5,
                    help="skip labels whose best action has min center distance below this")
    ap.add_argument("--min-sep", type=float, default=1.8,
                    help="skip labels whose best action has min close lateral separation below this")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    labels = load_labels(args.dataset)

    episodes = []
    for d, summary, dd in iter_episodes(args.dataset):
        coll = summary.get("outcome", {}).get("collisions", {}) or {}
        if sum(int(v) for v in coll.values()) > 10:
            print("skip colliding/corrupted episode:", os.path.basename(d), coll)
            continue
        episodes.append((d, summary, dd))
    if not episodes:
        print("no clean episodes")
        return

    ids = [os.path.basename(d) for d, _, _ in episodes]
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(1, int(round(n * args.test_ratio))) if n >= 3 else 0
    n_val = max(1, int(round(n * args.val_ratio))) if n >= 3 else 0
    test_ids = set(ids[:n_test])
    val_ids = set(ids[n_test:n_test + n_val])
    split_of = {eid: ("test" if eid in test_ids else "val" if eid in val_ids else "train")
                for eid in ids}

    out = defaultdict(list)
    dpo = []
    stats = Counter()
    label_stats = Counter()

    for d, summary, dd in episodes:
        eid = os.path.basename(d)
        split = split_of[eid]
        with open(dd) as f:
            rows = [json.loads(line) for line in f]
        for j, rec in enumerate(rows):
            if j == 0:
                prev_action = "hold"
            else:
                prev_rec = rows[j - 1]
                prev_action = (prev_rec.get("filtered") or prev_rec.get("raw")
                               or {}).get("action", "hold")
            t = round(float(rec["t"]), 2)
            lab = labels.get((eid, t))
            source = None
            if lab is not None:
                best = lab["best"]
                cost = float(best.get("cost", 1e9))
                lout = best.get("outcome", {}) or {}
                if cost > args.max_cost:
                    stats["skip_high_cost_label"] += 1
                    continue
                if lout.get("min_center") is not None and float(lout["min_center"]) < args.min_center:
                    stats["skip_low_center_label"] += 1
                    continue
                if (lout.get("min_sep_close") is not None
                        and float(lout["min_sep_close"]) < args.min_sep):
                    stats["skip_low_sep_label"] += 1
                    continue
                target = dict(best["action"])
                target["reason"] = make_reason(rec["state"], target)
                source = "planner"
                cost_gap = lab.get("cost_gap")
                if cost_gap is not None and float(cost_gap) > 20.0:
                    cands = sorted(lab["candidates"], key=lambda x: x["cost"])
                    worst = cands[-1]
                    if worst["cost"] - best["cost"] > 20.0:
                        prompt = SYSTEM + "\n\nObservation:\n" + observation(rec, prev_action)
                        dpo.append({
                            "prompt": prompt,
                            "chosen": action_json(target),
                            "rejected": action_json(worst["action"]),
                            "meta": {"episode_id": eid, "t": t,
                                     "chosen_cost": best["cost"],
                                     "rejected_cost": worst["cost"]},
                        })
                        stats["dpo_pairs"] += 1
            else:
                if rec.get("policy") == "random":
                    continue
                target = dict(rec.get("filtered") or rec.get("raw") or {})
                if not target:
                    continue
                if target.get("action") != "overtake":
                    d_apex = abs(float(rec["state"].get("d_apex_m", 1e9)))
                    if d_apex > 220.0 and rng.random() > args.hold_keep:
                        stats["drop_far_hold"] += 1
                        continue
                target["reason"] = make_reason(rec["state"], target)
                source = "stub"
                cost = None
                cost_gap = None

            msg = {"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": observation(rec, prev_action)},
                {"role": "assistant", "content": action_json(target)},
            ], "meta": {"episode_id": eid, "t": t, "source": source,
                        "split": split, "cost": cost, "cost_gap": cost_gap}}
            out[split].append(msg)
            stats["sft_" + split] += 1
            stats["source_" + str(source)] += 1
            label_stats[target.get("action", "?")] += 1

    for split in ("train", "val", "test"):
        path = os.path.join(args.out, split + ".jsonl")
        with open(path, "w") as f:
            for r in out[split]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(split, len(out[split]), "->", path)

    with open(os.path.join(args.out, "dpo.jsonl"), "w") as f:
        for r in dpo:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("dpo", len(dpo), "->", os.path.join(args.out, "dpo.jsonl"))

    meta = {"episodes": ids, "splits": split_of,
            "counts": dict(stats), "actions": dict(label_stats),
            "labels_total": len(labels)}
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
