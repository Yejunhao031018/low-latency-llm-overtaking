#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Counterfactual labeling for collected episodes.

This is intentionally a separate pass:
  1. collect_dataset.py records clean episodes + snapshots near T2;
  2. label_dataset.py replays each snapshot with several candidate actions
     and writes a cost-to-go label.

Why separate?  CARLA's transform/velocity restore is approximate.  Doing the
rollouts inline corrupts the actual episode (we observed exactly this), so the
data-collection run and the counterfactual run must be isolated.

Usage:
    python label_dataset.py --dataset dataset/pilot --max-states 20
    python label_dataset.py --dataset dataset/pilot --episode ep000_...
"""
import argparse
import glob
import json
import os

import carla

from dataset_runner import Episode, OUT_DEFAULT


def _load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def _iter_episode_dirs(root):
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "decisions.jsonl")):
            yield d


def _select_records(records, max_states):
    recs = [r for r in records if r.get("snapshot")]
    # Prefer states in the actual overtake window, then sample evenly.
    recs.sort(key=lambda r: abs(r.get("state", {}).get("d_apex_m", 1e9) - 0.0))
    if len(recs) > max_states:
        idx = [int(i * (len(recs) - 1) / max(1, max_states - 1)) for i in range(max_states)]
        recs = [recs[i] for i in idx]
    return recs


def label_dataset(root=OUT_DEFAULT, max_states=20, horizon=4.0,
                  out_path=None, episode_filter=None, port=2000):
    client = carla.Client("127.0.0.1", port)
    client.set_timeout(30.0)
    world = client.get_world()
    print("connected:", client.get_server_version(), "| map:", world.get_map().name)

    episode_dirs = list(_iter_episode_dirs(root))
    if episode_filter:
        episode_dirs = [d for d in episode_dirs if episode_filter in os.path.basename(d)]
    if not episode_dirs:
        print("no episode dirs under", root)
        return

    out_path = out_path or os.path.join(root, "labels.jsonl")
    n_written = 0
    with open(out_path, "w") as fout:
        for edir in episode_dirs:
            summary = _load_json(os.path.join(edir, "summary.json"), {})
            coll = summary.get("outcome", {}).get("collisions", {}) or {}
            if sum(int(v) for v in coll.values()) > 10:
                print("skip corrupted/colliding episode:", os.path.basename(edir), coll)
                continue
            cfg = summary.get("config") or {}
            cfg = dict(cfg)
            cfg["record_snapshot"] = True
            cfg["label_horizon_s"] = horizon
            eid = summary.get("episode_id", os.path.basename(edir))
            print("=" * 70)
            print("episode", eid, "|", cfg.get("lead_policy"), "|", cfg)
            ep = Episode(world, cfg, eid)
            try:
                records = []
                with open(os.path.join(edir, "decisions.jsonl")) as f:
                    for line in f:
                        records.append(json.loads(line))
                selected = _select_records(records, max_states)
                print("snapshot states:", len(selected), "/", len(records))
                for r in selected:
                    ep.restore_json(r["snapshot"])
                    # CARLA applies set_transform on the next tick; read after it
                    # so the labeler does not accidentally label the spawn pose.
                    ep.world.tick()
                    state = ep.read_state(update=False)
                    cands = ep.label_state(state)
                    best = cands[0]
                    rec = dict(episode_id=eid, t=r["t"],
                               state=r["state"],
                               executed=r.get("executed"),
                               candidates=cands,
                               best=best,
                               best_action=best["action"],
                               best_cost=best["cost"],
                               cost_gap=(cands[1]["cost"] - best["cost"] if len(cands) > 1 else None))
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_written += 1
                    print("  t=%6.2f d_apex=%7.1f gap=%6.1f -> best=%s cost=%.1f" % (
                        r["t"], state["d_apex_m"], state["ahead_m"],
                        best["name"], best["cost"]))
            finally:
                ep.close()
    print("=" * 70)
    print("labels:", out_path, "n=", n_written)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=OUT_DEFAULT)
    ap.add_argument("--max-states", type=int, default=20, help="per episode")
    ap.add_argument("--horizon", type=float, default=6.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--episode", default=None, help="substring filter on episode id")
    ap.add_argument("--port", type=int, default=int(os.environ.get("CARLA_PORT", "2000")))
    args = ap.parse_args()
    label_dataset(args.dataset, args.max_states, args.horizon,
                  args.out, args.episode, args.port)


if __name__ == "__main__":
    main()
