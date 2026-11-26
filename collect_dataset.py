#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch data collection for the tactical-LLM paper.

Examples:
    # 5 safe stub episodes with randomized start gap / lead behavior / labels
    python collect_dataset.py --episodes 5 --policy stub --label --max-labels 4 \
        --out dataset/pilot

    # 3 random-exploration episodes (hard negatives / near misses)
    python collect_dataset.py --episodes 3 --policy random --stop-on-collision \
        --out dataset/random

    # one fixed baseline episode (same as the paper's nominal setting)
    python collect_dataset.py --episodes 1 --fixed --start-gap 26 \
        --lead-policy constant --policy stub --out dataset/nominal

Requires a running CARLA Town04 server (port 2000 by default).
"""
import argparse
import json
import os
import random
import sys

import carla

from dataset_runner import OUT_DEFAULT, run_episode


def sample_config(rng, args, k):
    gap = rng.uniform(args.min_gap, args.max_gap)
    lead_f = rng.uniform(args.min_lead_factor, args.max_lead_factor)
    ego_f = rng.uniform(args.min_ego_factor, args.max_ego_factor)
    lead_policy = rng.choice(["constant", "variable", "defensive"])
    return dict(
        policy=args.policy,
        lead_policy=lead_policy,
        seed=args.seed + k,
        start_gap_m=round(gap, 1),
        lead_speed_factor=round(lead_f, 3),
        ego_speed_factor=round(ego_f, 3),
        weather=args.weather,
        label=args.label,
        max_labels=args.max_labels,
        label_every_s=args.label_every_s,
        label_horizon_s=args.label_horizon_s,
        stop_on_collision=args.stop_on_collision,
        max_time_s=args.max_time_s,
        tick_shield=(args.policy != "random" and not args.no_tick_shield),
        no_rendering=True,
        corner_radius=args.corner_radius, racing_off_in=args.racing_off_in,
        off_cap_m=args.off_cap,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--policy", default="stub", choices=["stub", "hold", "random", "openai"])
    ap.add_argument("--fixed", action="store_true",
                    help="do not randomize gap/speed/lead policy")
    ap.add_argument("--lead-policy", default="constant",
                    choices=["constant", "variable", "defensive"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start-gap", type=float, default=26.0)
    ap.add_argument("--lead-factor", type=float, default=0.90)
    ap.add_argument("--ego-factor", type=float, default=1.05)
    ap.add_argument("--min-gap", type=float, default=12.0)
    ap.add_argument("--max-gap", type=float, default=45.0)
    ap.add_argument("--min-lead-factor", type=float, default=0.86)
    ap.add_argument("--max-lead-factor", type=float, default=0.94)
    ap.add_argument("--min-ego-factor", type=float, default=1.00)
    ap.add_argument("--max-ego-factor", type=float, default=1.06)
    ap.add_argument("--weather", default="ClearNoon")
    ap.add_argument("--label", action="store_true",
                    help="after collection, run the separate counterfactual labeler")
    ap.add_argument("--max-labels", type=int, default=4,
                    help="kept for compatibility; label selection uses --label-states")
    ap.add_argument("--label-every-s", type=float, default=8.0)
    ap.add_argument("--label-horizon-s", type=float, default=6.0)
    ap.add_argument("--label-states", type=int, default=4,
                    help="counterfactual states per episode")
    ap.add_argument("--stop-on-collision", action="store_true")
    ap.add_argument("--max-time-s", type=float, default=120.0)
    ap.add_argument("--no-tick-shield", action="store_true",
                    help="disable the per-tick shield (use for failure/hard-negative data)")
    ap.add_argument("--corner-radius", type=float, default=188.0)
    ap.add_argument("--racing-off-in", type=float, default=3.3)
    ap.add_argument("--off-cap", type=float, default=0.0)
    args = ap.parse_args()

    port = int(os.environ.get("CARLA_PORT", "2000"))
    client = carla.Client("127.0.0.1", port)
    client.set_timeout(30.0)
    world = client.get_world()
    print("connected:", client.get_server_version(), "| map:", world.get_map().name)
    os.makedirs(args.out, exist_ok=True)
    index_path = os.path.join(args.out, "index.jsonl")
    rng = random.Random(args.seed)

    for k in range(args.episodes):
        if args.fixed:
            cfg = dict(
                policy=args.policy, lead_policy=args.lead_policy, seed=args.seed + k,
                start_gap_m=args.start_gap, lead_speed_factor=args.lead_factor,
                ego_speed_factor=args.ego_factor, weather=args.weather,
                label=args.label, max_labels=args.max_labels,
                label_every_s=args.label_every_s, label_horizon_s=args.label_horizon_s,
                stop_on_collision=args.stop_on_collision,
                max_time_s=args.max_time_s,
                tick_shield=(args.policy != "random" and not args.no_tick_shield),
                no_rendering=True,
                corner_radius=args.corner_radius, racing_off_in=args.racing_off_in,
                off_cap_m=args.off_cap,
            )
        else:
            cfg = sample_config(rng, args, k)

        eid = "ep%03d_%s_%s_gap%02.0f_seed%d" % (
            k, cfg["policy"], cfg["lead_policy"], cfg["start_gap_m"], cfg["seed"])
        print("=" * 70)
        print("episode", k + 1, "/", args.episodes, eid, cfg)
        summary, out_dir = run_episode(world, cfg, args.out, episode_id=eid)
        with open(index_path, "a") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        print("summary:", json.dumps(summary["outcome"], ensure_ascii=False))
        print("saved:", out_dir)

    print("=" * 70)
    print("index:", index_path)
    if args.label:
        from label_dataset import label_dataset
        label_dataset(args.out, max_states=args.label_states,
                      horizon=args.label_horizon_s, port=port)


if __name__ == "__main__":
    main()
