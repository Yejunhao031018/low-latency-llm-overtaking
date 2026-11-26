#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tiny numpy MLP baseline for the same SFT data.

This exists to answer a paper-critical question: do we even need an LLM for
this low-dimensional tactical state?  It uses only numpy, so it runs in the
CARLA venv without torch.

    python finetune/train_mlp.py --data dataset/sft --out dataset/sft/mlp

Classes:
    0 hold
    1 inside 3.0
    2 inside 3.5 / speed 1.05
    3 inside 3.5 / speed 1.10
    4 outside
"""
import argparse
import json
import os

import numpy as np


def feature_vector(user_obj):
    def g(k, default=0.0):
        v = user_obj.get(k, default)
        try:
            return float(v)
        except Exception:
            return default
    side = 1.0 if user_obj.get("inside_side", "right") == "right" else -1.0
    return np.array([
        g("ego_v") / 40.0,
        g("lead_v") / 40.0,
        np.clip(g("ahead_m") / 60.0, -2, 2),
        g("gap_m") / 60.0,
        np.clip(g("d_apex_m") / 300.0, -2, 2),
        np.clip(g("kappa") * 150.0, -2, 2),
        np.clip(g("ttc") / 20.0, 0, 2),
        g("actual_off_m") / 7.0,
        g("racing_off_m") / 7.0,
        g("actual_extra_m") / 7.0,
        g("lateral_sep_m") / 7.0,
        side,
    ], dtype=np.float64)


def action_class(a):
    if a.get("action") != "overtake":
        return 0
    extra = float(a.get("extra_inside_m", 0.0))
    speed = float(a.get("speed_factor", 1.0))
    if extra <= 0.5:
        return 0
    if extra < 3.25:
        return 1
    if speed >= 1.08:
        return 3
    if extra > 3.25:
        return 2
    return 1


def load(path):
    X, y, meta = [], [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            msgs = r["messages"]
            user = json.loads(msgs[1]["content"])
            ans = json.loads(msgs[2]["content"])
            X.append(feature_vector(user))
            y.append(action_class(ans))
            meta.append(r.get("meta", {}))
    return np.stack(X), np.array(y, dtype=np.int64), meta


class MLP:
    def __init__(self, d, h=32, k=5, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / d), (d, h))
        self.b1 = np.zeros(h)
        self.W2 = rng.normal(0, np.sqrt(2.0 / h), (h, k))
        self.b2 = np.zeros(k)

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        h = np.maximum(z1, 0)
        logits = h @ self.W2 + self.b2
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        return z1, h, p

    def predict(self, X):
        return self.forward(X)[2].argmax(1)

    def fit(self, X, y, class_weights, epochs=300, lr=1e-3, l2=1e-4, bs=64):
        n = len(X)
        rng = np.random.default_rng(1)
        for ep in range(epochs):
            perm = rng.permutation(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                xb, yb = X[idx], y[idx]
                z1, h, p = self.forward(xb)
                w = class_weights[yb][:, None]
                p[np.arange(len(yb)), yb] -= 1.0
                p *= w
                p /= max(1, len(yb))
                gW2 = h.T @ p + l2 * self.W2
                gb2 = p.sum(0)
                dh = p @ self.W2.T
                dh[z1 <= 0] = 0
                gW1 = xb.T @ dh + l2 * self.W1
                gb1 = dh.sum(0)
                self.W1 -= lr * gW1
                self.b1 -= lr * gb1
                self.W2 -= lr * gW2
                self.b2 -= lr * gb2
        return self


def eval_split(model, X, y, name):
    pred = model.predict(X)
    acc = float((pred == y).mean()) if len(y) else 0.0
    cm = np.zeros((5, 5), dtype=int)
    for a, b in zip(y, pred):
        cm[int(a), int(b)] += 1
    # safety-weighted: false overtake (hold -> overtake) is worse than false hold
    cost = 0
    for a, b in zip(y, pred):
        if a == 0 and b != 0:
            cost += 5
        elif a != 0 and b == 0:
            cost += 1
        elif a != b:
            cost += 2
    print("%s acc=%.3f safety_cost=%d n=%d" % (name, acc, cost, len(y)))
    print(cm)
    return dict(acc=acc, safety_cost=int(cost), n=int(len(y)), confusion=cm.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/sft")
    ap.add_argument("--out", default="dataset/sft/mlp")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=32)
    args = ap.parse_args()

    Xtr, ytr, _ = load(os.path.join(args.data, "train.jsonl"))
    Xva, yva, _ = load(os.path.join(args.data, "val.jsonl"))
    Xte, yte, _ = load(os.path.join(args.data, "test.jsonl"))
    mean = Xtr.mean(0)
    std = Xtr.std(0) + 1e-6
    Xtr = (Xtr - mean) / std
    Xva = (Xva - mean) / std
    Xte = (Xte - mean) / std

    counts = np.bincount(ytr, minlength=5).astype(float)
    class_weights = np.where(counts > 0, len(ytr) / (5.0 * np.maximum(counts, 1)), 0.0)
    model = MLP(Xtr.shape[1], h=args.hidden, seed=0)
    model.fit(Xtr, ytr, class_weights, epochs=args.epochs)
    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "mlp_model.npz"),
             W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2,
             mean=mean, std=std)
    metrics = {
        "train": eval_split(model, Xtr, ytr, "train"),
        "val": eval_split(model, Xva, yva, "val"),
        "test": eval_split(model, Xte, yte, "test"),
        "class_counts": counts.astype(int).tolist(),
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
