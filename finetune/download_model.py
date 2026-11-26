#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download Qwen2.5-1.5B-Instruct from ModelScope (works when HF is reset)."""
import argparse
import os

from modelscope import snapshot_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--cache", default="models")
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    path = snapshot_download(args.model, cache_dir=args.cache)
    print("MODEL_PATH=" + path)


if __name__ == "__main__":
    main()
