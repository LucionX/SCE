#!/usr/bin/env python3
"""
COCO Evaluation Script
Usage:
  python eval_coco.py --pred runs/detect/val/predictions.json \
                      --gt data/coco/annotations/instances_val2017.json \
                      --decimals 6
  python eval_coco.py -p pred.json -g gt.json -d 3 --percent
"""

import argparse
import json
import sys

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_args():
    parser = argparse.ArgumentParser(
        description="COCO Evaluation with pycocotools",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pred", "-p",
        type=str,
        required=True,
        help="Path to predictions.json (model output)",
    )
    parser.add_argument(
        "--gt", "-g",
        type=str,
        required=True,
        help="Path to ground-truth COCO annotation json (e.g. instances_val2017.json)",
    )
    parser.add_argument(
        "--iou_type", "-t",
        type=str,
        default="bbox",
        choices=["bbox", "segm", "keypoints"],
        help="IoU type for evaluation",
    )
    parser.add_argument(
        "--decimals", "-d",
        type=int,
        default=6,
        help="Number of decimal places to round (default: 6)",
    )
    parser.add_argument(
        "--percent",
        action="store_true",
        default=False,
        help="If set, multiply by 100 (output as percentage, e.g. 37.945); "
             "otherwise output as ratio (e.g. 0.379450)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save metrics json (default: print only, no file)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Load ----
    print(f"[INFO] Loading GT: {args.gt}")
    try:
        coco_gt = COCO(args.gt)
    except Exception as e:
        print(f"[ERROR] Failed to load GT: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loading predictions: {args.pred}")
    try:
        coco_dt = coco_gt.loadRes(args.pred)
    except Exception as e:
        print(f"[ERROR] Failed to load predictions: {e}", file=sys.stderr)
        sys.exit(1)

    # ---- Evaluate ----
    print(f"[INFO] Running COCOeval (iouType='{args.iou_type}') ...")
    e = COCOeval(coco_gt, coco_dt, iouType=args.iou_type)
    e.evaluate()
    e.accumulate()
    e.summarize()

    s = e.stats  # 12-element array, values in [0, 1]

    # ---- Format metrics ----
    scale = 100.0 if args.percent else 1.0
    unit = "%" if args.percent else ""

    metrics = {
        "mAP_0.5_0.95": round(float(s[0]) * scale, args.decimals),
        "mAP_50":       round(float(s[1]) * scale, args.decimals),
        "mAP_75":       round(float(s[2]) * scale, args.decimals),
        "mAP_small":    round(float(s[3]) * scale, args.decimals),
        "mAP_medium":   round(float(s[4]) * scale, args.decimals),
        "mAP_large":    round(float(s[5]) * scale, args.decimals),
        "AR_1":         round(float(s[6]) * scale, args.decimals),
        "AR_10":        round(float(s[7]) * scale, args.decimals),
        "AR_100":       round(float(s[8]) * scale, args.decimals),
        "AR_small":     round(float(s[9]) * scale, args.decimals),
        "AR_medium":    round(float(s[10]) * scale, args.decimals),
        "AR_large":     round(float(s[11]) * scale, args.decimals),
    }

    # ---- Print ----
    print("\n" + "=" * 50)
    print(f"  COCO Metrics ({args.iou_type})")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:>15s}: {v}{unit}")
    print("=" * 50)

    # ---- Save ----
    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[INFO] Metrics saved to: {args.output}")


if __name__ == "__main__":
    main()