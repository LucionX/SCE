"""Size-only scale response from baseline CIoU prediction JSON.

Matching, area endpoints, crowd handling and maxDets are provided by COCOeval.
Response uses original IoU=0.5 matches, then scales BOTH boxes without clipping.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


EPS = 1e-7
SCALES = (2.0, 0.5)
SIZE_ORDER = ("small", "medium", "large")
RESPONSE_COLUMNS = ("iou_o", "iou_s", "delta", "abs_gap", "M")
RAW_COLUMNS = [
    "image_id", "category_id", "gt_id", "dt_id", "score",
    "size_bucket", "gt_area", "scale", *RESPONSE_COLUMNS,
]


def xywh_to_xyxy(bbox):
    # COCO xywh uses the TOP-LEFT corner, not the center.
    x, y, w, h = bbox
    return np.array([x, y, x + w, y + h], dtype=np.float64)


def bbox_iou_xyxy(box1, box2, eps=EPS):
    # Used only for response; COCOeval computes matching IoUs itself.
    intersection_wh = np.maximum(
        np.minimum(box1[2:], box2[2:])
        - np.maximum(box1[:2], box2[:2]),
        0.0,
    )
    intersection = float(np.prod(intersection_wh))
    area1 = float(np.prod(np.maximum(box1[2:] - box1[:2], 0.0)))
    area2 = float(np.prod(np.maximum(box2[2:] - box2[:2], 0.0)))
    return intersection / (area1 + area2 - intersection + eps)


def scale_box_center(box, scale):
    center = (box[:2] + box[2:]) / 2.0
    half_wh = (box[2:] - box[:2]) * scale / 2.0
    # No image-boundary clipping.
    return np.concatenate([center - half_wh, center + half_wh])


def compute_response(gt_bbox, dt_bbox, sce_a=2.0):
    gt_box = xywh_to_xyxy(gt_bbox)
    dt_box = xywh_to_xyxy(dt_bbox)
    iou_o = bbox_iou_xyxy(dt_box, gt_box)

    rows = []
    for scale in SCALES:
        iou_s = bbox_iou_xyxy(
            scale_box_center(dt_box, scale),
            scale_box_center(gt_box, scale),
        )
        delta = iou_s - iou_o
        rows.append({
            "scale": scale,
            "iou_o": iou_o,
            "iou_s": iou_s,
            "delta": delta,
            "abs_gap": abs(delta),
            # This is m_raw in sce_loss.py, without batch normalization.
            "M": (min(iou_o, iou_s) / (max(iou_o, iou_s) + EPS)) ** sce_a,
        })
    return rows


def collect_responses(evaluator, sce_a=2.0):
    p = evaluator.params
    threshold_index = int(np.flatnonzero(np.isclose(p.iouThrs, 0.5))[0])
    area_to_size = {
        tuple(area_range): label
        for area_range, label in zip(p.areaRng, p.areaRngLbl)
        if label in SIZE_ORDER
    }

    records = []
    for result in evaluator.evalImgs:
        if result is None:
            continue
        size = area_to_size.get(tuple(result["aRng"]))
        if size is None:
            continue

        matches = result["dtMatches"][threshold_index]
        ignored = np.asarray(result["dtIgnore"][threshold_index], dtype=bool)
        valid_indices = np.flatnonzero((matches > 0) & ~ignored)

        for index in valid_indices:
            gt_id = int(matches[index])
            dt_id = int(result["dtIds"][index])
            gt = evaluator.cocoGt.anns[gt_id]
            dt = evaluator.cocoDt.anns[dt_id]
            metadata = {
                "image_id": result["image_id"],
                "category_id": result["category_id"],
                "gt_id": gt_id,
                "dt_id": dt_id,
                "score": dt["score"],
                "size_bucket": size,
                "gt_area": gt["area"],
            }
            # Keep the original pair for both scales, including iou_s == 0.
            for response in compute_response(gt["bbox"], dt["bbox"], sce_a):
                records.append({**metadata, **response})

    # Explicit columns also support zero valid matches.
    return pd.DataFrame(records, columns=RAW_COLUMNS)


def summarize_responses(raw_df):
    rows = []
    for size in SIZE_ORDER:
        for scale in SCALES:
            subset = raw_df.loc[
                (raw_df["size_bucket"] == size) & (raw_df["scale"] == scale)
            ]
            row = {"size_bucket": size, "scale": scale, "N": len(subset)}
            for column in RESPONSE_COLUMNS:
                # -1 denotes undefined response, not zero response.
                row[column] = float(subset[column].mean()) if len(subset) else -1.0
            rows.append(row)
    return pd.DataFrame(rows)


def analyze_scale_response(
    gt_json,
    pred_json,
    out_dir="ciou_scale_response",
    sce_a=2.0,
):
    coco_gt = COCO(str(gt_json))
    # Preserve the installed COCO loader's behavior, including empty JSON lists.
    coco_dt = coco_gt.loadRes(str(pred_json))

    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    # Leave all official parameters unchanged:
    # imgIds, catIds, areaRng, iouThrs, maxDets and useCats.
    evaluator.evaluate()

    # Only matching results are needed; AP accumulation is not required.
    raw_df = collect_responses(evaluator, sce_a=sce_a)
    size_df = summarize_responses(raw_df)

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "scale_response_raw.csv"
    size_path = output_dir / "size_response.csv"
    raw_df.to_csv(raw_path, index=False)
    size_df.to_csv(size_path, index=False)

    print("\nSize response: original IoU=0.5 matches; scales=2.0/0.5")
    print("M=m_raw; -1=undefined; response values are not percentages.")
    print(size_df.to_string(index=False))
    print(f"\nSaved: {raw_path}\nSaved: {size_path}")
    return raw_df, size_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", required=True, help="COCO GT annotation JSON")
    parser.add_argument("--pred", required=True, help="Baseline CIoU prediction JSON")
    parser.add_argument("--out-dir", default="ciou_scale_response")
    parser.add_argument("--sce-a", type=float, default=2.0)
    args = parser.parse_args()
    analyze_scale_response(args.gt, args.pred, args.out_dir, args.sce_a)
