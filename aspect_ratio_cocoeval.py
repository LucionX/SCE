# aspect_ratio_cocoeval.py

import argparse

import numpy as np
import pandas as pd

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def outside_ratio_bucket(bbox, ratio_min, ratio_max):
    """COCO bbox format: [x, y, w, h]."""
    _, _, w, h = bbox

    if w <= 0 or h <= 0:
        return True

    r = max(w / h, h / w)
    return not (ratio_min <= r < ratio_max)


class AspectRatioCOCOeval(COCOeval):
    """
    Replace area bucketing with aspect-ratio bucketing.

    r = max(w / h, h / w)

    Bucket rules:
    1. GTs outside the bucket are ignored, but retained for matching.
    2. Detections matched to ignored GTs are ignored.
    3. Unmatched detections outside the bucket are ignored.
    4. All other COCO matching and accumulation rules are retained.
    """

    def __init__(
        self,
        cocoGt=None,
        cocoDt=None,
        ratio_min=1.0,
        ratio_max=float("inf"),
    ):
        super().__init__(cocoGt, cocoDt, iouType="bbox")

        self.ratio_min = ratio_min
        self.ratio_max = ratio_max

        # Only evaluate the "all" area setting.
        # Aspect-ratio buckets replace the small/medium/large breakdown.
        self.params.areaRng = [self.params.areaRng[0]]
        self.params.areaRngLbl = ["all"]

    def _prepare(self):
        super()._prepare()

        # Preserve the original crowd ignore flag and add GT bucket filtering.
        for gts in self._gts.values():
            for gt in gts:
                if outside_ratio_bucket(
                    gt["bbox"],
                    self.ratio_min,
                    self.ratio_max,
                ):
                    gt["ignore"] = 1

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        # Use the original COCO matching logic unchanged.
        result = super().evaluateImg(
            imgId,
            catId,
            aRng,
            maxDet,
        )

        if result is None:
            return None

        # dtIds follow the original COCO confidence sorting and maxDet limit.
        # Build the bucket mask in exactly the same detection order.
        dt_outside = np.array(
            [
                outside_ratio_bucket(
                    self.cocoDt.anns[dt_id]["bbox"],
                    self.ratio_min,
                    self.ratio_max,
                )
                for dt_id in result["dtIds"]
            ],
            dtype=bool,
        )

        unmatched = result["dtMatches"] == 0

        # Replace the original unmatched-detection area rule with the
        # aspect-ratio rule. Matched detections retain their COCO ignore flag.
        result["dtIgnore"] = np.where(
            unmatched,
            dt_outside[np.newaxis, :],
            result["dtIgnore"],
        ).astype(bool)

        return result

    def bucket_ap(self):
        """
        COCO AP@[0.50:0.95], area=all, maxDets=100.

        Preserve the original COCO averaging rule:
        ignore precision entries equal to -1;
        return -1 when no valid entries exist.
        """
        p = self.params

        area_index = p.areaRngLbl.index("all")
        max_det_index = p.maxDets.index(100)

        # precision dimensions: [T, R, K, A, M]
        precision = self.eval["precision"][
            :, :, :, area_index, max_det_index
        ]

        valid = precision[precision > -1]

        return float(valid.mean()) if valid.size else -1.0


def evaluate_aspect_ratio(
    gt_json,
    pred_json,
    out_csv="aspect_ratio_ap.csv",
):
    ratio_ranges = [
        ("AR_[1,1.5)", 1.0, 1.5),
        ("AR_[1.5,3)", 1.5, 3.0),
        ("AR_[3,inf)", 3.0, float("inf")),
    ]

    results = []

    for name, r_min, r_max in ratio_ranges:
        # Fresh objects prevent annotation state from crossing buckets.
        coco_gt = COCO(gt_json)
        coco_dt = coco_gt.loadRes(pred_json)

        evaluator = AspectRatioCOCOeval(
            cocoGt=coco_gt,
            cocoDt=coco_dt,
            ratio_min=r_min,
            ratio_max=r_max,
        )

        evaluator.params.imgIds = sorted(coco_gt.getImgIds())

        evaluator.evaluate()
        evaluator.accumulate()

        # Do not call the original summarize(): its fixed 12-metric summary
        # includes area buckets that are not used here.
        ap = evaluator.bucket_ap()

        results.append(
            {
                "bucket": name,
                # Convert valid AP to percent; preserve the -1 sentinel.
                "AP": ap * 100.0 if ap >= 0 else -1.0,
            }
        )

    df = pd.DataFrame(results)

    print("\nAspect-ratio AP@[0.50:0.95] (%; -1 = undefined):")
    print(df.to_string(index=False))

    df.to_csv(out_csv, index=False)
    print(f"\nSaved to: {out_csv}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COCO AP by object aspect ratio")
    parser.add_argument("--gt", required=True, help="COCO ground-truth annotation JSON")
    parser.add_argument("--pred", required=True, help="COCO-format detection prediction JSON")
    parser.add_argument("--out-csv", default="aspect_ratio_ap.csv", help="Output CSV path")
    args = parser.parse_args()
    evaluate_aspect_ratio(args.gt, args.pred, out_csv=args.out_csv)
