import math
import torch

def bbox_scale_iou(
        box1,
        box2,
        xywh=True,

        GIoU=False,
        DIoU=False,
        CIoU=False,
        EIoU=False,
        SIoU=False,

        # alpha-IoU
        alpha=1.0,

        # Focal
        use_focal=False,
        focal_gamma=0.5,

        # SCE
        use_sce=False,
        lambda_sce=1.0,
        sce_a=2.0,
        sce_normalize=False,
    
        # normalized progress [0, 1]
        progress=0.0,

        # scale schedule
        s_max=2.0,
        s_min=0.5,
        schedule='fraction',

        eps=1e-7):

    # =========================================================
    # 1. Original boxes
    # =========================================================
    if xywh:
        x1, y1, w1, h1 = box1.chunk(4, -1)
        x2, y2, w2, h2 = box2.chunk(4, -1)

        b1_x1 = x1 - w1 / 2
        b1_x2 = x1 + w1 / 2
        b1_y1 = y1 - h1 / 2
        b1_y2 = y1 + h1 / 2

        b2_x1 = x2 - w2 / 2
        b2_x2 = x2 + w2 / 2
        b2_y1 = y2 - h2 / 2
        b2_y2 = y2 + h2 / 2

    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

        w1 = (b1_x2 - b1_x1).clamp(min=eps)
        h1 = (b1_y2 - b1_y1).clamp(min=eps)
        w2 = (b2_x2 - b2_x1).clamp(min=eps)
        h2 = (b2_y2 - b2_y1).clamp(min=eps)

        x1 = (b1_x1 + b1_x2) / 2
        y1 = (b1_y1 + b1_y2) / 2
        x2 = (b2_x1 + b2_x2) / 2
        y2 = (b2_y1 + b2_y2) / 2

    # =========================================================
    # 2. Raw original IoU
    # =========================================================
    inter = (
        (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0)
        *
        (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)
    )

    union = (
        w1 * h1
        + w2 * h2
        - inter
    )

    # IMPORTANT:
    # raw IoU for SCE
    iou_raw = inter / (union + eps)

    # alpha IoU only belongs to base IoU loss
    iou_alpha = torch.pow(iou_raw, alpha)

    # =========================================================
    # 3. Original geometric penalties
    # =========================================================
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)

    c2 = torch.pow(cw ** 2 + ch ** 2, alpha) + eps

    rho2 = torch.pow(
        (
            (
                b2_x1 + b2_x2
                - b1_x1 - b1_x2
            ) ** 2
            +
            (
                b2_y1 + b2_y2
                - b1_y1 - b1_y2
            ) ** 2
        ) / 4,
        alpha
    )

    # =========================================================
    # 4. Base IoU score
    # =========================================================
    if CIoU:

        v = (
            4 / math.pi ** 2
            * (
                torch.atan(w2 / h2)
                - torch.atan(w1 / h1)
            ).pow(2)
        )

        with torch.no_grad():
            alpha_ciou = (
                v
                /
                (v - iou_alpha + 1 + eps)
            )

        score = (
            iou_alpha
            - rho2 / c2
            - torch.pow(v * alpha_ciou + eps, alpha)
        )

    elif EIoU:

        rho_w2 = (
            (b2_x2 - b2_x1)
            - (b1_x2 - b1_x1)
        ) ** 2

        rho_h2 = (
            (b2_y2 - b2_y1)
            - (b1_y2 - b1_y1)
        ) ** 2

        cw2 = torch.pow(cw ** 2 + eps, alpha)
        ch2 = torch.pow(ch ** 2 + eps, alpha)

        score = (
            iou_alpha
            - rho2 / c2
            - rho_w2 / cw2
            - rho_h2 / ch2
        )

    elif SIoU:

        s_cw = (
            b2_x1 + b2_x2
            - b1_x1 - b1_x2
        ) * 0.5 + eps

        s_ch = (
            b2_y1 + b2_y2
            - b1_y1 - b1_y2
        ) * 0.5 + eps

        sigma = torch.sqrt(
            s_cw ** 2 + s_ch ** 2
        )

        sin_alpha_1 = torch.abs(s_cw) / sigma
        sin_alpha_2 = torch.abs(s_ch) / sigma

        threshold = math.sqrt(2) / 2

        sin_alpha = torch.where(
            sin_alpha_1 > threshold,
            sin_alpha_2,
            sin_alpha_1
        )

        angle_cost = torch.cos(
            torch.arcsin(sin_alpha) * 2
            - math.pi / 2
        )

        rho_x = (s_cw / cw) ** 2
        rho_y = (s_ch / ch) ** 2

        # Do NOT call this gamma:
        # gamma is reserved for focal.
        siou_gamma = angle_cost - 2

        distance_cost = (
            2
            - torch.exp(siou_gamma * rho_x)
            - torch.exp(siou_gamma * rho_y)
        )

        omega_w = (
            torch.abs(w1 - w2)
            / torch.maximum(w1, w2)
        )

        omega_h = (
            torch.abs(h1 - h2)
            / torch.maximum(h1, h2)
        )

        shape_cost = (
            torch.pow(
                1 - torch.exp(-omega_w), 4
            )
            +
            torch.pow(
                1 - torch.exp(-omega_h), 4
            )
        )

        score = (
            iou_alpha
            - torch.pow(
                0.5
                * (distance_cost + shape_cost)
                + eps,
                alpha
            )
        )

    elif DIoU:

        score = (
            iou_alpha
            - rho2 / c2
        )

    elif GIoU:

        c_area = cw * ch + eps

        score = (
            iou_alpha
            - torch.pow(
                (c_area - union) / c_area + eps,
                alpha
            )
        )

    else:
        score = iou_alpha

    # =========================================================
    # 5. Base regression loss
    # =========================================================
    base_loss = 1.0 - score

    # =========================================================
    # 6. Optional Focal weighting
    # =========================================================
    if use_focal:

        focal_weight = torch.pow(
            iou_raw.clamp(min=0.0, max=1.0),
            focal_gamma
        ).detach()

        effective_base_loss = (
            focal_weight * base_loss
        )

    else:

        effective_base_loss = base_loss

    # =========================================================
    # 7. Optional SCE
    # =========================================================
    if use_sce:
        scale_ratio = get_sce_scale(
            progress=progress,
            s_max=s_max,
            s_min=s_min,
            schedule=schedule,
        )

        # scale both boxes around their own centers
        ww1 = w1 * scale_ratio
        hh1 = h1 * scale_ratio

        ww2 = w2 * scale_ratio
        hh2 = h2 * scale_ratio

        sb1_x1 = x1 - ww1 / 2
        sb1_x2 = x1 + ww1 / 2
        sb1_y1 = y1 - hh1 / 2
        sb1_y2 = y1 + hh1 / 2

        sb2_x1 = x2 - ww2 / 2
        sb2_x2 = x2 + ww2 / 2
        sb2_y1 = y2 - hh2 / 2
        sb2_y2 = y2 + hh2 / 2

        inter_s = (
            (
                sb1_x2.minimum(sb2_x2)
                - sb1_x1.maximum(sb2_x1)
            ).clamp(0)
            *
            (
                sb1_y2.minimum(sb2_y2)
                - sb1_y1.maximum(sb2_y1)
            ).clamp(0)
        )

        union_s = (
            ww1 * hh1
            + ww2 * hh2
            - inter_s
        )

        # IMPORTANT:
        # raw scaled IoU remains in computation graph
        iou_s = inter_s / (union_s + eps)

        # SCR
        gap = torch.abs(
            iou_raw - iou_s
        )

        # SSM
        iou_min = torch.minimum(
            iou_raw,
            iou_s
        )

        iou_max = torch.maximum(
            iou_raw,
            iou_s
        )

        m_raw = torch.pow(
            iou_min
            / (iou_max + eps),
            sce_a
        ).detach()

        if sce_normalize:
            m = (
                m_raw
                / (m_raw.mean() + eps)
            )
        else:
            m = m_raw

        # ---------------------------------------------
        # SCE wrapper
        #
        # focal off:
        # M * (L_base + lambda * SCR)
        #
        # focal on:
        # M * (w_focal * L_base + lambda * SCR)
        # ---------------------------------------------
        loss = (
            m
            * (
                effective_base_loss
                + lambda_sce * gap
            )
        )

    else:
        loss = effective_base_loss

    return loss, score


def get_sce_scale(progress, s_max=2.0, s_min=0.5, schedule='fraction'):
    """Return the scale for iteration-based training progress in [0, 1].

    With bounds 2.0 -> 0.5, the fixed schedules are:
        cosine:   1.25 + 0.75 * cos(pi * progress)
        fraction: 2.0 / (1.0 + 3.0 * progress)
        linear:   2.0 - 1.5 * progress
    """
    progress, s_max, s_min = float(progress), float(s_max), float(s_min)
    if not all(math.isfinite(value) for value in (progress, s_max, s_min)):
        raise ValueError('SCE progress and scale bounds must be finite')
    if not 0.0 < s_min <= s_max:
        raise ValueError('SCE scale bounds must satisfy 0 < s_min <= s_max')
    if schedule not in ('cosine', 'fraction', 'linear'):
        raise ValueError(f'Unknown SCE scale schedule: {schedule}')

    progress = min(max(progress, 0.0), 1.0)
    if schedule == 'cosine':
        return s_min + (s_max - s_min) * 0.5 * (1.0 + math.cos(math.pi * progress))
    if schedule == 'linear':
        return s_max + (s_min - s_max) * progress
    # Preserve the existing fraction formula: 2 / (1 + 3 * progress)
    # for the default scale bounds 2.0 -> 0.5.
    return s_max * s_min / (s_min + (s_max - s_min) * progress)
