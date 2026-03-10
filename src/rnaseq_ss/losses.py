from __future__ import annotations

import torch
import torch.nn.functional as F


def multitask_loss(
    regression_pred: torch.Tensor,
    regression_target: torch.Tensor,
    cls_logits: torch.Tensor,
    cls_target: torch.Tensor,
    regression_weight: float = 1.0,
    classification_weight: float = 1.0,
) -> torch.Tensor:
    reg_loss = F.mse_loss(regression_pred, regression_target)
    cls_loss = F.binary_cross_entropy_with_logits(cls_logits, cls_target)
    return regression_weight * reg_loss + classification_weight * cls_loss
