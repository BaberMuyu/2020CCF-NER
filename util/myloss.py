import torch.nn as nn


def sl_loss(logits, gt, A, a, b):
    ce_loss = nn.functional.binary_cross_entropy(logits, gt, reduction="none")
    rce_loss = gt * (logits - 1) * A + (gt - 1) * logits * A
    # rce_loss = 0
    loss = a * ce_loss + b * rce_loss
    return loss
