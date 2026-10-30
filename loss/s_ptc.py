"""Stage-wise Prototype Topology Consistency (S-PTC)."""

import torch
import torch.nn.functional as F


def compute_s_ptc_loss(feat_new, feat_old, prototypes, temperature, KLDivLoss):
    """Match one modality's new and old assignments to a fixed prototype bank.

    Features have shape [B, D] and prototypes have shape [N, D]. The
    temperature controls the softmax sharpness of both assignment distributions.
    KLDivLoss uses batchmean reduction.
    The caller freezes the old features and controls the FP32/AMP context.
    """
    tau = temperature

    feat_new_norm = F.normalize(feat_new, p=2, dim=1)
    feat_old_norm = F.normalize(feat_old, p=2, dim=1)
    proto_norm = F.normalize(prototypes, p=2, dim=1)

    sim_new = torch.mm(feat_new_norm, proto_norm.t()) / tau
    sim_old = torch.mm(feat_old_norm, proto_norm.t()) / tau

    p_new = F.log_softmax(sim_new, dim=1)
    p_old = F.softmax(sim_old, dim=1)

    return KLDivLoss(p_new, p_old)
