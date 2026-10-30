"""Cross-Modal Prototype Topology Consistency (CM-PTC)."""

import torch
import torch.nn.functional as F


def compute_cm_ptc_loss(feats, labels, batch_size, tau_cm):
    """Align row distributions of the two modalities' prototype similarities.

    Features have shape [2B, D], with RGB samples before IR samples, and
    labels have shape [2B]. Only identities present in both modalities are
    used. Both KL directions retain gradients. The caller disables autocast,
    and features are converted to FP32 before computing the prototypes.
    Fewer than two common identities produce a constant zero loss.
    """
    loss_cm_ptc = torch.tensor(0.0, device=feats.device)
    feat_rgb = feats[:batch_size].float()
    feat_ir = feats[batch_size:].float()
    tgt_rgb = labels[:batch_size]
    tgt_ir = labels[batch_size:]

    unique_ids = torch.unique(labels)
    proto_rgb_list = []
    proto_ir_list = []
    for uid in unique_ids:
        mask_rgb = (tgt_rgb == uid)
        mask_ir = (tgt_ir == uid)
        if mask_rgb.any() and mask_ir.any():
            proto_rgb_list.append(feat_rgb[mask_rgb].mean(dim=0))
            proto_ir_list.append(feat_ir[mask_ir].mean(dim=0))

    if len(proto_rgb_list) >= 2:
        P_rgb = torch.stack(proto_rgb_list, dim=0)
        P_ir = torch.stack(proto_ir_list, dim=0)

        P_rgb = F.normalize(P_rgb, p=2, dim=1)
        P_ir = F.normalize(P_ir, p=2, dim=1)

        sim_rgb = torch.mm(P_rgb, P_rgb.t())
        sim_ir = torch.mm(P_ir, P_ir.t())

        sim_rgb = sim_rgb / tau_cm
        sim_ir = sim_ir / tau_cm

        log_p_rgb = F.log_softmax(sim_rgb, dim=1)
        p_rgb = F.softmax(sim_rgb, dim=1)
        log_p_ir = F.log_softmax(sim_ir, dim=1)
        p_ir = F.softmax(sim_ir, dim=1)

        loss_cm_ptc = 0.5 * (
            F.kl_div(log_p_rgb, p_ir, reduction='batchmean') +
            F.kl_div(log_p_ir, p_rgb, reduction='batchmean')
        )

    return loss_cm_ptc
