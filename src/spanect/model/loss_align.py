# import torch
# import torch.nn.functional as F

# def symmetric_alignment_loss(q, k, temperature=0.07):
#     """
#     对称 cross-attention 模态对齐损失。
#     Args:
#         q: 模态1作为query后的表示 (B, N, D)
#         k: 模态2作为key/value后的表示 (B, N, D)
#     Returns:
#         平均对称交叉熵损失
#     """
#     sim = torch.bmm(q, k.permute(0, 2, 1)) / temperature
#     sim = F.log_softmax(sim, dim=-1)
#     targets = torch.arange(sim.size(-1)).to(sim.device)
#     targets = targets.unsqueeze(0).expand(sim.size(0), -1).reshape(-1)

#     loss_qk = F.nll_loss(sim.view(-1, sim.size(-1)), targets)
#     loss_kq = F.nll_loss(sim.transpose(1, 2).contiguous().view(-1, sim.size(1)), targets)

#     return 0.5 * (loss_qk + loss_kq)
# def symmetric_alignment_loss(aligned_pairs, temperature=0.07):
#     total_loss = 0
#     for query_feat, aligned_feat in aligned_pairs:   # 每个都是 (N, D)
#         sim = torch.mm(query_feat, aligned_feat.t()) / temperature   # (N, N)
#         labels = torch.arange(sim.size(0), device=sim.device)
#         loss_qk = F.cross_entropy(sim, labels)          # 行方向
#         loss_kq = F.cross_entropy(sim.t(), labels)      # 列方向（对称）
#         total_loss += (loss_qk + loss_kq) * 0.5
#     return total_loss / len(aligned_pairs)

import torch
import torch.nn.functional as F

def symmetric_alignment_loss(aligned_pairs, temperature=0.07):
    """
    aligned_pairs: list of (query_feat, aligned_feat) for each modality
    """
    loss = 0
    for query_feat, aligned_feat in aligned_pairs:
        sim = torch.bmm(query_feat.unsqueeze(1), aligned_feat.unsqueeze(2)).squeeze()  # (B,)
        sim = sim / temperature

        sim1 = F.log_softmax(sim, dim=0)
        sim2 = F.log_softmax(sim, dim=0)

        loss_1 = F.nll_loss(sim1, torch.arange(sim.size(0), device=sim.device))
        loss_2 = F.nll_loss(sim2, torch.arange(sim.size(0), device=sim.device))

        loss += (loss_1 + loss_2) * 0.5

    return loss / len(aligned_pairs)
