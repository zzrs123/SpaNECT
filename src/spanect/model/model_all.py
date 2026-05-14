import torch
import torch.nn as nn
import torch.nn.functional as F

# class CrossAttentionAligner(nn.Module):
#     def __init__(self, dim, heads=4):
#         super().__init__()
#         self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
#         self.norm = nn.LayerNorm(dim)

#     def forward(self, query, key_value):
#         attn_output, _ = self.cross_attn(query, key_value, key_value)
#         return self.norm(attn_output)

class CrossAttentionAligner(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.last_attn = None  # ⭐ 新增：缓存最近一次前向的注意力 (B, 1, N)

    def forward(self, query, key_value, mask=None):
        # 第一层 Cross Attention + Residual + Norm
        attn_output, attn_w  = self.cross_attn(
            query, key_value, key_value, 
            key_padding_mask=mask,
            need_weights=True,          # ⭐
            average_attn_weights=True   # ⭐ -> (B, 1, N)
        )
        self.last_attn = attn_w  # ⭐ (B, 1, N)

        x = query + attn_output 
        x = self.norm1(x)

        # 第二层 FFN + Residual + Norm
        x2 = self.ffn(x)
        x = x + x2
        x = self.norm2(x)
        return x


class ProductOfExperts(nn.Module):
    def forward(self, means, logvars):
        vars = torch.exp(logvars) + 1e-8
        T = 1. / vars
        mu_joint = torch.sum(means * T, dim=0) / torch.sum(T, dim=0)
        var_joint = 1. / torch.sum(T, dim=0)
        logvar_joint = torch.log(var_joint)
        return mu_joint, logvar_joint

# class ProductOfExperts(nn.Module):
#     def __init__(self, n_modalities=3, learn_confidence=True):
#         super().__init__()
#         if learn_confidence:
#             self.confidences = nn.Parameter(torch.ones(n_modalities))
#         else:
#             self.confidences = None

#     def forward(self, means, logvars):
#         vars = torch.exp(logvars) + 1e-8  # 防止除0
#         precisions = 1. / vars  # precision = 1/variance

#         if self.confidences is not None:
#             precisions = precisions * self.confidences.view(-1, 1, 1)

#         weighted_means = means * precisions
#         mu_joint = torch.sum(weighted_means, dim=0) / torch.sum(precisions, dim=0)
#         var_joint = 1. / torch.sum(precisions, dim=0)
#         logvar_joint = torch.log(var_joint + 1e-8)

#         return mu_joint, logvar_joint


class SharedDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.decoder = nn.Linear(latent_dim, output_dim)

    def forward(self, z):
        return self.decoder(z)
    
class SelectiveDecoder(nn.Module):
    def __init__(self, latent_dim, output_dims_dict):
        """
        Args:
            output_dims_dict: dict {modality_name: output_dim}
        """
        super().__init__()
        self.decoder_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                # nn.LayerNorm(latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, dim)
            )
            for name, dim in output_dims_dict.items()
        })

    def forward(self, z, modality='gene'):
        """
        Args:
            z: latent embedding (B, latent_dim)
            modality: str, which head to decode
        Returns:
            Decoded output
        """
        return self.decoder_heads[modality](z)

# class MultiModalGraphModel(nn.Module):
#     def __init__(self, encoders, align_dim, output_dim):
#         super().__init__()
#         self.encoders = nn.ModuleList(encoders)
#         self.aligners = nn.ModuleList([
#             CrossAttentionAligner(align_dim) for _ in range(len(encoders))
#         ])
#         self.aggregator = ProductOfExperts()
#         self.decoder = SharedDecoder(align_dim, output_dim)
#         self.decoder = SelectiveDecoder(align_dim, output_dim)

#     def forward(self, inputs):
#         feats = [enc(A, X) for enc, (A, X) in zip(self.encoders, inputs)]

#         # aligned_feats = []
#         # for i, q in enumerate(feats):
#         #     query = q.unsqueeze(1)
#         #     keys = torch.stack(feats, dim=1)
#         #     aligned = self.aligners[i](query, keys).squeeze(1)
#         #     aligned_feats.append(aligned)
        
#         aligned_pairs = []
#         for i in range(len(feats)):
#             query = feats[i].unsqueeze(1)  # (B, 1, D)
#             keys = torch.stack(feats, dim=1)  # (B, N, D)
#             aligned_feat = self.aligners[i](query, keys)  # (B, 1, D)
#             aligned_pairs.append((feats[i], aligned_feat.squeeze(1)))  # (raw_feat, aligned_feat)

#         # means = torch.stack(aligned_feats)
#         # logvars = torch.stack([torch.zeros_like(m) for m in aligned_feats])
#         # z, logvar = self.aggregator(means, logvars)
#          # 下面 PoE 聚合
#         means = torch.stack([aligned_feat for _, aligned_feat in aligned_pairs])
#         logvars = torch.stack([torch.zeros_like(m) for m in means])
#         z, logvar = self.aggregator(means, logvars)

#         recon = self.decoder(z)
#         return z, recon, logvar,aligned_pairs

# 备份 for multi-omics,下面的版本是DLPFC等纯空转版本
# class MultiModalGraphModel(nn.Module):
#     def __init__(self, encoders, align_dim, output_dims_dict):
#         super().__init__()
#         self.encoders = nn.ModuleList(encoders)
#         self.aligners = nn.ModuleList([
#             CrossAttentionAligner(align_dim) for _ in range(len(encoders))
#         ])
#         # self.aggregator = ProductOfExperts(n_modalities=len(encoders))
#         self.aggregator = ProductOfExperts()
#         self.decoder = SelectiveDecoder(align_dim, output_dims_dict)

#     def forward(self, inputs, modality_to_decode='gene'):
#         feats = [enc(A, X) for enc, (A, X) in zip(self.encoders, inputs)]

#         aligned_pairs = []
#         for i in range(len(feats)):
#             query = feats[i].unsqueeze(1)  # (B, 1, D)
#             keys = torch.stack(feats, dim=1)  # (B, N, D)
#             aligned_feat = self.aligners[i](query, keys)  # (B, 1, D)
#             aligned_pairs.append((feats[i], aligned_feat.squeeze(1)))

#         means = torch.stack([aligned_feat for _, aligned_feat in aligned_pairs])
#         logvars = torch.stack([torch.zeros_like(m) for m in means])
#         z, logvar = self.aggregator(means, logvars)

#         # 🎯 关键变化在这里！！
#         recon = self.decoder(z, modality=modality_to_decode)

#         return z, recon, logvar, aligned_pairs


# model/model_all.py
# 1. 适配了多组学任务对于多解码器的需求
# 2. 增加了 Cross-Attn 层的注意力权重缓存，方便后续可视化
# import torch
# import torch.nn as nn

class MultiModalGraphModel(nn.Module):
    def __init__(self, encoders, align_dim, output_dims_dict):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.aligners = nn.ModuleList([
            CrossAttentionAligner(align_dim) for _ in range(len(encoders))
        ])
        self.aggregator = ProductOfExperts()
        self.decoder = SelectiveDecoder(align_dim, output_dims_dict)
        # 方便遍历时得到各模态名字的顺序（与 output_dims_dict 的键一致）
        self.modality_names = list(output_dims_dict.keys())
        self.last_attn_weights = None  # ⭐ 新增：dict {query_mod: (B, N)}

    def forward(self, inputs, modality_to_decode='all'):
        # 编码
        feats = [enc(A, X) for enc, (A, X) in zip(self.encoders, inputs)]

        # Cross-attn 对齐
        aligned_pairs = []
        attn_dict = {}
        for i in range(len(feats)):
            query = feats[i].unsqueeze(1)              # (B, 1, D)
            keys  = torch.stack(feats, dim=1)          # (B, N, D)  <-- 这里的 keys 永远是张量，不是字符串
            aligned_feat = self.aligners[i](query, keys)   # (B, 1, D)
            aligned_pairs.append((feats[i], aligned_feat.squeeze(1)))

            if self.aligners[i].last_attn is not None:
                attn_dict[self.modality_names[i]] = self.aligners[i].last_attn.squeeze(1)  # (B, N)

        # PoE Aggregator 融合
        means   = torch.stack([af for _, af in aligned_pairs])
        logvars = torch.stack([torch.zeros_like(m) for m in means])
        z, logvar = self.aggregator(means, logvars)

        # Decode
        # ✅ 新增：all_omics —— 解码所有组学模态，但不解码 celltype
        if modality_to_decode in ('all', 'all_omics'):
            decode_targets = list(self.decoder.decoder_heads.keys())

            if modality_to_decode == 'all_omics':
                # 兼容不同命名写法
                exclude = {'celltype', 'cell_type', 'cell-type'}
                decode_targets = [m for m in decode_targets if m not in exclude]

            recon = {name: self.decoder(z, modality=name) for name in decode_targets}
        else:
            recon = self.decoder(z, modality=modality_to_decode)

        # 缓存注意力
        self.last_attn_weights = attn_dict

        return z, recon, logvar, aligned_pairs
    
    