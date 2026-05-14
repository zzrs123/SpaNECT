"""Training pipeline facade functions for SpaNECT."""

import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..model.encoder import GCN
from ..model.frozen_encoder import load_frozen_encoder
from ..model.loss_align import symmetric_alignment_loss
from ..model.model_all import MultiModalGraphModel
from ._graph_ops import augment, normalize


def train_pipeline(model):
    print("[INFO] Reproduce Stability...")
    torch.manual_seed(model.config["seed"])
    np.random.seed(model.config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(model.config["seed"])
        torch.cuda.manual_seed_all(model.config["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(model.config["seed"])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)

    print("[INFO] Starting training...")
    model.modalities = model._get_modalities()

    for modality in model.modalities:
        train_single_modality(model, modality)
    train_multimodal(model)


def train_single_modality(model, modality):
    print(f"[INFO] Training {modality} modality...")

    modalities = model.modalities if model.modalities is not None else model._get_modalities()
    mod_map = {m: i for i, m in enumerate(modalities)}

    X = model.Xs[mod_map[modality]]
    A = model.A

    encoder = GCN(
        in_dim=X.shape[1],
        hid_dims=model.config["single_branch_configs"][modality]["hid_dims"],
    ).to(model.device)
    optimizer = torch.optim.Adam(
        encoder.parameters(),
        lr=model.config["single_branch_configs"][modality]["lr"],
        weight_decay=1e-5,
    )

    for epoch in tqdm(
        range(model.config["single_branch_configs"][modality]["epochs"]),
        desc=f"{modality} Training",
        ncols=100,
    ):
        encoder.train()
        optimizer.zero_grad()
        A1, X1 = augment(
            A,
            X,
            model.config["single_branch_configs"][modality]["pd1"],
            model.config["single_branch_configs"][modality]["pm1"],
        )
        A2, X2 = augment(
            A,
            X,
            model.config["single_branch_configs"][modality]["pd2"],
            model.config["single_branch_configs"][modality]["pm2"],
        )
        A_norm1, A_norm2 = normalize(A1, add_self_loops=True), normalize(A2, add_self_loops=True)
        Z1, Z2 = encoder(A_norm1, A_norm2, X1, X2)
        S = Z1 @ Z2.T
        src, dst = A._indices()
        mask = torch.full(S.shape, True, device=model.device)
        mask[src, dst] = False
        mask.fill_diagonal_(False)
        loss_ali = -torch.diag(S).mean()
        loss_nei = -S[src, dst].mean()
        S_mask = torch.masked_select(S, mask)
        S_mask = torch.sigmoid(
            (S_mask - model.config["single_branch_configs"][modality]["loss_s"])
            / model.config["single_branch_configs"][modality]["loss_tau"]
        )
        loss_spa = S_mask.mean()
        loss = (
            loss_ali
            + model.config["single_branch_configs"][modality]["loss_lam"] * loss_nei
            + model.config["single_branch_configs"][modality]["loss_gam"] * loss_spa
        )
        loss.backward()
        optimizer.step()

    model.encoder_path = f"{model.save_path}/single_encoder_{modality}.pt"
    os.makedirs(os.path.dirname(model.encoder_path), exist_ok=True)
    torch.save(encoder.state_dict(), model.encoder_path)
    model.encoders.append(encoder)


def train_multimodal(model):
    print("[INFO] Training multimodal...")

    modalities = model.modalities if model.modalities is not None else model._get_modalities()

    encoders = [
        load_frozen_encoder(
            f"{model.save_path}/single_encoder_{mod}.pt",
            in_dim=model.Xs[i].shape[1],
            hid_dims=model.config["single_branch_configs"][mod]["hid_dims"],
            out_dim=model.config["out_dim"],
        ).to(model.device)
        for i, mod in enumerate(modalities)
    ]

    output_dims_dict = {mod: model.Xs[i].shape[1] for i, mod in enumerate(modalities)}
    model.model = MultiModalGraphModel(
        encoders,
        align_dim=model.config["out_dim"],
        output_dims_dict=output_dims_dict,
    ).to(model.device)
    optimizer = torch.optim.Adam(model.model.parameters(), lr=model.config["mm_lr"])
    A_norm = normalize(model.A, add_self_loops=True)
    modality_to_decode = "all" if model.multiomics else "gene"

    for epoch in tqdm(range(model.config["MM_epochs"]), desc="Multimodal Training", ncols=100):
        model.model.train()
        optimizer.zero_grad()
        inputs = [(A_norm, X) for X in model.Xs]

        z, recon, logvar, aligned_pairs = model.model(inputs, modality_to_decode=modality_to_decode)

        if isinstance(recon, dict):
            recon_loss = 0.0
            for i, mod in enumerate(modalities):
                recon_loss = recon_loss + F.mse_loss(recon[mod], model.Xs[i])
        else:
            recon_loss = F.mse_loss(recon, model.Xs[0])

        kl_loss = -0.5 * torch.sum(1 + logvar - z**2 - logvar.exp(), dim=1).mean()
        align_loss = symmetric_alignment_loss(aligned_pairs)

        loss = recon_loss + model.config["kl_weight"] * kl_loss + model.config["align_weight"] * align_loss
        loss.backward()
        optimizer.step()

    print("[INFO] Multimodal training completed.")


__all__ = ["train_pipeline", "train_single_modality", "train_multimodal"]
