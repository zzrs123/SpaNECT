"""Smoke test for training preprocess API exposure."""

from spanect import SpaNECT
from spanect.datasets import load_adata as datasets_load_adata
from spanect.tl._preprocess import (
    preprocess as tl_preprocess_preprocess,
    preprocess_rna as tl_preprocess_preprocess_rna,
    preprocess_atac_from_obsm as tl_preprocess_preprocess_atac_from_obsm,
    preprocess_omics2_from_obsm as tl_preprocess_preprocess_omics2_from_obsm,
    preprocess_rna_from_X as tl_preprocess_preprocess_rna_from_X,
)

sn = SpaNECT


def main():
    checks = {
        "sn.tl._preprocess": hasattr(sn.tl, "_preprocess"),
        "sn.tl.load_adata": hasattr(sn.tl, "load_adata"),
        "sn.tl.preprocess": hasattr(sn.tl, "preprocess"),
        "sn.tl.preprocess_rna": hasattr(sn.tl, "preprocess_rna"),
        "sn.tl.preprocess_rna_from_X": hasattr(sn.tl, "preprocess_rna_from_X"),
        "sn.tl.preprocess_atac_from_obsm": hasattr(sn.tl, "preprocess_atac_from_obsm"),
        "sn.tl.preprocess_omics2_from_obsm": hasattr(sn.tl, "preprocess_omics2_from_obsm"),
    }
    for key, ok in checks.items():
        print(f"{key}: {ok}")

    if not all(checks.values()):
        raise SystemExit(1)

    assert sn.tl.load_adata is datasets_load_adata, "sn.tl.load_adata should expose datasets.load_adata"
    assert sn.tl.preprocess is tl_preprocess_preprocess, "sn.tl.preprocess should expose tl._preprocess (-> pp)"
    assert sn.tl.preprocess_rna is tl_preprocess_preprocess_rna, "sn.tl.preprocess_rna should expose tl._preprocess"
    assert (
        sn.tl.preprocess_rna_from_X is tl_preprocess_preprocess_rna_from_X
    ), "sn.tl.preprocess_rna_from_X should expose tl._preprocess"
    assert (
        sn.tl.preprocess_atac_from_obsm is tl_preprocess_preprocess_atac_from_obsm
    ), "sn.tl.preprocess_atac_from_obsm should expose tl._preprocess"
    assert (
        sn.tl.preprocess_omics2_from_obsm is tl_preprocess_preprocess_omics2_from_obsm
    ), "sn.tl.preprocess_omics2_from_obsm should expose tl._preprocess"

    print("training preprocess API smoke test passed")


if __name__ == "__main__":
    main()
