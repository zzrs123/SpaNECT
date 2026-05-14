"""Smoke test for pp (preprocess) capability exposure."""

from spanect import SpaNECT
from spanect.pp import (
    preprocess,
    preprocess_rna,
    preprocess_rna_from_X,
    preprocess_atac_from_obsm,
    preprocess_omics2_from_obsm,
)

sn = SpaNECT


def main():
    checks = {
        "sn.pp": hasattr(sn, "pp"),
        "sn.pp.preprocess": hasattr(sn.pp, "preprocess"),
        "sn.pp.preprocess_rna": hasattr(sn.pp, "preprocess_rna"),
        "sn.pp.preprocess_rna_from_X": hasattr(sn.pp, "preprocess_rna_from_X"),
        "sn.pp.preprocess_atac_from_obsm": hasattr(sn.pp, "preprocess_atac_from_obsm"),
        "sn.pp.preprocess_omics2_from_obsm": hasattr(sn.pp, "preprocess_omics2_from_obsm"),
    }
    for key, ok in checks.items():
        print(f"{key}: {ok}")

    if not all(checks.values()):
        raise SystemExit(1)

    assert sn.pp.preprocess is preprocess
    assert sn.pp.preprocess_rna is preprocess_rna
    assert sn.pp.preprocess_rna_from_X is preprocess_rna_from_X
    assert sn.pp.preprocess_atac_from_obsm is preprocess_atac_from_obsm
    assert sn.pp.preprocess_omics2_from_obsm is preprocess_omics2_from_obsm

    print("pp capability smoke test passed")


if __name__ == "__main__":
    main()
