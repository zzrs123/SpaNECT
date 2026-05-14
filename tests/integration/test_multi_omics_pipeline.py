"""
Integration test for multi-omics pipeline.
Equivalent to all_m.py but in pytest format.

Usage:
    pytest tests/integration/test_multi_omics_pipeline.py -v
    python tests/integration/test_multi_omics_pipeline.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

import pytest
import pandas as pd


class TestMultiOmicsPipeline:
    """Integration tests for multi-omics pipeline."""

    @pytest.fixture
    def config_path(self):
        return project_root / "config" / "tonsil24.yaml"

    @pytest.fixture
    def h5ad_base_path(self):
        return Path("/home/dugaoyuan/AAA-new-base/ztest_tmvc_model/1-04-full-model-can/FullModel/Train_Data/TONSIL_NEW")

    @pytest.fixture
    def output_dir(self):
        output = project_root / "testresults" / "integration_multi_omics"
        output.mkdir(parents=True, exist_ok=True)
        return output

    @pytest.fixture
    def slices(self):
        return ["tonsil"]

    def test_multi_omics_pipeline(self, config_path, h5ad_base_path, output_dir, slices):
        """Run multi-omics pipeline on TONSIL data."""
        from spanect import SpaNECT

        all_results = []

        for sid in slices:
            print(f"\n[INFO] Processing {sid}...")

            h5_file = h5ad_base_path / f"{sid}.h5ad"
            if not h5_file.exists():
                pytest.skip(f"h5ad file not found: {h5_file}")

            save_path = output_dir / sid
            save_path.mkdir(parents=True, exist_ok=True)

            model = SpaNECT(
                config_path=str(config_path),
                use_h5ad=True,
                h5ad_path=str(h5_file),
                dataset="multi-omics-ADT",
                data_name=sid,
                save_path=str(save_path),
                multiomics=True,
                has_labels=True,
                omics2_key="X_adt",
            )

            model.load()

            if "ground_truth" in model.adata.obs:
                model.adata = model.adata[model.adata.obs["ground_truth"].notna()].copy()

            if "cell_type" not in model.adata.obsm:
                pytest.skip("Need adata.obsm['cell_type'] for multiomics mode.")
            if "X_adt" not in model.adata.obsm:
                pytest.skip("Need adata.obsm['X_adt'] for multiomics mode.")

            model.construct_graph()
            model.train()
            metrics = model.evaluate()
            model.plot(spot_size=2.0)

            if isinstance(metrics, dict):
                ari = float(metrics.get("ARI", 0.0))
                nmi = float(metrics.get("NMI", 0.0))
                acc = float(metrics.get("ACC", 0.0))
                f1 = float(metrics.get("F1", 0.0))
            else:
                ari, nmi, acc, f1 = metrics

            print(f"[INFO] Results for {sid}: ARI={ari:.4f}, NMI={nmi:.4f}")
            all_results.append({"slice": sid, "ARI": ari, "NMI": nmi, "ACC": acc, "F1": f1})

            if "SpaNECT_niche" in model.adata.obs:
                model.adata.obs["SpaNECT_niche"] = model.adata.obs["SpaNECT_niche"].astype(str)
            processed_file = save_path / f"{sid}.h5ad"
            model.adata.write_h5ad(processed_file, compression="gzip")

        summary_csv = output_dir / "summary_metrics.csv"
        pd.DataFrame(all_results).to_csv(summary_csv, index=False)

        assert len(all_results) == len(slices), "Should process all slices"
        print(f"\n[PASS] Multi-omics pipeline completed. Results saved to {output_dir}")


def main():
    """Run integration test manually."""
    print("=" * 60)
    print("Multi-Omics Pipeline Integration Test")
    print("=" * 60)

    test = TestMultiOmicsPipeline()
    config_path = project_root / "config" / "tonsil24.yaml"
    h5ad_base_path = Path("/home/dugaoyuan/AAA-new-base/ztest_tmvc_model/1-04-full-model-can/FullModel/Train_Data/TONSIL_NEW")
    output_dir = project_root / "testresults" / "integration_multi_omics"
    slices = ["tonsil"]

    test.test_multi_omics_pipeline(config_path, h5ad_base_path, output_dir, slices)

    print("\n[SUCCESS] Integration test passed!")


if __name__ == "__main__":
    main()
