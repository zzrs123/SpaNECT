"""
Integration test for single-omics pipeline.
Equivalent to all.py but in pytest format.

Usage:
    pytest tests/integration/test_single_omics_pipeline.py -v
    python tests/integration/test_single_omics_pipeline.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

import pytest
import pandas as pd


class TestSingleOmicsPipeline:
    """Integration tests for single-omics pipeline."""

    @pytest.fixture
    def config_path(self):
        return project_root / "config" / "dlpfc1.yaml"

    @pytest.fixture
    def data_path(self):
        return "/home/dugaoyuan/AAA-new-base/ztest_tmvc_model/1-04-full-model-can/FullModel/DLPFC"

    @pytest.fixture
    def output_dir(self):
        output = project_root / "testresults" / "integration_single_omics"
        output.mkdir(parents=True, exist_ok=True)
        return output

    @pytest.fixture
    def data_names(self):
        return ["151507"]

    def test_single_omics_pipeline(self, config_path, data_path, output_dir, data_names):
        """Run single-omics pipeline on DLPFC data."""
        from spanect import SpaNECT

        all_results = []

        for data_name in data_names:
            print(f"\n[INFO] Processing {data_name}...")

            save_path = output_dir / data_name
            save_path.mkdir(parents=True, exist_ok=True)

            model = SpaNECT(
                config_path=str(config_path),
                dataset='DLPFC',
                data_name=data_name,
                save_path=str(save_path),
                multiomics=False,
            )

            model.load()
            model.construct_graph()
            model.train()
            metrics = model.evaluate()
            model.plot(spot_size=300)

            if isinstance(metrics, dict):
                ari = float(metrics.get("ARI", 0.0))
                nmi = float(metrics.get("NMI", 0.0))
                acc = float(metrics.get("ACC", 0.0))
                f1 = float(metrics.get("F1", 0.0))
            else:
                ari, nmi, acc, f1 = metrics

            print(f"[INFO] Results for {data_name}: ARI={ari:.4f}, NMI={nmi:.4f}")
            all_results.append({"slice": data_name, "ARI": ari, "NMI": nmi, "ACC": acc, "F1": f1})

            processed_file = save_path / f"{data_name}_processed.h5ad"
            model.adata.write_h5ad(processed_file, compression="gzip")

        summary_csv = output_dir / "summary_metrics.csv"
        pd.DataFrame(all_results).to_csv(summary_csv, index=False)

        assert len(all_results) == len(data_names), "Should process all slices"
        assert all(r["ARI"] > 0 for r in all_results), "ARI should be positive"
        print(f"\n[PASS] Single-omics pipeline completed. Results saved to {output_dir}")


def main():
    """Run integration test manually."""
    print("=" * 60)
    print("Single-Omics Pipeline Integration Test")
    print("=" * 60)

    test = TestSingleOmicsPipeline()
    config_path = project_root / "config" / "dlpfc1.yaml"
    data_path = "/home/dugaoyuan/AAA-new-base/ztest_tmvc_model/1-04-full-model-can/FullModel/DLPFC"
    output_dir = project_root / "testresults" / "integration_single_omics"
    data_names = ["151507"]

    test.test_single_omics_pipeline(config_path, data_path, output_dir, data_names)

    print("\n[SUCCESS] Integration test passed!")


if __name__ == "__main__":
    main()
