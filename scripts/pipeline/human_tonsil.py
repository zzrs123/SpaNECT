"""
Multi-omics pipeline using SpaNECT class.
Supports RNA+ATAC/RNA+ADT or more modalities datasets.
"""
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import os
import pandas as pd

from spanect import SpaNECT


def run():
    """
    Batch run multiple slices for multi-omics data.
    """
    project_dir = Path(__file__).resolve().parent
    
    config_path = project_dir / "config" / "tonsil24.yaml"
    dataset = "multi-omics-ADT"
    
    base_h5ad = Path("/home/dugaoyuan/AAA-new-base/ztest_tmvc_model/1-04-full-model-can/FullModel/Train_Data/TONSIL_NEW")
    output_path = project_dir / "testresults" / "multiomics_tonsil"
    os.makedirs(output_path, exist_ok=True)
    
    slices = ["tonsil"]
    
    all_results = []
    
    for sid in slices:
        print(f"\n[INFO] Processing {sid}...")
        
        h5_file = base_h5ad / f"{sid}.h5ad"
        save_path = output_path / sid
        os.makedirs(save_path, exist_ok=True)
        
        model = SpaNECT(
            config_path=str(config_path),
            use_h5ad=True,
            h5ad_path=str(h5_file),
            dataset=dataset,
            data_name=sid,
            save_path=str(save_path),
            multiomics=True,
            has_labels=True,
        )
        
        model.load()
        
        if "ground_truth" in model.adata.obs:
            model.adata = model.adata[model.adata.obs["ground_truth"].notna()].copy()
        
        if "X_atac" not in model.adata.obsm and "X_adt" in model.adata.obsm:
            model.adata.obsm["X_atac"] = model.adata.obsm["X_adt"]
            print("[INFO] Aliased obsm['X_adt'] -> obsm['X_atac']")
        
        if "cell_type" not in model.adata.obsm:
            print("[WARN] adata.obsm['cell_type'] not found. Multiomics mode requires cell_type.")
            print("[INFO] Available obsm keys:", list(model.adata.obsm.keys()))
            raise KeyError("Need adata.obsm['cell_type'] for multiomics mode.")
        
        model.construct_graph()
        
        model.train()
        
        metrics = model.evaluate()
        if isinstance(metrics, dict):
            ari = float(metrics.get("ARI", 0.0))
            nmi = float(metrics.get("NMI", 0.0))
            acc = float(metrics.get("ACC", 0.0))
            f1 = float(metrics.get("F1", 0.0))
        else:
            ari, nmi, acc, f1 = metrics
        print(f"[INFO] Evaluation results for {sid}: ARI={ari:.4f}, NMI={nmi:.4f}, ACC={acc:.4f}, F1={f1:.4f}")
        
        all_results.append({"slice": sid, "ARI": ari, "NMI": nmi, "ACC": acc, "F1": f1})
        
        model.plot(spot_size=1.0)
        
        model.adata.obs["domain_pred"] = model.adata.obs["domain_pred"].astype(str)
        processed_file = save_path / f"{sid}.h5ad"
        model.adata.write_h5ad(processed_file, compression="gzip")
        print(f"[INFO] Saved processed AnnData -> {processed_file}")
    
    summary_csv = output_path / "summary_metrics.csv"
    pd.DataFrame(all_results).to_csv(summary_csv, index=False)
    print(f"\n[INFO] Saved summary metrics CSV -> {summary_csv}")
    print("\nALL DONE")


if __name__ == "__main__":
    run()
