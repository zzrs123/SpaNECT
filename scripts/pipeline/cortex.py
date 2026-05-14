import os
from pathlib import Path
import pandas as pd
from spanect import SpaNECT

def run():
    """
    批量运行多个切片，完成数据加载、图构建、模型训练、评估与可视化
    """
    project_dir = Path(__file__).resolve().parent
    # 配置参数
    config_path = project_dir / "config" / "dlpfc1.yaml"  # 配置文件路径
    data_names = [
                  '151507', 
                # '151508', 
                #   '151509', 
                #   '151510',  # 数据集文件名列表
                #   '151669', '151670', '151671',
                #    '151672',
                #   '151673', '151674', '151675', '151676'
                  ]
    # output_path = './results_all/other_params/'  # 输出目录
    output_path = project_dir / "testresults" / "all_dlpfc"  # 输出目录
    os.makedirs(output_path, exist_ok=True)

    # 加载配置文件
    print("[INFO] Loading configuration...")
    
    # 创建 SpaNECT 类实例
    final_model = SpaNECT(config_path=str(config_path), dataset='DLPFC', data_name=None, save_path=str(output_path), multiomics=False)
    config = final_model.config  # 获取配置字典

    # 用于保存每个切片的评估指标
    all_results = []

    # 循环处理每个切片
    for data_name in data_names:
        print(f"\n[INFO] Processing {data_name}...")
        # final_model.output_path = os.path.join(output_path, data_name)
        final_model.data_name = data_name
        final_model.save_path = str(output_path / data_name)
        # 加载数据和构建图
        print(f"[INFO] Loading data and constructing graph for {data_name}...")
        final_model.load()
        final_model.construct_graph()
        
        # 执行训练
        print(f"[INFO] Training model for {data_name}...")
        final_model.train()
        
        # 评估模型
        print(f"[INFO] Evaluating model for {data_name}...")
        metrics = final_model.evaluate()
        if isinstance(metrics, dict):
            ari = float(metrics.get('ARI', 0.0))
            nmi = float(metrics.get('NMI', 0.0))
            acc = float(metrics.get('ACC', 0.0))
            f1 = float(metrics.get('F1', 0.0))
        else:
            ari, nmi, acc, f1 = metrics
        print(f"[INFO] Evaluation results for {data_name}: ARI={ari:.4f}, NMI={nmi:.4f}, ACC={acc:.4f}, F1={f1:.4f}")
        
        # 将评估结果添加到 all_results 中
        all_results.append([data_name, ari, nmi, acc, f1])
        
        # 可视化结果
        print(f"[INFO] Visualizing results for {data_name}...")
        final_model.plot()

        # 保存结果
        print(f"[INFO] Saving results for {data_name}...")
        os.makedirs(output_path, exist_ok=True)
        # final_model.adata.write_h5ad(os.path.join(output_path, f"{data_name}_processed.h5ad"))
        final_model.adata.obs['Proposed_cluster'] = final_model.adata.obs['domain_pred']  # 存储最终聚类结果
        final_model.adata.obsm['Proposed'] = final_model.adata.obsm['embed']  # 存储嵌入表示
        final_model.adata.obs[['Proposed_cluster']].to_csv(output_path / f"{data_name}_FinalLabels.csv", index=False, header=False)
        final_model.adata.write_h5ad(output_path / f"{data_name}_processed.h5ad")
    # 将所有切片的评估结果保存到 CSV 文件中
    results_df = pd.DataFrame(all_results, columns=['Slice', 'ARI', 'NMI', 'ACC', 'F1'])
    results_df.to_csv(output_path / "all_slices_metrics.csv", index=False)
    print(f"[INFO] All tasks completed. Results saved to {output_path}")


if __name__ == "__main__":
    # 直接执行运行任务
    run()
