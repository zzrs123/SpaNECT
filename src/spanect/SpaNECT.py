import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import os, yaml
import numpy as np
import torch
import pandas as pd

from . import datasets, gr, pl, tl
from . import characterization as ch


class SpaNECT:
    def __init__(self, config_path=None, config=None,
                 use_h5ad=False, h5ad_path=None,
                 dataset='DLPFC', data_name='151507', save_path='./results',
                 multiomics=True,
                 has_labels=True,
                 n_clusters=None,
                 omics2_key=None):
        self.config = yaml.safe_load(open(config_path)) if config_path else config
        self.device = torch.device(self.config.get('device', 'cuda:0')
                                   if torch.cuda.is_available() else 'cpu')
        self.use_h5ad = use_h5ad
        self.h5ad_path = Path(h5ad_path) if h5ad_path else None
        self.dataset = dataset
        self.data_name = data_name
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.multiomics = multiomics
        self.has_labels = has_labels
        self.user_n_clusters = n_clusters
        self.omics2_key = omics2_key
        self.adata = None
        self.Xs, self.A, self.Y = None, None, None
        self.encoders = []
        self.model = None
        self.modalities = None

    def _get_modalities(self):
        """Return modality names in order matching Xs."""
        return ['gene', 'omics2', 'celltype'] if self.multiomics else ['gene', 'image', 'celltype']

    def load(self):
        if self.use_h5ad and self.h5ad_path:
            self.adata = datasets.load_h5ad(self.h5ad_path)
        else:
            self.adata = datasets.load_adata(
                platform='Visium',
                data_path=self.config['data_path'],
                data_name=self.data_name
            )
        print(f"[INFO] Loaded {'h5ad' if self.use_h5ad else 'raw'}: {self.data_name}")

    def _to_numpy(self, x):
        return x.toarray() if hasattr(x, 'toarray') else np.asarray(x)

    def construct_graph(self):
        """
        Construct spatial graph from coordinates.

        This method ONLY builds the adjacency matrix A.
        Feature assembly is handled by get_model_input().
        """
        if not self.use_h5ad:
            data_dir = Path(self.config['data_path']) / self.data_name
            gdict = gr.build_spatial_graph(
                self.adata.obsm['spatial'],
                k=self.config['k']
            )
            self.A = gdict['adj_norm'].to(self.device)

        if self.use_h5ad:
            coords = self.adata.obsm['spatial']
            gdict = gr.build_spatial_graph(coords, k=self.config['k'])
            self.A = gdict['adj_norm'].to(self.device)

        if not self.A.is_sparse:
            dense_A = self.A.to_dense()
            indices = torch.nonzero(dense_A).t()
            values = dense_A[indices[0], indices[1]]
            self.A = torch.sparse_coo_tensor(indices, values, dense_A.size(), device=self.device)

        print(f"[INFO] Constructed graph for dataset {self.data_name}")
        assert self.A.shape[0] == self.A.shape[1], "Adjacency matrix should be square."
        print(f"[INFO] Graph ready: A={self.A.shape}")

    def get_model_input(self):
        """
        Assemble model inputs from all feature sources.

        This method calls tl.get_model_input() to assemble:
        - RNA features (via pp)
        - Image/Omics2 features (via tl._image_feat or pp)
        - Cell type features (via tl._cell_type)
        - Labels Y

        Returns
        -------
        tuple
            (Xs, Y, adata_filtered)
        """
        Xs, Y, adata_filtered = tl.get_model_input(
            adata=self.adata,
            data_path=self.config.get('data_path'),
            data_name=self.data_name,
            config=self.config,
            device=self.device,
            multiomics=self.multiomics,
            use_h5ad=self.use_h5ad,
            h5ad_path=self.h5ad_path,
            omics2_key=self.omics2_key,
            has_labels=self.has_labels,
        )

        self.Xs = Xs
        self.Y = Y

        if adata_filtered is not None:
            self.adata = adata_filtered

        if self.Y is not None:
            n_y = len(self.Y) if not hasattr(self.Y, 'shape') else (
                self.Y.shape[0] if len(self.Y.shape) > 0 else int(self.Y)
            )
            assert self.Xs[0].shape[0] == n_y, "Mismatch between features and labels size."

        print(f"[INFO] Model inputs ready: Xs={[x.shape for x in self.Xs]}, "
              f"Y={'None' if self.Y is None else 'available'}")

        return Xs, Y, adata_filtered

    def construct_graph_and_inputs(self):
        """
        Convenience method that calls construct_graph() and get_model_input().

        This is the recommended entry point for preparing all model inputs.
        """
        self.construct_graph()
        self.get_model_input()

        print(f"[INFO] Features+Graph ready: Xs={[x.shape for x in self.Xs]}, A={self.A.shape}, "
              f"Y={'None' if self.Y is None else 'available'}")

    def validate_inputs(self, verbose=True):
        """
        Validate that all three branches have proper inputs.

        This method checks:
        1. Xs has 3 elements (gene/omics1, image/omics2, celltype)
        2. Each element has correct shape
        3. A (adjacency matrix) is built
        4. Y (labels) exists if has_labels=True

        Parameters
        ----------
        verbose : bool
            Whether to print detailed information.

        Returns
        -------
        dict
            Validation result with keys:
            - 'valid': bool, whether all inputs are valid
            - 'Xs': dict with shape info for each modality
            - 'A': dict with shape info for adjacency matrix
            - 'Y': dict with label info
            - 'errors': list of error messages
        """
        errors = []
        result = {'valid': True, 'Xs': {}, 'A': {}, 'Y': {}, 'errors': errors}

        modalities = self._get_modalities()

        if self.Xs is None:
            errors.append("Xs is None. Call get_model_input() first.")
            result['valid'] = False
        elif len(self.Xs) != 3:
            errors.append(f"Xs should have 3 elements, got {len(self.Xs)}")
            result['valid'] = False
        else:
            for i, (X, mod_name) in enumerate(zip(self.Xs, modalities)):
                if X is None:
                    errors.append(f"Xs[{i}] ({mod_name}) is None")
                    result['valid'] = False
                else:
                    result['Xs'][mod_name] = {
                        'shape': tuple(X.shape),
                        'dtype': str(X.dtype),
                        'device': str(X.device) if hasattr(X, 'device') else 'numpy',
                        'nan_count': int(torch.isnan(X).sum()) if hasattr(X, 'isnan') else 'N/A',
                        'zero_ratio': float((X == 0).sum() / X.numel()) if hasattr(X, 'numel') else 'N/A',
                    }

        if self.A is None:
            errors.append("A (adjacency matrix) is None. Call construct_graph() first.")
            result['valid'] = False
        else:
            result['A'] = {
                'shape': tuple(self.A.shape),
                'is_sparse': self.A.is_sparse if hasattr(self.A, 'is_sparse') else False,
                'device': str(self.A.device) if hasattr(self.A, 'device') else 'numpy',
            }

        if self.has_labels:
            if self.Y is None:
                errors.append("Y (labels) is None but has_labels=True")
                result['valid'] = False
            else:
                y_np = self._safe_to_numpy_1d(self.Y)
                result['Y'] = {
                    'shape': y_np.shape,
                    'n_classes': len(np.unique(y_np[~pd.isna(y_np)])) if y_np is not None else 0,
                    'has_nan': bool(pd.isna(y_np).any()) if y_np is not None else False,
                }
        else:
            result['Y'] = {'note': 'has_labels=False'}

        if verbose:
            print("\n" + "=" * 60)
            print("INPUT VALIDATION REPORT")
            print("=" * 60)

            print(f"\n[Mode] {'Multi-omics' if self.multiomics else 'Single-omics'}")
            print(f"[Modalities] {modalities}")

            print("\n[Xs - Feature Tensors]")
            if result['Xs']:
                for mod_name, info in result['Xs'].items():
                    print(f"  - {mod_name}: shape={info['shape']}, dtype={info['dtype']}, "
                          f"device={info['device']}, nan={info['nan_count']}, zero_ratio={info['zero_ratio']:.2%}" if isinstance(info['zero_ratio'], float) else f"  - {mod_name}: shape={info['shape']}, dtype={info['dtype']}, device={info['device']}")
            else:
                print("  [ERROR] Xs is empty or None")

            print("\n[A - Adjacency Matrix]")
            if result['A']:
                print(f"  - shape={result['A']['shape']}, sparse={result['A']['is_sparse']}, device={result['A']['device']}")
            else:
                print("  [ERROR] A is None")

            print("\n[Y - Labels]")
            if result['Y']:
                if 'note' in result['Y']:
                    print(f"  - {result['Y']['note']}")
                else:
                    print(f"  - shape={result['Y']['shape']}, n_classes={result['Y']['n_classes']}, has_nan={result['Y']['has_nan']}")
            else:
                print("  [ERROR] Y is None")

            if errors:
                print("\n[ERRORS]")
                for err in errors:
                    print(f"  - {err}")
            else:
                print("\n[STATUS] All inputs valid ✓")

            print("=" * 60)

        return result

    def train(self):
        return tl.train_pipeline(self)

    def _train_single_modality(self, modality):
        return tl.train_single_modality(self, modality)

    def _train_multimodal(self):
        return tl.train_multimodal(self)

    def _safe_to_numpy_1d(self, y):
        if y is None:
            return None
        try:
            import pandas as pd
            if isinstance(y, pd.Series):
                return y.values
        except Exception:
            pass
        if torch.is_tensor(y):
            return y.detach().cpu().numpy()
        y = np.asarray(y)
        return y

    def _auto_select_k(self, embed, k_min=2, k_max=10, random_state=0):
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
        except ImportError:
            print("[WARN] scikit-learn is not installed. Fallback to k=7.")
            return 7
        best_k, best_sc = None, -1.0
        for k in range(k_min, k_max + 1):
            km = KMeans(n_clusters=k, n_init='auto', random_state=random_state)
            labels = km.fit_predict(embed)
            if len(np.unique(labels)) < 2:
                continue
            try:
                sc = silhouette_score(embed, labels)
            except Exception:
                continue
            if sc > best_sc:
                best_sc, best_k = sc, k
        if best_k is None:
            print("[WARN] Auto-selecting n_clusters failed. Fallback to k=7.")
            return 7
        print(f"[INFO] Auto-selected n_clusters={best_k} (max silhouette={best_sc:.4f})")
        return best_k

    def evaluate(self):
        return tl.evaluate_pipeline(self)

    def plot(self, spot_size=150):
        return pl.plot_pipeline(self, spot_size=spot_size)

    def get_modality_weights_per_cell(self, reduce='mean'):
        return ch.get_modality_weights_per_cell(model=self.model, Xs=self.Xs, A=self.A, reduce=reduce)

    def get_region_modality_weights(self, region_key):
        return ch.get_region_modality_weights(
            model=self.model,
            adata=self.adata,
            Xs=self.Xs,
            A=self.A,
            region_key=region_key,
            reduce='mean',
        )

    def attach_modality_weights(self, reduce='mean', key='modality_weights'):
        ok = ch.attach_modality_weights(
            model=self.model,
            adata=self.adata,
            Xs=self.Xs,
            A=self.A,
            reduce=reduce,
            key=key,
        )
        if not ok:
            print("[WARN] no attention weights found; did you run a forward pass?")
            return
        print(f"[INFO] wrote modality weights to obsm['{key}'] and obs/uns.")

    def attach_region_modality_summary(self, region_key, key='modality_weights'):
        ch.attach_region_modality_summary(adata=self.adata, region_key=region_key, key=key)
        print(f"[INFO] wrote region summary to uns['{key}_by_{region_key}_*'].")



    def _supervised_eval_dispatch(self):
        return tl.supervised_eval_dispatch(self)
