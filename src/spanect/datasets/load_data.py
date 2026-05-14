import os
import sys
import numpy as np
import anndata
import scanpy as sc
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from scipy.sparse import issparse,csr_matrix
from sklearn.preprocessing import maxabs_scale, MaxAbsScaler
from torch.utils.data import TensorDataset

import matplotlib.pyplot as plt

from pathlib import Path, PurePath
from typing import Optional, Union
from anndata import AnnData
import numpy as np
from PIL import Image
import pandas as pd
# import stlearn
# from _compat import Literal
import scanpy
import scipy
import matplotlib.pyplot as plt

def read_10X_Visium(path, 
                    data_name='151673',
                    genome = None,
                    count_file ='filtered_feature_bc_matrix.h5', 
                    library_id = None, 
                    load_images =True, 
                    quality ='hires',
                    image_path = None):
    adata = sc.read_visium(path, 
                        genome = genome,
                        count_file = count_file,
                        library_id = library_id,
                        load_images = load_images,
                        )
    adata.var_names_make_unique()
    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]
    if quality == "fulres":
        image_coor = adata.obsm["spatial"]
        img = plt.imread(image_path, 0)
        adata.uns["spatial"][library_id]["images"]["fulres"] = img
    else:
        scale = adata.uns["spatial"][library_id]["scalefactors"][
            "tissue_" + quality + "_scalef"]
        image_coor = adata.obsm["spatial"] * scale
    adata.obs["imagecol"] = image_coor[:, 0]
    adata.obs["imagerow"] = image_coor[:, 1]
    adata.uns["spatial"][library_id]["use_quality"] = quality

    # layer_guess or y_true for evaluate
    path = Path(path)  
    df_meta = pd.read_csv(path / 'metadata.tsv', sep='\t')
    adata.obs['layer_guess'] = df_meta['layer_guess']

    # cell_type in adata.uns
    # 尝试读取CSV文件，检查前几行数据
    # df_cell_type = pd.read_csv( path/ 'cell_type_distribution.csv', sep=',', header=None)  # header=0 假设第一行是列名
    df_cell_type = pd.read_csv( path/ f'cell_type_decon_{data_name}.csv', sep=',', header=None)  # header=0 假设第一行是列名
    # df_cell_type = pd.read_csv( path/ f'151673_cell_type_cardfree.csv', sep=',', header=None)  # header=0 假设第一行是列名
    
    # print(df_cell_type.head())
    # # 检查原始数据的形状
    # print(df_cell_type.shape)
    # print("============")
    df_cell_type = df_cell_type.iloc[1:, 1:]
    # 检查原始数据的形状
    # print(df_cell_type.shape)
    # print("=============")
    # print(df_cell_type.head)
    df_cell_type = df_cell_type.reset_index(drop=True)
    df_cell_type.index = adata.obs.index
    # print(df_cell_type.index)
    # print(adata.obs.index)
    # 将df_cell_type的索引设置为与adata.obs的索引一致
    # df_cell_type.index = adata.obs.index
    adata.obsm['cell_type'] = df_cell_type.values
       # 确保adata.obsm['cell_type']中的数据是数值类型
    if not np.issubdtype(adata.obsm['cell_type'].dtype, np.number):
        adata.obsm['cell_type'] = adata.obsm['cell_type'].astype(float)
    # print("====================")
    # print(adata.obsm['cell_type'])
    # 获取cell_type矩阵
    cell_type_matrix = adata.obsm['cell_type']

    # 计算每一行的和
    row_sums = np.sum(cell_type_matrix, axis=1)

    # 检查每一行的和是否为1
    is_sum_one = np.allclose(row_sums, 1.0)
    assert(is_sum_one,1)
    # # 打印结果
    # print("每一行的和是否都为1:", is_sum_one)

    # # 如果需要，可以打印每一行的和
    # print("每一行的和:", row_sums)
    # assert()
    return adata


def load_adata(
        platform='Visium',
        data_path="./data/DLPFC",
        data_name='151673',
        save_path="../Results",
        verbose=True,
        ):
    assert platform in ['Visium', 'ST', 'MERFISH', 'slideSeq', 'stereoSeq']
    if platform in ['Visium', 'ST']:
        if platform == 'Visium':
            adata = read_10X_Visium(os.path.join(data_path, data_name), data_name=data_name)
            adata = adata[~pd.isnull(adata.obs['layer_guess'])]

    #     else:
    #         adata = ReadOldST(os.path.join(data_path, data_name))
    # elif platform == 'MERFISH':
    #     adata = read_merfish(os.path.join(data_path, data_name))
    # elif platform == 'slideSeq':
    #     adata = read_SlideSeq(os.path.join(data_path, data_name))
    # elif platform == 'seqFish':
    #     adata = read_seqfish(os.path.join(data_path, data_name))
    # elif platform == 'stereoSeq':
    #     adata = read_stereoSeq(os.path.join(data_path, data_name))
    # else:
    #     raise ValueError(
    #                         f"""\
    #                         {self.platform!r} does not support.
    #                         """
    #                     )
    verbose = False
    if verbose:
        save_data_path = Path(os.path.join(save_path, "Data", data_name))
        save_data_path.mkdir(parents=True, exist_ok=True)
        adata.write(os.path.join(save_data_path, f'{data_name}_raw.h5ad'), compression="gzip")
    return adata


def load_h5ad(h5ad_path):
    return sc.read_h5ad(h5ad_path)


__all__ = ["read_10X_Visium", "load_adata", "load_h5ad"]
