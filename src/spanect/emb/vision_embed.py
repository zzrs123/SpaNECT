"""
提取 Visium / 10x 空间转录组切片的图像表征
------------------------------------------------
使用 BYOL(ResNet50) 做自监督训练，首轮生成 embeddings.npy，
后续重复调用将直接加载已有文件加速。

依赖：
  pip install torchvision opencv-python scikit-image pillow
"""
from pathlib import Path
import os, numpy as np, cv2, torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms as T
from tqdm import tqdm
from PIL import Image

# BYOL 来自 https://github.com/lucidrains/byol-pytorch
from byol_pytorch import BYOL
import torchvision.models as models


class VisionEmbed:
    def __init__(self,
                 slide_dir: str | Path,
                 adata=None,
                 data_name=None,
                 epoch_num: int = 1,
                 patch_size: int = 512,
                 device: str | torch.device = "cuda:5"):

        self.slide_dir = Path(slide_dir)
        self.adata = adata
        self.data_name = data_name
        self.epoch_num = epoch_num
        self.patch_size = patch_size
        self.device = torch.device(device)

        # 输出文件_without_noise_dark
        # self.emb_path   = self.slide_dir /f"{self.data_name}_embeddings.npy"
        self.emb_path = self.slide_dir / f"{self.data_name}_embeddings.npy"

    # ---------- 对外主入口 ----------
    def run(self) -> np.ndarray:
        """返回 (n_spots × d) 特征矩阵，若已存在则直接加载"""
        if self.emb_path.exists():
            print(f"[✓] Found {self.emb_path}, loading ...")
            embeddings = np.load(self.emb_path)
            # 假设 adata 也已经存在或需要重新加载
            # 如果 adata 不需要更新，可以返回默认值或加载现有数据
            self.adata.obsm['image_feat'] = embeddings
            return embeddings, self.adata  # 确保返回两个值
            # return np.load(self.emb_path)

        # 1. 切 patch
        clip_dir = self._clip_patches()             # *.png

        # 2. 频域滤波 & Resize
        filt_dir = self._filter_patches(clip_dir)

        # 3. BYOL 训练
        learner, dataset = self._train_byol(filt_dir)

        # 4. 推理得到 embedding
        embeddings = self._infer_embeddings(learner, dataset)

        np.save(self.emb_path, embeddings)
        print(f"[✓] embeddings saved to {self.emb_path}")
        self.adata.obsm['image_feat'] = embeddings
        return embeddings, self.adata

    def _to_bgr(self, img):
        """
        将 adata.uns['...']['hires'] 里的多样格式
        (PIL.Image / np.ndarray-RGB / np.ndarray-ARGB) 统一转 OpenCV(BGR) ndarray
        """
        import numpy as np, cv2
        if img is None:
            return None
        # 1) PIL Image
        if "PIL" in str(type(img)):
            img = np.asarray(img)
        # 2) 处理 Alpha 通道
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]
        # 3) 如果是 RGB → BGR
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img.copy()

    # ---------- 1. 切 patch ----------
    def _clip_patches(self) -> Path:

        # from utils.utils_st import LoadSingle10xAdata   # 避免循环引用

        # adata  = LoadSingle10xAdata(path=self.slide_dir, image_emb=False,
        #                             label=True, filter_na=True).run()

        adata = self.adata
        coords = adata.obsm['spatial']                  # n×2
        # im = cv2.imread(str(self.slide_dir / "spatial" / "tissue_full_image.tif"))
        tif_path = self.slide_dir / "spatial" / "tissue_full_image.tif"
        if tif_path.exists():
            im = cv2.imread(str(tif_path), cv2.IMREAD_COLOR)

        # 2⃣  若不存在，再从 adata.uns['spatial'][slice_id]['images']['hires'] 提取
        else:
            # slice_id 采用 h5ad / slide 文件夹本身的名字
            # slice_id = self.slide_dir.name
            hires = adata.uns['spatial'].get(self.data_name, {}).get('images', {}).get('hires')
            # im = self._to_bgr(hires)
            im = hires
        # print(im)
        # print(coords)
        # 3⃣ 兜底报错
        if im is None:
            raise FileNotFoundError(
                "既找不到 tissue_full_image.tif，"
                "也未在 adata.uns['spatial'][slice_id]['images']['hires'] 中取得图像。")
        # 初始亮度提升(AD)
        # 提升亮度
        im = cv2.convertScaleAbs(im, alpha=1.5, beta=0)

        ps = int(3.5 * 144)                             # 和原脚本保持一致
        # AD数据集图片较小，需要修改ps参数
        # ps = 80
        clip_dir = self.slide_dir / "clip_image" / self.data_name
        clip_dir.mkdir(parents=True, exist_ok=True)

        for i, coord in tqdm(enumerate(adata.obsm['spatial']), total=len(adata), desc="Clipping patches"):
            l, t = int(coord[0] - ps / 2), int(coord[1] - ps / 2)

            patch = im[t:t+ps, l:l+ps]
            # print(patch.shape)
            patch = cv2.resize(patch, (512, 512))

            cv2.imwrite(str(clip_dir / f"{i}.png"), patch)
        # H, W = im.shape[:2]
        # for i, (x, y) in tqdm(enumerate(coords), total=len(coords), desc="Clipping patches"):
        #     cx, cy = int(x), int(y)              # x: col, y: row
        #     l, t   = cx - ps//2, cy - ps//2
        #     r, b   = l + ps, t + ps

        #     # -------- ① 与边界比对，超出部分用 0 填充 --------
        #     l_pad = max(0, -l);     r_pad = max(0,  r-W)
        #     t_pad = max(0, -t);     b_pad = max(0,  b-H)

        #     l = max(0, l); r = min(W, r)
        #     t = max(0, t); b = min(H, b)

        #     sub = im[t:b, l:r]                      # 实际截取到的区域
        #     if sub.size == 0:                       # 仍为空，跳过或用黑图
        #         sub = np.zeros((ps, ps, 3), dtype=np.uint8)

        #     if (l_pad or r_pad or t_pad or b_pad):  # 需要补零到 ps×ps
        #         patch = np.zeros((ps, ps, 3), dtype=sub.dtype)
        #         patch[t_pad:t_pad+sub.shape[0], l_pad:l_pad+sub.shape[1]] = sub
        #     else:
        #         patch = sub

        #     patch = cv2.resize(patch, (512, 512))   # 保证此时 patch 一定非空
        #     cv2.imwrite(str(clip_dir / f"{i}.png"), patch)
        return clip_dir

    # ---------- 2. 频域滤波 ----------
    def _filter_patches(self, src_dir: Path) -> Path:
        def process(fp: Path, dst_dir: Path, lower=245, upper=275):
            img = cv2.imread(str(fp), 0)  # gray
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            mask = np.zeros(img.shape); mask[lower:upper, lower:upper] = 1
            img_filtered = np.fft.ifft2(np.fft.ifftshift(fshift * mask))
            img_filtered = cv2.GaussianBlur(np.abs(img_filtered), (15, 15), 0)
            rgb = cv2.cvtColor(np.float32(img_filtered), cv2.COLOR_GRAY2RGB)
            rgb = cv2.resize(rgb, (224, 224))
            cv2.imwrite(str(dst_dir / fp.name), rgb)

        dst_dir = self.slide_dir / "patch_png_filtered" / self.data_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for fn in tqdm(sorted(src_dir.glob("*.png"), key=lambda p: int(p.stem)),
                       desc="Filtering patches"):
            process(fn, dst_dir)
        return dst_dir

    # ---------- 3. BYOL 训练 ----------
    def _train_byol(self, img_dir: Path):
        class ImgDS(Dataset):
            def __init__(self, root, tfm):
                self.files = sorted(list(root.glob("*.png")), key=lambda p: int(p.stem))
                self.tfm = tfm
            def __len__(self): return len(self.files)
            def __getitem__(self, idx):
                img = Image.open(self.files[idx]).convert("RGB")
                return self.tfm(img)

        tfm = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        ds = ImgDS(img_dir, tfm)

        base_bs = 43                       # 原始想用的 batch 大小 (DLPFC=43)
        N = len(ds)                        # 数据集实际样本数

        # 若整除后“只剩 1 张图”，就把 batch_size 减 1
        if N % base_bs == 1:
            eff_bs = base_bs - 1          # 42
        else:
            eff_bs = base_bs if N >= base_bs else N   # 小于 base_bs 时直接用 N
        # dlpfc是43
        dl = DataLoader(
            ds,
            batch_size=eff_bs,
            shuffle=True,
            num_workers=0,
        )
        # dl    = DataLoader(ds, batch_size=42, shuffle=True, num_workers=0)

        aug = torch.nn.Sequential(
            T.RandomGrayscale(0.2), T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(), T.RandomResizedCrop((256, 256)),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        )
        learner = BYOL(models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
                       image_size=256, hidden_layer='avgpool',
                       augment_fn=aug).to(self.device).double()
        opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

        for ep in range(self.epoch_num):
            for img in tqdm(dl, desc=f"BYOL epoch {ep+1}"):
                img = img.to(self.device).double()
                loss = learner(img)
                opt.zero_grad(); loss.backward(); opt.step()
                learner.update_moving_average()
        learner.eval()
        return learner, ds

    # ---------- 4. 推理 ----------
    def _infer_embeddings(self, learner, dataset):
        embs = []
        for img in tqdm(dataset, desc="Infer embeddings"):
            img = img.to(self.device).double().unsqueeze(0)
            with torch.no_grad():
                _, e = learner(img, return_embedding=True)
            embs.append(e.cpu().numpy())
        return np.vstack(embs)


__all__ = ["VisionEmbed"]
