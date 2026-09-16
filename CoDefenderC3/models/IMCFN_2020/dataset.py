"""
IMCFN Dataset — PE binary → Color Image (Jet Colormap) → Tensor
Vasan et al. 2020

将 PE 二进制文件转换为 jet 彩色可视化图:
  1. 读取 raw bytes → reshape 为灰度图
  2. 应用 matplotlib jet colormap → RGB
  3. Resize 到 img_size × img_size
  4. (可选) ImageNet normalize
"""
import os
import math
import hashlib
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def _pe_to_color_image(pe_path, img_size=224, colormap="jet"):
    """将 PE 二进制文件转换为彩色可视化图像"""
    with open(pe_path, "rb") as f:
        bytez = np.frombuffer(f.read(), dtype=np.uint8)

    if len(bytez) == 0:
        return None

    # 计算最佳宽度 (Nataraj 2011 的宽度选择策略)
    file_size = len(bytez)
    if file_size < 10 * 1024:
        width = 32
    elif file_size < 30 * 1024:
        width = 64
    elif file_size < 60 * 1024:
        width = 128
    elif file_size < 100 * 1024:
        width = 256
    elif file_size < 200 * 1024:
        width = 384
    elif file_size < 500 * 1024:
        width = 512
    elif file_size < 1000 * 1024:
        width = 768
    else:
        width = 1024

    height = math.ceil(len(bytez) / width)
    padded = np.zeros(width * height, dtype=np.uint8)
    padded[:len(bytez)] = bytez
    gray = padded.reshape(height, width)

    # 应用 colormap
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(colormap)
        colored = (cmap(gray / 255.0)[:, :, :3] * 255).astype(np.uint8)
    except Exception:
        # fallback: 灰度转 3ch
        colored = np.stack([gray, gray, gray], axis=2)

    img = Image.fromarray(colored)
    img = img.resize((img_size, img_size), Image.BILINEAR)
    return img


class IMCFNImageDataset(Dataset):
    """
    IMCFN 图像数据集:
      PE binary → jet colormap image → tensor
      支持磁盘缓存避免重复转换
    """
    def __init__(self, samples, img_size=224, cache_dir=None,
                 colormap="jet", imagenet_normalize=True):
        """
        Args:
            samples: [(pe_path, label), ...]
            img_size: 输出图像尺寸
            cache_dir: 缓存目录 (None = 不缓存)
            colormap: matplotlib colormap 名称
            imagenet_normalize: 是否使用 ImageNet 均值/标准差归一化
        """
        self.samples = samples
        self.img_size = img_size
        self.cache_dir = cache_dir
        self.colormap = colormap

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        # 默认 transform
        transform_list = [transforms.ToTensor()]
        if imagenet_normalize:
            transform_list.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]))
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.samples)

    def _get_cache_path(self, pe_path):
        if self.cache_dir is None:
            return None
        h = hashlib.md5(pe_path.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.png")

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        cache_path = self._get_cache_path(path)

        # 尝试从缓存加载
        img = None
        if cache_path and os.path.exists(cache_path):
            try:
                img = Image.open(cache_path).convert("RGB")
            except Exception:
                img = None

        # 生成图像
        if img is None:
            try:
                img = _pe_to_color_image(path, self.img_size, self.colormap)
                if img is not None and cache_path:
                    img.save(cache_path)
            except Exception:
                img = None

        # fallback: 黑色图
        if img is None:
            img = Image.new("RGB", (self.img_size, self.img_size))

        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size))

        tensor = self.transform(img)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return tensor, label_tensor
