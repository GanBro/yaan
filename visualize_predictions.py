import os
import glob
import random
from datetime import datetime, timedelta
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


# =========================
# 1. 配置
# =========================
class Config:
    # 数据路径
    data_dir = "/mnt/hdd1/wubo/projects/yaan_precip_4to2/pre_region_npz"
    ckpt_path = "/mnt/hdd1/wubo/projects/yaan_precip_4to2/checkpoints_4to2/best_model.pth"

    # 输出目录
    save_dir = "/mnt/hdd1/wubo/projects/yaan_precip_4to2/vis_results"

    # 数据设置
    input_len = 4
    pred_len = 2
    train_ratio = 0.7
    val_ratio = 0.15

    # 模型设置
    in_channels = 4
    out_channels = 2
    base_channels = 16

    # 其他
    use_log1p = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 可视化设置
    sample_index = 0
    save_fig = True
    fig_dpi = 150
    show_target_box = True


cfg = Config()
os.makedirs(cfg.save_dir, exist_ok=True)


# =========================
# 2. 工具函数
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_time_from_filename(path: str) -> datetime:
    name = os.path.basename(path).replace(".npz", "")
    return datetime.strptime(name, "%Y%m%d%H")


def is_consecutive_hours(times: List[datetime]) -> bool:
    for i in range(1, len(times)):
        if times[i] - times[i - 1] != timedelta(hours=1):
            return False
    return True


def split_files_by_time(file_list: List[str], train_ratio=0.7, val_ratio=0.15):
    n = len(file_list)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = file_list[:n_train]
    val_files = file_list[n_train:n_train + n_val]
    test_files = file_list[n_train + n_val:]
    return train_files, val_files, test_files


def inverse_transform(x, use_log1p=True):
    if use_log1p:
        return np.expm1(x)
    return x


def mask_to_bbox(mask: np.ndarray):
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return rows[0], rows[-1], cols[0], cols[-1]


# =========================
# 3. Dataset
# =========================
class PrecipDataset4to2(Dataset):
    def __init__(self, file_list, input_len=4, pred_len=2, use_log1p=True):
        self.file_list = sorted(file_list)
        self.input_len = input_len
        self.pred_len = pred_len
        self.use_log1p = use_log1p

        self.samples = []
        total_need = input_len + pred_len

        for start in range(len(self.file_list) - total_need + 1):
            seq_files = self.file_list[start:start + total_need]
            seq_times = [parse_time_from_filename(f) for f in seq_files]

            if not is_consecutive_hours(seq_times):
                continue

            in_files = seq_files[:input_len]
            out_files = seq_files[input_len:]
            self.samples.append((in_files, out_files))

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_npz(npz_path):
        data = np.load(npz_path)
        pre = data["PRE"].astype(np.float32)
        target_mask = data["target_mask"].astype(np.float32)
        lat = data["lat"]
        lon = data["lon"]

        pre = np.nan_to_num(pre, nan=0.0, posinf=0.0, neginf=0.0)
        pre = np.clip(pre, 0.0, None)
        return pre, target_mask, lat, lon

    def __getitem__(self, idx):
        in_files, out_files = self.samples[idx]

        x_list = []
        y_list = []
        target_mask = None
        lat_ref = None
        lon_ref = None

        for f in in_files:
            pre, mask, lat, lon = self.load_npz(f)
            x_list.append(pre)
            if target_mask is None:
                target_mask = mask
            if lat_ref is None:
                lat_ref = lat
            if lon_ref is None:
                lon_ref = lon

        for f in out_files:
            pre, _, _, _ = self.load_npz(f)
            y_list.append(pre)

        x = np.stack(x_list, axis=0)   # [4,H,W]
        y = np.stack(y_list, axis=0)   # [2,H,W]

        if self.use_log1p:
            x = np.log1p(x)
            y = np.log1p(y)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        target_mask = torch.from_numpy(target_mask).float()

        meta = {
            "in_files": in_files,
            "out_files": out_files,
            "lat": lat_ref,
            "lon": lon_ref,
        }

        return x, y, target_mask, meta


# =========================
# 4. U-Net
# =========================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        x = nn.functional.pad(
            x,
            [diff_x // 2, diff_x - diff_x // 2,
             diff_y // 2, diff_y - diff_y // 2]
        )

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class SmallUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=2, base_ch=16):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.bottleneck = DoubleConv(base_ch * 4, base_ch * 8)
        self.up1 = Up(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up2 = Up(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up3 = Up(base_ch * 2, base_ch, base_ch)
        self.outc = nn.Conv2d(base_ch, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.bottleneck(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x


# =========================
# 5. 绘图
# =========================
def draw_bbox(ax, bbox, color="red", linewidth=1.2):
    if bbox is None:
        return
    r0, r1, c0, c1 = bbox
    xs = [c0, c1, c1, c0, c0]
    ys = [r0, r0, r1, r1, r0]
    ax.plot(xs, ys, color=color, linewidth=linewidth)


def visualize_sample(model, dataset, sample_index=0):
    model.eval()

    x, y, target_mask, meta = dataset[sample_index]
    x_in = x.unsqueeze(0).to(cfg.device)

    with torch.no_grad():
        pred = model(x_in).cpu().numpy()[0]   # [2,H,W]

    x_np = x.numpy()
    y_np = y.numpy()

    x_real = inverse_transform(x_np, cfg.use_log1p)
    y_real = inverse_transform(y_np, cfg.use_log1p)
    pred_real = inverse_transform(pred, cfg.use_log1p)
    pred_real = np.clip(pred_real, 0.0, None)

    target_mask_np = target_mask.numpy()
    bbox = mask_to_bbox(target_mask_np)

    vmax = max(
        float(x_real.max()) if x_real.size > 0 else 0.0,
        float(y_real.max()) if y_real.size > 0 else 0.0,
        float(pred_real.max()) if pred_real.size > 0 else 0.0,
    )
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    titles = [
        f"Input t-3\n{os.path.basename(meta['in_files'][0]).replace('.npz','')}",
        f"Input t-2\n{os.path.basename(meta['in_files'][1]).replace('.npz','')}",
        f"Input t-1\n{os.path.basename(meta['in_files'][2]).replace('.npz','')}",
        f"Input t\n{os.path.basename(meta['in_files'][3]).replace('.npz','')}",
        f"GT t+1\n{os.path.basename(meta['out_files'][0]).replace('.npz','')}",
        f"GT t+2\n{os.path.basename(meta['out_files'][1]).replace('.npz','')}",
        "Pred t+1",
        "Pred t+2",
    ]

    images = [
        x_real[0], x_real[1], x_real[2], x_real[3],
        y_real[0], y_real[1],
        pred_real[0], pred_real[1]
    ]

    im_ref = None
    for ax, title, img in zip(axes, titles, images):
        im = ax.imshow(img, cmap="Blues", vmin=0.0, vmax=vmax)
        if im_ref is None:
            im_ref = im
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        if cfg.show_target_box:
            draw_bbox(ax, bbox, color="red", linewidth=1.2)

    cbar = fig.colorbar(im_ref, ax=axes.tolist(), shrink=0.85)
    cbar.set_label("Precipitation", rotation=90)

    fig.suptitle(
        f"4-hour Inputs -> 2-hour Forecasts | sample_index={sample_index}",
        fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if cfg.save_fig:
        save_path = os.path.join(cfg.save_dir, f"sample_{sample_index:04d}.png")
        plt.savefig(save_path, dpi=cfg.fig_dpi, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    plt.show()


# =========================
# 6. 主函数
# =========================
def main():
    set_seed(42)

    file_list = sorted(glob.glob(os.path.join(cfg.data_dir, "*.npz")))
    assert len(file_list) > cfg.input_len + cfg.pred_len, "数据量太少，无法构造样本"

    _, _, test_files = split_files_by_time(
        file_list,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio
    )

    test_set = PrecipDataset4to2(
        test_files,
        input_len=cfg.input_len,
        pred_len=cfg.pred_len,
        use_log1p=cfg.use_log1p
    )

    print(f"test samples = {len(test_set)}")
    assert len(test_set) > 0, "测试集为空，请检查数据划分或连续性。"

    if cfg.sample_index >= len(test_set):
        raise IndexError(
            f"sample_index={cfg.sample_index} 超出测试集范围，"
            f"测试集样本数为 {len(test_set)}"
        )

    model = SmallUNet(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        base_ch=cfg.base_channels
    ).to(cfg.device)

    state_dict = torch.load(cfg.ckpt_path, map_location=cfg.device)
    model.load_state_dict(state_dict)
    print(f"Loaded model from: {cfg.ckpt_path}")

    visualize_sample(model, test_set, cfg.sample_index)


if __name__ == "__main__":
    main()