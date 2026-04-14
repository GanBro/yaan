import os
import glob
import json
import random
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# 1. 配置
# =========================
class Config:
    # 数据路径
    data_dir = "/mnt/hdd1/wubo/projects/yaan_precip_4to2/pre_region_npz"
    save_dir = "/mnt/hdd1/wubo/projects/yaan_precip_4to2/checkpoints_4to2_stratified"

    # 新划分 jsonl 路径（优先使用）
    split_dir = "/mnt/hdd1/wubo/projects/yaan_precip_4to2/split_rebuild_results"
    train_jsonl = os.path.join(split_dir, "train_samples.jsonl")
    val_jsonl = os.path.join(split_dir, "val_samples.jsonl")
    test_jsonl = os.path.join(split_dir, "test_samples.jsonl")

    # 输入输出长度
    input_len = 4
    pred_len = 2

    # 旧版划分比例（当 jsonl 不存在时 fallback）
    train_ratio = 0.7
    val_ratio = 0.15

    # 训练参数
    batch_size = 8
    num_workers = 4
    epochs = 50
    lr = 1e-3
    weight_decay = 1e-5
    clip_grad = 5.0
    early_stop_patience = 8

    # 模型参数
    in_channels = 4
    out_channels = 2
    base_channels = 16

    # 其他
    use_log1p = True
    use_target_mask_loss = True
    eval_with_target_mask = True   
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CSI 阈值
    csi_threshold = 0.1


cfg = Config()


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


def print_device_info():
    print("=" * 50)
    print(f"Using device: {cfg.device}")

    if torch.cuda.is_available():
        print("CUDA available: True")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA available: False")
        print("Running on CPU")
    print("=" * 50)


def read_jsonl_samples(jsonl_path: str) -> List[Tuple[List[str], List[str]]]:
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            in_files = row["in_files"]
            out_files = row["out_files"]
            samples.append((in_files, out_files))
    return samples


def summarize_samples(samples: List[Tuple[List[str], List[str]]], name: str):
    print(f"{name} samples = {len(samples)}")
    if len(samples) == 0:
        return

    first_in, first_out = samples[0]
    last_in, last_out = samples[-1]

    print(f"{name} first out = {os.path.basename(first_out[0]).replace('.npz', '')}")
    print(f"{name} last  out = {os.path.basename(last_out[-1]).replace('.npz', '')}")


# =========================
# 3. Dataset
# =========================
class PrecipDataset4to2(Dataset):
    def __init__(
        self,
        file_list=None,
        input_len=4,
        pred_len=2,
        use_log1p=True,
        prebuilt_samples=None,
    ):
        self.input_len = input_len
        self.pred_len = pred_len
        self.use_log1p = use_log1p

        if prebuilt_samples is not None:
            self.samples = prebuilt_samples
        else:
            self.file_list = sorted(file_list)
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

        pre = np.nan_to_num(pre, nan=0.0, posinf=0.0, neginf=0.0)
        pre = np.clip(pre, 0.0, None)
        return pre, target_mask

    def __getitem__(self, idx):
        in_files, out_files = self.samples[idx]

        x_list = []
        y_list = []
        target_mask = None

        for f in in_files:
            pre, mask = self.load_npz(f)
            x_list.append(pre)
            if target_mask is None:
                target_mask = mask

        for f in out_files:
            pre, _ = self.load_npz(f)
            y_list.append(pre)

        x = np.stack(x_list, axis=0)   # [4,H,W]
        y = np.stack(y_list, axis=0)   # [2,H,W]

        if self.use_log1p:
            x = np.log1p(x)
            y = np.log1p(y)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        target_mask = torch.from_numpy(target_mask).float()

        return x, y, target_mask


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
# 5. Loss 和指标
# =========================
class MaskedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask=None):
        diff = torch.abs(pred - target)

        if mask is None:
            return diff.mean()

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        mask = mask.unsqueeze(1)  # [B,1,H,W]
        diff = diff * mask
        denom = torch.clamp(mask.sum() * pred.size(1), min=1.0)
        return diff.sum() / denom


def inverse_transform(x, use_log1p=True):
    if use_log1p:
        return torch.expm1(x)
    return x


@torch.no_grad()
def calc_metrics(pred, target, use_log1p=True, mask=None):
    pred_real = inverse_transform(pred, use_log1p)
    target_real = inverse_transform(target, use_log1p)

    pred_real = torch.clamp(pred_real, min=0.0)
    target_real = torch.clamp(target_real, min=0.0)

    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        mask = mask.unsqueeze(1)
        mae = (torch.abs(pred_real - target_real) * mask).sum() / torch.clamp(mask.sum() * pred_real.size(1), min=1.0)
        rmse = torch.sqrt((((pred_real - target_real) ** 2) * mask).sum() / torch.clamp(mask.sum() * pred_real.size(1), min=1.0))
    else:
        mae = torch.mean(torch.abs(pred_real - target_real))
        rmse = torch.sqrt(torch.mean((pred_real - target_real) ** 2))

    return mae.item(), rmse.item()


@torch.no_grad()
def calc_csi(pred, target, threshold=0.1, use_log1p=True, mask=None):
    pred_real = inverse_transform(pred, use_log1p)
    target_real = inverse_transform(target, use_log1p)

    pred_real = torch.clamp(pred_real, min=0.0)
    target_real = torch.clamp(target_real, min=0.0)

    pred_event = (pred_real >= threshold)
    target_event = (target_real >= threshold)

    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        mask = mask.unsqueeze(1).bool()

        pred_event = pred_event & mask
        target_event = target_event & mask

    tp = (pred_event & target_event).sum().float()
    fp = (pred_event & (~target_event)).sum().float()
    fn = ((~pred_event) & target_event).sum().float()

    denom = tp + fp + fn
    if denom.item() == 0:
        return 1.0

    return (tp / denom).item()


# =========================
# 6. train / val
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device, use_mask_loss=False):
    model.train()
    total_loss = 0.0

    for x, y, target_mask in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        target_mask = target_mask.to(device, non_blocking=True)

        optimizer.zero_grad()
        pred = model(x)

        if use_mask_loss:
            loss = criterion(pred, y, target_mask)
        else:
            loss = criterion(pred, y, None)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
    use_log1p=True,
    use_mask_loss=False,
    csi_threshold=0.1,
    eval_with_target_mask=False,
):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_csi = 0.0
    total_count = 0

    for x, y, target_mask in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        target_mask = target_mask.to(device, non_blocking=True)

        pred = model(x)

        loss_mask = target_mask if use_mask_loss else None
        metric_mask = target_mask if eval_with_target_mask else None

        loss = criterion(pred, y, loss_mask)
        mae, rmse = calc_metrics(pred, y, use_log1p, metric_mask)
        csi = calc_csi(pred, y, threshold=csi_threshold, use_log1p=use_log1p, mask=metric_mask)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_mae += mae * bs
        total_rmse += rmse * bs
        total_csi += csi * bs
        total_count += bs

    return (
        total_loss / total_count,
        total_mae / total_count,
        total_rmse / total_count,
        total_csi / total_count,
    )


# =========================
# 7. 构建数据集
# =========================
def build_datasets():
    use_jsonl = (
        os.path.exists(cfg.train_jsonl)
        and os.path.exists(cfg.val_jsonl)
        and os.path.exists(cfg.test_jsonl)
    )

    if use_jsonl:
        print("=" * 50)
        print("Using stratified split from JSONL files")
        print(f"train_jsonl = {cfg.train_jsonl}")
        print(f"val_jsonl   = {cfg.val_jsonl}")
        print(f"test_jsonl  = {cfg.test_jsonl}")
        print("=" * 50)

        train_samples = read_jsonl_samples(cfg.train_jsonl)
        val_samples = read_jsonl_samples(cfg.val_jsonl)
        test_samples = read_jsonl_samples(cfg.test_jsonl)

        summarize_samples(train_samples, "train")
        summarize_samples(val_samples, "val")
        summarize_samples(test_samples, "test")

        train_set = PrecipDataset4to2(
            input_len=cfg.input_len,
            pred_len=cfg.pred_len,
            use_log1p=cfg.use_log1p,
            prebuilt_samples=train_samples,
        )
        val_set = PrecipDataset4to2(
            input_len=cfg.input_len,
            pred_len=cfg.pred_len,
            use_log1p=cfg.use_log1p,
            prebuilt_samples=val_samples,
        )
        test_set = PrecipDataset4to2(
            input_len=cfg.input_len,
            pred_len=cfg.pred_len,
            use_log1p=cfg.use_log1p,
            prebuilt_samples=test_samples,
        )
    else:
        print("=" * 50)
        print("JSONL split files not found, fallback to old time-order split")
        print("=" * 50)

        file_list = sorted(glob.glob(os.path.join(cfg.data_dir, "*.npz")))
        assert len(file_list) > cfg.input_len + cfg.pred_len, "数据量太少，无法构造样本"

        train_files, val_files, test_files = split_files_by_time(
            file_list, cfg.train_ratio, cfg.val_ratio
        )

        train_set = PrecipDataset4to2(train_files, cfg.input_len, cfg.pred_len, cfg.use_log1p)
        val_set = PrecipDataset4to2(val_files, cfg.input_len, cfg.pred_len, cfg.use_log1p)
        test_set = PrecipDataset4to2(test_files, cfg.input_len, cfg.pred_len, cfg.use_log1p)

        print(f"train samples = {len(train_set)}")
        print(f"val   samples = {len(val_set)}")
        print(f"test  samples = {len(test_set)}")

    return train_set, val_set, test_set


# =========================
# 8. 主函数
# =========================
def main():
    set_seed(42)
    os.makedirs(cfg.save_dir, exist_ok=True)

    print_device_info()

    train_set, val_set, test_set = build_datasets()

    pin_memory = True if cfg.device == "cuda" else False

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory
    )

    model = SmallUNet(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        base_ch=cfg.base_channels
    ).to(cfg.device)

    criterion = MaskedL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    early_stop_count = 0
    best_path = os.path.join(cfg.save_dir, "best_model.pth")

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, cfg.device, cfg.use_target_mask_loss
        )

        val_loss, val_mae, val_rmse, val_csi = validate_one_epoch(
            model,
            val_loader,
            criterion,
            cfg.device,
            cfg.use_log1p,
            cfg.use_target_mask_loss,
            cfg.csi_threshold,
            cfg.eval_with_target_mask,
        )

        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch:03d}/{cfg.epochs}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"val_mae={val_mae:.6f} "
            f"val_rmse={val_rmse:.6f} "
            f"val_csi@{cfg.csi_threshold}={val_csi:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_count = 0
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model: {best_path}")
        else:
            early_stop_count += 1
            if early_stop_count >= cfg.early_stop_patience:
                print("Early stopping triggered.")
                break

    print("\n===== Testing =====")
    model.load_state_dict(torch.load(best_path, map_location=cfg.device, weights_only=True))
    test_loss, test_mae, test_rmse, test_csi = validate_one_epoch(
        model,
        test_loader,
        criterion,
        cfg.device,
        cfg.use_log1p,
        cfg.use_target_mask_loss,
        cfg.csi_threshold,
        cfg.eval_with_target_mask,
    )

    print(
        f"Test: loss={test_loss:.6f}, "
        f"mae={test_mae:.6f}, "
        f"rmse={test_rmse:.6f}, "
        f"csi@{cfg.csi_threshold}={test_csi:.6f}"
    )


if __name__ == "__main__":
    main()