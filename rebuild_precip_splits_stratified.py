import os
import glob
import json
import random
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

import numpy as np


class Config:
    # 原始 npz 数据目录
    data_dir = "/mnt/hdd1/wubo/projects/yaan_precip_4to2/pre_region_npz"

    # 输出目录
    save_dir = "/mnt/hdd1/wubo/projects/yaan_precip_4to2/split_rebuild_results"

    # 与原训练保持一致
    input_len = 4
    pred_len = 2

    # 新划分比例
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    # 事件定义
    event_threshold = 0.1

    # 是否只用 target_mask 内区域做分层统计
    use_target_mask_only = False

    # 样本强度分桶边界（基于未来两帧 event_ratio）
    # [0, b1) = no_rain_like
    # [b1, b2) = weak
    # [b2, b3) = moderate
    # [b3, 1] = heavy
    event_ratio_bins = [0.001, 0.03, 0.15]

    # 为了减少相邻窗口大量重叠带来的泄漏，先按连续时间块分组后再分配
    # 例如 block_size=24，表示每 24 个样本窗口作为一个 block
    block_size = 24

    # 随机种子
    seed = 42


cfg = Config()
os.makedirs(cfg.save_dir, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def parse_time_from_filename(path: str) -> datetime:
    name = os.path.basename(path).replace(".npz", "")
    return datetime.strptime(name, "%Y%m%d%H")


def is_consecutive_hours(times: List[datetime]) -> bool:
    for i in range(1, len(times)):
        if times[i] - times[i - 1] != timedelta(hours=1):
            return False
    return True


def load_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    pre = data["PRE"].astype(np.float32)
    target_mask = data["target_mask"].astype(np.float32)

    pre = np.nan_to_num(pre, nan=0.0, posinf=0.0, neginf=0.0)
    pre = np.clip(pre, 0.0, None)
    return pre, target_mask


def build_samples(file_list: List[str], input_len: int, pred_len: int):
    file_list = sorted(file_list)
    total_need = input_len + pred_len
    samples = []

    for start in range(len(file_list) - total_need + 1):
        seq_files = file_list[start:start + total_need]
        seq_times = [parse_time_from_filename(f) for f in seq_files]
        if not is_consecutive_hours(seq_times):
            continue

        in_files = seq_files[:input_len]
        out_files = seq_files[input_len:]
        samples.append((in_files, out_files))

    return samples


def compute_sample_stats(sample: Tuple[List[str], List[str]]) -> Dict:
    in_files, out_files = sample

    y_list = []
    target_mask = None
    for f in out_files:
        pre, mask = load_npz(f)
        y_list.append(pre)
        if target_mask is None:
            target_mask = mask

    y = np.stack(y_list, axis=0)  # [pred_len, H, W]

    if cfg.use_target_mask_only:
        valid = target_mask > 0.5
        valid3 = np.broadcast_to(valid[None, :, :], y.shape)
        arr = y[valid3]
    else:
        arr = y.reshape(-1)

    sample_sum = float(arr.sum())
    sample_mean = float(arr.mean()) if arr.size > 0 else 0.0
    sample_max = float(arr.max()) if arr.size > 0 else 0.0
    event_ratio = float((arr >= cfg.event_threshold).mean()) if arr.size > 0 else 0.0
    nonzero_ratio = float((arr > 0).mean()) if arr.size > 0 else 0.0

    out_times = [parse_time_from_filename(f) for f in out_files]
    center_time = out_times[0]

    return {
        "in_files": in_files,
        "out_files": out_files,
        "out_start": out_times[0].strftime("%Y%m%d%H"),
        "out_end": out_times[-1].strftime("%Y%m%d%H"),
        "center_time": center_time,
        "sample_sum": sample_sum,
        "sample_mean": sample_mean,
        "sample_max": sample_max,
        "event_ratio": event_ratio,
        "nonzero_ratio": nonzero_ratio,
    }


def assign_bucket(event_ratio: float) -> str:
    b1, b2, b3 = cfg.event_ratio_bins
    if event_ratio < b1:
        return "no_rain_like"
    if event_ratio < b2:
        return "weak"
    if event_ratio < b3:
        return "moderate"
    return "heavy"


def build_blocks(sample_stats: List[Dict], block_size: int):
    blocks = []
    for i in range(0, len(sample_stats), block_size):
        chunk = sample_stats[i:i + block_size]
        if not chunk:
            continue

        mean_event_ratio = float(np.mean([x["event_ratio"] for x in chunk]))
        mean_sample_sum = float(np.mean([x["sample_sum"] for x in chunk]))
        max_sample_max = float(np.max([x["sample_max"] for x in chunk]))
        bucket = assign_bucket(mean_event_ratio)

        blocks.append({
            "block_id": len(blocks),
            "bucket": bucket,
            "samples": chunk,
            "size": len(chunk),
            "mean_event_ratio": mean_event_ratio,
            "mean_sample_sum": mean_sample_sum,
            "max_sample_max": max_sample_max,
            "start_time": chunk[0]["out_start"],
            "end_time": chunk[-1]["out_end"],
        })
    return blocks


def split_blocks_stratified(blocks: List[Dict]):
    grouped = {}
    for blk in blocks:
        grouped.setdefault(blk["bucket"], []).append(blk)

    train_blocks, val_blocks, test_blocks = [], [], []

    for bucket, bucket_blocks in grouped.items():
        random.shuffle(bucket_blocks)
        n = len(bucket_blocks)
        n_train = int(round(n * cfg.train_ratio))
        n_val = int(round(n * cfg.val_ratio))
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        n_test = n - n_train - n_val

        train_blocks.extend(bucket_blocks[:n_train])
        val_blocks.extend(bucket_blocks[n_train:n_train + n_val])
        test_blocks.extend(bucket_blocks[n_train + n_val:])

    random.shuffle(train_blocks)
    random.shuffle(val_blocks)
    random.shuffle(test_blocks)
    return train_blocks, val_blocks, test_blocks


def flatten_blocks(blocks: List[Dict]) -> List[Dict]:
    out = []
    for blk in blocks:
        out.extend(blk["samples"])
    return out


def summarize_split(samples: List[Dict]) -> Dict:
    if not samples:
        return {"num_samples": 0}

    sums = np.array([x["sample_sum"] for x in samples], dtype=np.float64)
    means = np.array([x["sample_mean"] for x in samples], dtype=np.float64)
    maxs = np.array([x["sample_max"] for x in samples], dtype=np.float64)
    events = np.array([x["event_ratio"] for x in samples], dtype=np.float64)
    nonzeros = np.array([x["nonzero_ratio"] for x in samples], dtype=np.float64)

    bucket_counts = {}
    for x in samples:
        b = assign_bucket(x["event_ratio"])
        bucket_counts[b] = bucket_counts.get(b, 0) + 1

    return {
        "num_samples": int(len(samples)),
        "sample_sum_mean": float(np.mean(sums)),
        "sample_sum_median": float(np.median(sums)),
        "sample_sum_p90": float(np.percentile(sums, 90)),
        "sample_mean_mean": float(np.mean(means)),
        "sample_max_mean": float(np.mean(maxs)),
        "sample_max_p95": float(np.percentile(maxs, 95)),
        "event_ratio_mean": float(np.mean(events)),
        "event_ratio_median": float(np.median(events)),
        "event_ratio_p90": float(np.percentile(events, 90)),
        "nonzero_ratio_mean": float(np.mean(nonzeros)),
        "bucket_counts": bucket_counts,
        "time_start": samples[0]["out_start"],
        "time_end": samples[-1]["out_end"],
    }


def write_sample_list(path: str, samples: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for i, s in enumerate(samples):
            row = {
                "new_sample_index": i,
                "out_start": s["out_start"],
                "out_end": s["out_end"],
                "sample_sum": s["sample_sum"],
                "sample_mean": s["sample_mean"],
                "sample_max": s["sample_max"],
                "event_ratio": s["event_ratio"],
                "nonzero_ratio": s["nonzero_ratio"],
                "in_files": s["in_files"],
                "out_files": s["out_files"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: str, original_summary: Dict, new_summary: Dict, blocks_summary: Dict):
    lines = []
    lines.append("重划分报告（按事件强度分层 + 时间块分配）")
    lines.append("=" * 88)
    lines.append(f"data_dir = {cfg.data_dir}")
    lines.append(f"input_len = {cfg.input_len}")
    lines.append(f"pred_len = {cfg.pred_len}")
    lines.append(f"event_threshold = {cfg.event_threshold}")
    lines.append(f"use_target_mask_only = {cfg.use_target_mask_only}")
    lines.append(f"event_ratio_bins = {cfg.event_ratio_bins}")
    lines.append(f"block_size = {cfg.block_size}")
    lines.append("")

    lines.append("一、原始样本整体概况")
    lines.append(json.dumps(original_summary, ensure_ascii=False, indent=2))
    lines.append("")

    lines.append("二、block 分布概况")
    lines.append(json.dumps(blocks_summary, ensure_ascii=False, indent=2))
    lines.append("")

    lines.append("三、新划分后各集合概况")
    lines.append(json.dumps(new_summary, ensure_ascii=False, indent=2))
    lines.append("")

    lines.append("四、说明")
    lines.append("1. 该脚本并非随机打散单个样本，而是先按连续窗口构成时间块，再做分层分配，以减少相邻重叠样本泄漏。")
    lines.append("2. 分层依据为未来两帧 event_ratio；可按任务需要改成 sample_sum 或 target_mask 内 event_ratio。")
    lines.append("3. 如果目标是严格时间外推评估，则不应使用该方案作为最终论文主划分，而更适合作为分布平衡评估或辅助实验。")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    set_seed(cfg.seed)

    file_list = sorted(glob.glob(os.path.join(cfg.data_dir, "*.npz")))
    assert len(file_list) > cfg.input_len + cfg.pred_len, "数据量太少，无法构造样本"

    raw_samples = build_samples(file_list, cfg.input_len, cfg.pred_len)
    sample_stats = [compute_sample_stats(s) for s in raw_samples]

    original_summary = summarize_split(sample_stats)

    blocks = build_blocks(sample_stats, cfg.block_size)
    blocks_summary = {
        "num_blocks": len(blocks),
        "bucket_counts": {},
    }
    for blk in blocks:
        b = blk["bucket"]
        blocks_summary["bucket_counts"][b] = blocks_summary["bucket_counts"].get(b, 0) + 1

    train_blocks, val_blocks, test_blocks = split_blocks_stratified(blocks)

    train_samples = flatten_blocks(train_blocks)
    val_samples = flatten_blocks(val_blocks)
    test_samples = flatten_blocks(test_blocks)

    new_summary = {
        "train": summarize_split(train_samples),
        "val": summarize_split(val_samples),
        "test": summarize_split(test_samples),
    }

    write_sample_list(os.path.join(cfg.save_dir, "train_samples.jsonl"), train_samples)
    write_sample_list(os.path.join(cfg.save_dir, "val_samples.jsonl"), val_samples)
    write_sample_list(os.path.join(cfg.save_dir, "test_samples.jsonl"), test_samples)

    write_report(
        os.path.join(cfg.save_dir, "split_rebuild_report.txt"),
        original_summary,
        new_summary,
        blocks_summary,
    )

    print("=" * 88)
    print("重划分完成")
    print(f"输出目录: {cfg.save_dir}")
    print("已生成:")
    print("  - train_samples.jsonl")
    print("  - val_samples.jsonl")
    print("  - test_samples.jsonl")
    print("  - split_rebuild_report.txt")
    print("=" * 88)


if __name__ == "__main__":
    main()