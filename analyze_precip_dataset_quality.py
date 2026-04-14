import os
import glob
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np


class Config:
    # 与训练脚本保持一致
    data_dir = "/mnt/hdd1/wubo/projects/yaan_precip_4to2/pre_region_npz"
    input_len = 4
    pred_len = 2
    train_ratio = 0.7
    val_ratio = 0.15

    # 事件阈值（与训练脚本 CSI 阈值一致）
    event_threshold = 0.1

    # 是否只在目标区域内统计
    use_target_mask_only = False

    # 输出目录
    save_dir = "/mnt/hdd1/wubo/projects/yaan_precip_4to2/analysis_results"

    # 可视化候选样本输出数量
    topk_heavy_samples = 30
    topk_event_samples = 30


cfg = Config()
os.makedirs(cfg.save_dir, exist_ok=True)


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


class SequenceBuilder:
    def __init__(self, file_list: List[str], input_len: int, pred_len: int):
        self.file_list = sorted(file_list)
        self.input_len = input_len
        self.pred_len = pred_len
        self.samples = self._build_samples()

    def _build_samples(self):
        total_need = self.input_len + self.pred_len
        samples = []
        for start in range(len(self.file_list) - total_need + 1):
            seq_files = self.file_list[start:start + total_need]
            seq_times = [parse_time_from_filename(f) for f in seq_files]
            if not is_consecutive_hours(seq_times):
                continue
            in_files = seq_files[:self.input_len]
            out_files = seq_files[self.input_len:]
            samples.append((in_files, out_files))
        return samples


def load_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    pre = data["PRE"].astype(np.float32)
    target_mask = data["target_mask"].astype(np.float32)

    pre = np.nan_to_num(pre, nan=0.0, posinf=0.0, neginf=0.0)
    pre = np.clip(pre, 0.0, None)
    return pre, target_mask


def summarize_array(arr: np.ndarray, prefix: str = ""):
    flat = arr.reshape(-1)
    return {
        f"{prefix}count": int(flat.size),
        f"{prefix}mean": float(np.mean(flat)),
        f"{prefix}std": float(np.std(flat)),
        f"{prefix}min": float(np.min(flat)),
        f"{prefix}p50": float(np.percentile(flat, 50)),
        f"{prefix}p90": float(np.percentile(flat, 90)),
        f"{prefix}p95": float(np.percentile(flat, 95)),
        f"{prefix}p99": float(np.percentile(flat, 99)),
        f"{prefix}max": float(np.max(flat)),
    }


def format_stats(title: str, stats: dict) -> str:
    lines = [title]
    for k, v in stats.items():
        if isinstance(v, float):
            lines.append(f"  {k}: {v:.6f}")
        else:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def analyze_split(split_name: str, sample_pairs: List[Tuple[List[str], List[str]]], threshold: float, use_target_mask_only: bool):
    sample_infos = []

    pixel_values_all = []
    event_ratio_all = []
    sample_sum_all = []
    sample_max_all = []

    total_pixels = 0
    total_event_pixels = 0
    total_nonzero_pixels = 0

    for idx, (in_files, out_files) in enumerate(sample_pairs):
        y_list = []
        target_mask = None

        for f in out_files:
            pre, mask = load_npz(f)
            y_list.append(pre)
            if target_mask is None:
                target_mask = mask

        y = np.stack(y_list, axis=0)  # [pred_len, H, W]

        if use_target_mask_only:
            valid = target_mask > 0.5
            valid3 = np.broadcast_to(valid[None, :, :], y.shape)
            arr = y[valid3]
        else:
            arr = y.reshape(-1)

        if arr.size == 0:
            continue

        sample_sum = float(arr.sum())
        sample_mean = float(arr.mean())
        sample_max = float(arr.max())
        event_ratio = float((arr >= threshold).mean())
        nonzero_ratio = float((arr > 0).mean())

        total_pixels += int(arr.size)
        total_event_pixels += int((arr >= threshold).sum())
        total_nonzero_pixels += int((arr > 0).sum())

        pixel_values_all.append(arr)
        event_ratio_all.append(event_ratio)
        sample_sum_all.append(sample_sum)
        sample_max_all.append(sample_max)

        out_times = [os.path.basename(x).replace('.npz', '') for x in out_files]
        sample_infos.append({
            "sample_index": idx,
            "out_start": out_times[0],
            "out_end": out_times[-1],
            "sample_sum": sample_sum,
            "sample_mean": sample_mean,
            "sample_max": sample_max,
            "event_ratio": event_ratio,
            "nonzero_ratio": nonzero_ratio,
        })

    if not pixel_values_all:
        return None

    all_pixels = np.concatenate(pixel_values_all, axis=0)
    stats = summarize_array(all_pixels, prefix="pixel_")
    stats.update({
        "num_samples": len(sample_infos),
        "global_event_ratio": float(total_event_pixels / max(total_pixels, 1)),
        "global_nonzero_ratio": float(total_nonzero_pixels / max(total_pixels, 1)),
        "mean_sample_sum": float(np.mean(sample_sum_all)),
        "median_sample_sum": float(np.median(sample_sum_all)),
        "mean_sample_max": float(np.mean(sample_max_all)),
        "median_sample_max": float(np.median(sample_max_all)),
        "mean_event_ratio": float(np.mean(event_ratio_all)),
        "median_event_ratio": float(np.median(event_ratio_all)),
    })

    sample_infos_sorted_by_sum = sorted(sample_infos, key=lambda x: x["sample_sum"], reverse=True)
    sample_infos_sorted_by_event = sorted(sample_infos, key=lambda x: x["event_ratio"], reverse=True)

    return {
        "stats": stats,
        "samples_by_sum": sample_infos_sorted_by_sum,
        "samples_by_event": sample_infos_sorted_by_event,
        "all_samples": sample_infos,
    }


def save_rank_file(path: str, items: List[dict], topk: int, title: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(title + "\n")
        f.write("=" * 80 + "\n")
        for row in items[:topk]:
            f.write(
                f"sample_index={row['sample_index']:4d} | "
                f"out_start={row['out_start']} | out_end={row['out_end']} | "
                f"sum={row['sample_sum']:.6f} | mean={row['sample_mean']:.6f} | "
                f"max={row['sample_max']:.6f} | event_ratio={row['event_ratio']:.6f} | "
                f"nonzero_ratio={row['nonzero_ratio']:.6f}\n"
            )


def main():
    file_list = sorted(glob.glob(os.path.join(cfg.data_dir, "*.npz")))
    assert len(file_list) > cfg.input_len + cfg.pred_len, "数据量太少，无法构造样本"

    train_files, val_files, test_files = split_files_by_time(
        file_list, cfg.train_ratio, cfg.val_ratio
    )

    split_to_files = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    report_lines = []
    report_lines.append("降水数据质量分析报告")
    report_lines.append("=" * 80)
    report_lines.append(f"data_dir = {cfg.data_dir}")
    report_lines.append(f"input_len = {cfg.input_len}")
    report_lines.append(f"pred_len = {cfg.pred_len}")
    report_lines.append(f"event_threshold = {cfg.event_threshold}")
    report_lines.append(f"use_target_mask_only = {cfg.use_target_mask_only}")
    report_lines.append("")

    for split_name, files in split_to_files.items():
        builder = SequenceBuilder(files, cfg.input_len, cfg.pred_len)
        result = analyze_split(
            split_name=split_name,
            sample_pairs=builder.samples,
            threshold=cfg.event_threshold,
            use_target_mask_only=cfg.use_target_mask_only,
        )

        if result is None:
            report_lines.append(f"[{split_name}] 无有效样本")
            report_lines.append("")
            continue

        report_lines.append(format_stats(f"[{split_name}]", result["stats"]))
        report_lines.append("")

        heavy_path = os.path.join(cfg.save_dir, f"{split_name}_top{cfg.topk_heavy_samples}_by_sum.txt")
        event_path = os.path.join(cfg.save_dir, f"{split_name}_top{cfg.topk_event_samples}_by_event_ratio.txt")

        save_rank_file(
            heavy_path,
            result["samples_by_sum"],
            cfg.topk_heavy_samples,
            f"{split_name}: 按未来两帧总降水量排序 Top-{cfg.topk_heavy_samples}"
        )
        save_rank_file(
            event_path,
            result["samples_by_event"],
            cfg.topk_event_samples,
            f"{split_name}: 按未来两帧事件像素占比排序 Top-{cfg.topk_event_samples}"
        )

    report_path = os.path.join(cfg.save_dir, "dataset_quality_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("=" * 80)
    print("分析完成")
    print(f"报告文件: {report_path}")
    print(f"Top 降水样本列表: {cfg.save_dir}/*_top*_by_sum.txt")
    print(f"Top 事件样本列表: {cfg.save_dir}/*_top*_by_event_ratio.txt")
    print("=" * 80)


if __name__ == "__main__":
    main()
