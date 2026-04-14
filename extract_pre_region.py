import os
import re
import glob
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import xarray as xr


# =========================
# 1. 配置区
# =========================

# 原始 GRIB2 数据根目录
ROOT_DIR = "/hdd/nas/disk4/b103/wubo/DATA/YaanData/汉源气象资料/r_hor/2025/2025"

# 预处理输出目录
SAVE_DIR = "/mnt/hdd1/wubo/projects/yaan_precip_4to2/pre_region_npz"

# 胡老师项目目标区域
TARGET_LON_MIN = 102.41432366
TARGET_LAT_MIN = 29.44775911
TARGET_LON_MAX = 102.50608120
TARGET_LAT_MAX = 29.54251692

# 是否扩展输入区域
USE_EXPANDED_REGION = True
EXPAND_DEGREE = 0.10

if USE_EXPANDED_REGION:
    CROP_LON_MIN = TARGET_LON_MIN - EXPAND_DEGREE
    CROP_LAT_MIN = TARGET_LAT_MIN - EXPAND_DEGREE
    CROP_LON_MAX = TARGET_LON_MAX + EXPAND_DEGREE
    CROP_LAT_MAX = TARGET_LAT_MAX + EXPAND_DEGREE
else:
    CROP_LON_MIN = TARGET_LON_MIN
    CROP_LAT_MIN = TARGET_LAT_MIN
    CROP_LON_MAX = TARGET_LON_MAX
    CROP_LAT_MAX = TARGET_LAT_MAX

os.makedirs(SAVE_DIR, exist_ok=True)

# 文件名匹配：HOR-PRE-YYYYMMDDHH.GRB2
PATTERN = re.compile(r"HOR-PRE-(\d{10})\.GRB2$")


# =========================
# 2. 收集并去重
# =========================
def collect_unique_files(root_dir: str) -> Dict[str, str]:
    """
    按 HOR-PRE-YYYYMMDDHH 去重。
    同一小时若有多份文件，排序后取最后一个。
    """
    all_files = glob.glob(os.path.join(root_dir, "*", "*.GRB2"))
    groups = defaultdict(list)

    for f in all_files:
        name = os.path.basename(f)
        m = PATTERN.search(name)
        if m is None:
            continue
        time_key = m.group(1)
        groups[time_key].append(f)

    unique = {}
    for time_key, files in groups.items():
        files = sorted(files)
        unique[time_key] = files[-1]

    return dict(sorted(unique.items(), key=lambda x: x[0]))


# =========================
# 3. 读取降水 PRE
# =========================
def read_grib_pre(grib_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    读取 GRIB2 中的降水量。
    当前这批 r_hor 数据在 cfgrib 中通常显示为 unknown，
    但业务语义上它就是 PRE。
    返回：
        pre, lat, lon, raw_var_name
    """
    ds = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""}
    )

    data_vars = list(ds.data_vars)

    if "unknown" in ds.data_vars:
        raw_var_name = "unknown"
    elif "PRE" in ds.data_vars:
        raw_var_name = "PRE"
    else:
        raise KeyError(
            f"Cannot find precipitation variable. Available data_vars: {data_vars}"
        )

    pre = ds[raw_var_name].values.astype(np.float32)

    if "latitude" in ds.coords:
        lat = ds["latitude"].values
    elif "lat" in ds.coords:
        lat = ds["lat"].values
    else:
        raise KeyError("Cannot find latitude coordinates.")

    if "longitude" in ds.coords:
        lon = ds["longitude"].values
    elif "lon" in ds.coords:
        lon = ds["lon"].values
    else:
        raise KeyError("Cannot find longitude coordinates.")

    return pre, lat, lon, raw_var_name


# =========================
# 4. 裁剪区域
# =========================
def crop_region(
    pre: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float
):
    """
    支持：
    1) lat:[H], lon:[W]
    2) lat:[H,W], lon:[H,W]
    """
    if lat.ndim == 1 and lon.ndim == 1:
        lat_mask = (lat >= lat_min) & (lat <= lat_max)
        lon_mask = (lon >= lon_min) & (lon <= lon_max)

        if lat_mask.sum() == 0 or lon_mask.sum() == 0:
            raise ValueError("No grid falls inside crop region.")

        pre_crop = pre[np.ix_(lat_mask, lon_mask)]
        lat_crop = lat[lat_mask]
        lon_crop = lon[lon_mask]

    elif lat.ndim == 2 and lon.ndim == 2:
        mask = (
            (lon >= lon_min) & (lon <= lon_max) &
            (lat >= lat_min) & (lat <= lat_max)
        )

        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]

        if len(rows) == 0 or len(cols) == 0:
            raise ValueError("No grid falls inside crop region.")

        r0, r1 = rows[0], rows[-1] + 1
        c0, c1 = cols[0], cols[-1] + 1

        pre_crop = pre[r0:r1, c0:c1]
        lat_crop = lat[r0:r1, c0:c1]
        lon_crop = lon[r0:r1, c0:c1]
    else:
        raise ValueError("Unsupported lat/lon dimensions.")

    return pre_crop, lat_crop, lon_crop


# =========================
# 5. 目标区域 mask
# =========================
def build_target_mask(lat_crop, lon_crop):
    if lat_crop.ndim == 1 and lon_crop.ndim == 1:
        lat2d, lon2d = np.meshgrid(lat_crop, lon_crop, indexing="ij")
    else:
        lat2d, lon2d = lat_crop, lon_crop

    mask = (
        (lon2d >= TARGET_LON_MIN) & (lon2d <= TARGET_LON_MAX) &
        (lat2d >= TARGET_LAT_MIN) & (lat2d <= TARGET_LAT_MAX)
    ).astype(np.float32)

    return mask


# =========================
# 6. 主函数
# =========================
def main():
    unique_files = collect_unique_files(ROOT_DIR)
    print(f"[INFO] unique hours = {len(unique_files)}")

    if len(unique_files) == 0:
        print("[ERROR] No GRB2 files found.")
        return

    for idx, (time_key, grib_path) in enumerate(unique_files.items(), 1):
        save_path = os.path.join(SAVE_DIR, f"{time_key}.npz")

        if os.path.exists(save_path):
            print(f"[{idx}/{len(unique_files)}] skip existing: {save_path}")
            continue

        try:
            pre, lat, lon, raw_var_name = read_grib_pre(grib_path)

            pre = np.nan_to_num(pre, nan=0.0, posinf=0.0, neginf=0.0)
            pre = np.clip(pre, 0.0, None)

            pre_crop, lat_crop, lon_crop = crop_region(
                pre, lat, lon,
                CROP_LON_MIN, CROP_LAT_MIN,
                CROP_LON_MAX, CROP_LAT_MAX
            )

            target_mask = build_target_mask(lat_crop, lon_crop)

            np.savez_compressed(
                save_path,
                PRE=pre_crop.astype(np.float32),            # 统一按业务名保存
                lat=lat_crop.astype(np.float32),
                lon=lon_crop.astype(np.float32),
                target_mask=target_mask.astype(np.float32),
                time=np.array(time_key),
                source=np.array(grib_path),
                raw_var_name=np.array(raw_var_name),        # 原始读取名
                business_var_name=np.array("PRE"),          # 业务变量名
            )

            print(
                f"[{idx}/{len(unique_files)}] saved: {save_path}, "
                f"shape={pre_crop.shape}, raw_var={raw_var_name}, business_var=PRE"
            )

        except Exception as e:
            print(f"[ERROR] {time_key} -> {grib_path}")
            print(f"        {repr(e)}")


if __name__ == "__main__":
    main()