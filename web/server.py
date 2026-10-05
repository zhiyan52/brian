"""
脑萎缩定量分析平台 — FastAPI 后端
（脑萎缩 / 阿尔茨海默早期辅助筛查场景）

功能：
    - 接受 .nii / .nii.gz 或 .zip（DICOM 序列）上传
    - 3D U-Net 滑窗推理 → CSF / GM / WM 分割
    - 定量报告：体积、TIV 归一化比值、脑实质分数 BPF、脑室 CSF 估算
    - 三视图叠加图 + 3D 表面网格（前端交互式渲染）

启动：
    .\\.venv\\Scripts\\python.exe web/server.py
    浏览器打开 http://127.0.0.1:8000
"""

from __future__ import annotations

import base64
import io
import shutil
import sys
import tempfile
import threading
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import load_config, load_model  # noqa: E402
from predict_test import invert_prediction_to_native_space  # noqa: E402
from ibsr_unet.data.transforms import EnsureSingleChannelFirstd  # noqa: E402
from ibsr_unet.inference import sliding_window_predict  # noqa: E402
from monai.transforms import (  # noqa: E402
    Compose,
    CropForegroundd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "train.yaml"
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "best_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PREPROCESS = Compose(
    [
        LoadImaged(keys=["image"], image_only=False),
        EnsureSingleChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=False),
        CropForegroundd(keys=["image"], source_key="image"),
    ]
)

TISSUE_NAMES = {1: "CSF", 2: "GM", 3: "WM"}
TISSUE_COLORS_HEX = {1: "#3399ff", 2: "#ff8c33", 3: "#33e673"}
LABEL_COLORS = {
    1: (0.20, 0.60, 1.00, 0.45),
    2: (1.00, 0.55, 0.20, 0.45),
    3: (0.20, 0.90, 0.45, 0.45),
}

app = FastAPI(title="脑萎缩定量分析平台")
_MODEL = None
_INFER_LOCK = threading.Lock()


# ── 模型 ──────────────────────────────────────────────────────────────


def get_model() -> torch.nn.Module:
    global _MODEL
    if _MODEL is None:
        config = load_config(CONFIG_PATH)
        _MODEL = load_model(config, CHECKPOINT_PATH, DEVICE)
    return _MODEL


# ── DICOM → NIfTI ────────────────────────────────────────────────────


def dicom_zip_to_nifti(zip_path: Path, output_dir: Path) -> Path:
    """解压 DICOM zip，用 SimpleITK 读取序列为 3D 体，保存为 NIfTI。"""
    extract_dir = output_dir / "dcm"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(extract_dir))
    if not series_ids:
        # 递归查找子目录
        for sub in sorted(extract_dir.rglob("*")):
            if sub.is_dir():
                series_ids = reader.GetGDCMSeriesIDs(str(sub))
                if series_ids:
                    extract_dir = sub
                    break
    if not series_ids:
        raise ValueError("zip 中未找到有效 DICOM 序列")

    dcm_files = reader.GetGDCMSeriesFileNames(str(extract_dir), series_ids[0])
    reader.SetFileNames(dcm_files)
    image = reader.Execute()

    nifti_path = output_dir / "converted.nii.gz"
    sitk.WriteImage(image, str(nifti_path))
    return nifti_path


# ── 定量指标 ──────────────────────────────────────────────────────────


# 去颅骨 T1 MRI 的年龄/性别参考范围（综合 ICBM、ADNI、OASIS 文献近似值）
# 指标为体积占 TIV 的百分比；BPF = (GM+WM)/TIV
# 去颅骨后 CSF 仅含脑室+脑沟，故 CSF/TIV 远低于含颅骨全脑数据
REFERENCE_RANGES = {
    # age_bucket: {sex: {metric: (p10, p50, p90)}}  p=百分位
    "20-40": {
        "M": {"GM_pct": (55.0, 60.0, 64.0), "WM_pct": (33.0, 38.0, 43.0),
              "CSF_pct": (0.5, 1.8, 3.0), "BPF": (0.965, 0.982, 0.993)},
        "F": {"GM_pct": (56.0, 61.0, 65.0), "WM_pct": (32.0, 37.0, 42.0),
              "CSF_pct": (0.5, 1.6, 2.8), "BPF": (0.968, 0.984, 0.993)},
    },
    "40-60": {
        "M": {"GM_pct": (50.0, 56.0, 61.0), "WM_pct": (34.0, 38.5, 43.5),
              "CSF_pct": (1.0, 3.0, 5.5), "BPF": (0.940, 0.968, 0.988)},
        "F": {"GM_pct": (51.0, 57.0, 62.0), "WM_pct": (33.0, 37.5, 42.5),
              "CSF_pct": (0.8, 2.7, 5.0), "BPF": (0.943, 0.971, 0.989)},
    },
    "60-75": {
        "M": {"GM_pct": (44.0, 51.0, 57.0), "WM_pct": (34.5, 38.5, 43.0),
              "CSF_pct": (2.0, 5.5, 9.5), "BPF": (0.890, 0.938, 0.975)},
        "F": {"GM_pct": (45.0, 52.0, 58.0), "WM_pct": (33.5, 37.5, 42.0),
              "CSF_pct": (1.8, 5.0, 8.8), "BPF": (0.895, 0.943, 0.977)},
    },
    "75+": {
        "M": {"GM_pct": (38.0, 46.0, 53.0), "WM_pct": (34.0, 38.0, 42.0),
              "CSF_pct": (3.5, 9.0, 15.0), "BPF": (0.830, 0.900, 0.955)},
        "F": {"GM_pct": (39.0, 47.0, 54.0), "WM_pct": (33.0, 37.0, 41.0),
              "CSF_pct": (3.0, 8.3, 14.0), "BPF": (0.835, 0.907, 0.958)},
    },
}

# 年萎缩率参考（正常衰老 vs 病理）
# GM 年萎缩率：正常 < 0.5%/年；MCI 0.5-1.5%；AD > 1.5%
ATROPHY_RATE_THRESHOLDS = {"normal": 0.5, "mci": 1.5}  # %/year


def get_age_bucket(age: int | None) -> str:
    if age is None:
        return "60-75"  # 默认中老年组
    if age < 40:
        return "20-40"
    if age < 60:
        return "40-60"
    if age < 75:
        return "60-75"
    return "75+"


def percentile_rank(value: float, p10: float, p50: float, p90: float) -> float:
    """根据三百分位近似估算 value 在该分布中的百分位 (0-100)。"""
    if value <= p10:
        return 10.0 * (value / p10) if p10 > 0 else 0.0
    if value <= p50:
        return 10.0 + 40.0 * (value - p10) / (p50 - p10)
    if value <= p90:
        return 50.0 + 40.0 * (value - p50) / (p90 - p50)
    return 90.0 + 10.0  # 高于 P90


def compute_reference_comparison(
    gm_pct: float, wm_pct: float, csf_pct: float, bpf: float,
    age: int | None, sex: str | None,
) -> dict:
    """计算各指标相对同龄同性别人群的百分位与判定。"""
    bucket = get_age_bucket(age)
    sex_key = sex if sex in ("M", "F") else "M"
    ref = REFERENCE_RANGES[bucket][sex_key]

    # 对于 GM/WM/BPF：低于 P10 为异常偏低（萎缩）
    # 对于 CSF：高于 P90 为异常偏高（萎缩）
    gm_pr = percentile_rank(gm_pct, *ref["GM_pct"])
    wm_pr = percentile_rank(wm_pct, *ref["WM_pct"])
    csf_pr = percentile_rank(csf_pct, *ref["CSF_pct"])
    bpf_pr = percentile_rank(bpf, *ref["BPF"])

    def classify_low(pr: float) -> tuple[str, str]:
        if pr < 10:
            return "显著低于同龄", "#ff4d4d"
        if pr < 25:
            return "偏低", "#ff8c33"
        if pr > 75:
            return "正常偏高", "#33e673"
        return "正常范围", "#33e673"

    def classify_high(pr: float) -> tuple[str, str]:
        if pr > 90:
            return "显著高于同龄", "#ff4d4d"
        if pr > 75:
            return "偏高", "#ff8c33"
        return "正常范围", "#33e673"

    gm_status, gm_color = classify_low(gm_pr)
    wm_status, wm_color = classify_low(wm_pr)
    bpf_status, bpf_color = classify_low(bpf_pr)
    csf_status, csf_color = classify_high(csf_pr)

    return {
        "age_bucket": bucket,
        "sex": sex_key,
        "reference": {
            "GM_pct": {"p10": ref["GM_pct"][0], "p50": ref["GM_pct"][1], "p90": ref["GM_pct"][2]},
            "WM_pct": {"p10": ref["WM_pct"][0], "p50": ref["WM_pct"][1], "p90": ref["WM_pct"][2]},
            "CSF_pct": {"p10": ref["CSF_pct"][0], "p50": ref["CSF_pct"][1], "p90": ref["CSF_pct"][2]},
            "BPF": {"p10": ref["BPF"][0], "p50": ref["BPF"][1], "p90": ref["BPF"][2]},
        },
        "percentiles": {
            "GM": round(gm_pr, 1), "WM": round(wm_pr, 1),
            "CSF": round(csf_pr, 1), "BPF": round(bpf_pr, 1),
        },
        "status": {
            "GM": gm_status, "WM": wm_status, "CSF": csf_status, "BPF": bpf_status,
        },
        "status_colors": {
            "GM": gm_color, "WM": wm_color, "CSF": csf_color, "BPF": bpf_color,
        },
    }


def compute_clinical_metrics(
    pred: np.ndarray,
    voxel_volume_mm3: float,
    age: int | None = None,
    sex: str | None = None,
) -> dict:
    """从分割结果计算脑萎缩相关定量指标。"""
    counts = {v: int(np.count_nonzero(pred == v)) for v in (0, 1, 2, 3)}
    vols = {TISSUE_NAMES[v]: counts[v] * voxel_volume_mm3 / 1000.0 for v in (1, 2, 3)}

    tiv = vols["CSF"] + vols["GM"] + vols["WM"]
    bpf = (vols["GM"] + vols["WM"]) / tiv if tiv > 0 else 0.0

    # 脑室 CSF 估算：连通域分析，取中央区域的大连通域
    csf_mask = (pred == 1).astype(np.uint8)
    labeled, n_components = ndimage.label(csf_mask)
    ventricular_vol = 0.0
    if n_components > 0:
        brain_mask = (pred > 0).astype(np.uint8)
        coords = np.array(np.nonzero(brain_mask))
        lo = coords.min(axis=1)
        hi = coords.max(axis=1)
        center = (lo + hi) / 2.0
        extent = hi - lo

        for comp_id in range(1, n_components + 1):
            comp = labeled == comp_id
            comp_size = int(comp.sum())
            if comp_size < 200:
                continue
            comp_coords = np.array(np.nonzero(comp))
            centroid = comp_coords.mean(axis=0)
            in_central = all(
                abs(centroid[a] - center[a]) < 0.30 * extent[a] for a in (0, 1)
            ) and (centroid[2] < center[2] + 0.2 * extent[2])
            if in_central:
                ventricular_vol += comp_size * voxel_volume_mm3 / 1000.0

    vbr = ventricular_vol / (vols["GM"] + vols["WM"]) if (vols["GM"] + vols["WM"]) > 0 else 0.0

    csf_ratio = vols["CSF"] / tiv if tiv > 0 else 0.0
    gm_pct = vols["GM"] / tiv * 100 if tiv > 0 else 0.0
    wm_pct = vols["WM"] / tiv * 100 if tiv > 0 else 0.0
    csf_pct = csf_ratio * 100

    # 参考范围对比（年龄/性别）
    ref_cmp = compute_reference_comparison(gm_pct, wm_pct, csf_pct, bpf, age, sex)

    # 综合萎缩评分：综合 GM 百分位、BPF 百分位、CSF 百分位（反向）
    # 分数越低越萎缩
    composite_score = (
        ref_cmp["percentiles"]["GM"] * 0.35
        + ref_cmp["percentiles"]["BPF"] * 0.35
        + (100 - ref_cmp["percentiles"]["CSF"]) * 0.30
    )

    # 分级：优先用百分位，再用 CSF 比值/VBR 兜底
    if composite_score >= 50 and csf_pct < ref_cmp["reference"]["CSF_pct"]["p90"]:
        atrophy_grade = "正常范围"
        atrophy_color = "#33e673"
    elif composite_score >= 30:
        atrophy_grade = "轻度萎缩"
        atrophy_color = "#ffc233"
    elif composite_score >= 15:
        atrophy_grade = "中度萎缩"
        atrophy_color = "#ff8c33"
    else:
        atrophy_grade = "重度萎缩"
        atrophy_color = "#ff4d4d"

    # 临床解读建议
    interpretations = []
    if atrophy_grade != "正常范围":
        interpretations.append(f"提示{atrophy_grade}，建议结合 MMSE/MoCA 认知评估及临床病史。")
    if ref_cmp["status"]["GM"] != "正常范围":
        interpretations.append(f"灰质{ref_cmp['status']['GM']}（同龄百分位 {ref_cmp['percentiles']['GM']}）。")
    if ref_cmp["status"]["CSF"] != "正常范围":
        interpretations.append(f"脑脊液{ref_cmp['status']['CSF']}，提示脑室/脑沟扩大。")
    if vbr > 0.05:
        interpretations.append(f"VBR={vbr:.3f}，脑室扩大明显。")
    if not interpretations:
        interpretations.append("各指标在同龄同性别参考范围内，未见明显萎缩征象。")

    return {
        "volumes_ml": {k: round(v, 1) for k, v in vols.items()},
        "tiv_ml": round(tiv, 1),
        "ratios": {
            "GM/TIV": round(vols["GM"] / tiv, 4) if tiv > 0 else 0,
            "WM/TIV": round(vols["WM"] / tiv, 4) if tiv > 0 else 0,
            "CSF/TIV": round(csf_ratio, 4),
            "BPF": round(bpf, 4),
        },
        "ventricular_csf_ml": round(ventricular_vol, 1),
        "vbr": round(vbr, 4),
        "atrophy_grade": atrophy_grade,
        "atrophy_color": atrophy_color,
        "composite_score": round(composite_score, 1),
        "reference_comparison": ref_cmp,
        "interpretations": interpretations,
        "age": age,
        "sex": sex,
    }


# ── 可视化 ────────────────────────────────────────────────────────────


def render_overlay(image: np.ndarray, label: np.ndarray) -> bytes:
    """三视图叠加 PNG。"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=110)
    views = [
        ("Axial (Z)", lambda v, i: v[:, :, i]),
        ("Sagittal (X)", lambda v, i: v[i, :, :]),
        ("Coronal (Y)", lambda v, i: v[:, i, :]),
    ]
    middles = [image.shape[2] // 2, image.shape[0] // 2, image.shape[1] // 2]

    for col, ((title, slicer), idx) in enumerate(zip(views, middles)):
        img_2d = np.rot90(slicer(image, idx))
        seg_2d = np.rot90(slicer(label, idx))

        axes[0, col].imshow(img_2d, cmap="gray")
        axes[0, col].set_title(f"{title} — MRI")
        axes[0, col].axis("off")

        axes[1, col].imshow(img_2d, cmap="gray")
        overlay = np.zeros((*seg_2d.shape, 4), dtype=float)
        for value, color in LABEL_COLORS.items():
            overlay[seg_2d == value] = color
        axes[1, col].imshow(overlay)
        axes[1, col].set_title(f"{title} — Segmentation")
        axes[1, col].axis("off")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in LABEL_COLORS.values()]
    fig.legend(handles, ["CSF", "Gray Matter", "White Matter"],
               loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def extract_3d_mesh(label: np.ndarray, max_dim: int = 64) -> dict:
    """降采样后 marching_cubes 提取各组织表面网格。"""
    shape = label.shape
    factors = [max(1, int(np.ceil(s / max_dim))) for s in shape]
    small = label[:: factors[0], :: factors[1], :: factors[2]]

    meshes = {}
    for val, name in TISSUE_NAMES.items():
        mask = (small == val).astype(np.float32)
        if mask.sum() < 10:
            continue
        mask = np.pad(mask, 1, mode="constant")
        try:
            verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
        except RuntimeError:
            continue
        # 还原到原始体素坐标（缩放回去）
        scale = [f * (s / max(s - 1, 1)) for f, s in zip(factors, shape)]
        verts = verts * np.array(scale)
        meshes[name] = {
            "vertices": verts.tolist(),
            "faces": faces.tolist(),
            "color": TISSUE_COLORS_HEX[val],
        }
    return meshes


# ── 推理主流程 ────────────────────────────────────────────────────────


def run_inference(image_path: Path, age: int | None = None, sex: str | None = None) -> dict:
    nii = nib.load(image_path)
    native_image = np.asarray(nii.dataobj, dtype=np.float32)
    if native_image.ndim == 4 and native_image.shape[-1] == 1:
        native_image = native_image[..., 0]
    voxel_vol = float(np.prod(nii.header.get_zooms()[:3]))

    batch = PREPROCESS({"image": str(image_path)})
    image = batch["image"].as_tensor().unsqueeze(0).to(DEVICE)

    model = get_model()
    with _INFER_LOCK, torch.no_grad():
        logits = sliding_window_predict(
            model=model, images=image,
            roi_size=(96, 96, 96), sw_batch_size=1, overlap=0.5,
        )

    cropped_pred = logits.argmax(dim=1, keepdim=True)
    native_pred = invert_prediction_to_native_space(
        prediction=cropped_pred.cpu(), batch=batch, reference_path=image_path,
    )[0].cpu().numpy().astype(np.uint8)

    metrics = compute_clinical_metrics(native_pred, voxel_vol, age=age, sex=sex)
    overlay_png = render_overlay(native_image, native_pred)
    mesh = extract_3d_mesh(native_pred)

    return {
        "shape": list(native_image.shape),
        "metrics": metrics,
        "overlay_png_base64": base64.b64encode(overlay_png).decode("ascii"),
        "mesh": mesh,
    }


def load_nifti_array(image_path: Path) -> tuple[np.ndarray, float]:
    nii = nib.load(image_path)
    arr = np.asarray(nii.dataobj, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    voxel_vol = float(np.prod(nii.header.get_zooms()[:3]))
    return arr, voxel_vol


def register_images(fixed_path: Path, moving_path: Path) -> np.ndarray:
    """用 SimpleITK 刚性配准，把 moving 配到 fixed 空间，返回配准后的 moving 数组。"""
    fixed = sitk.ReadImage(str(fixed_path), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)

    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.01)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=100,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(initial, inPlace=False)
    reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    try:
        final_tf = reg.Execute(fixed, moving)
    except Exception:
        # 配准失败时退回到初始变换
        final_tf = initial

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetTransform(final_tf)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    registered = resampler.Execute(moving)
    return sitk.GetArrayFromImage(registered).astype(np.float32)


# ── API 路由 ──────────────────────────────────────────────────────────


def _parse_age_sex(age: str | None, sex: str | None) -> tuple[int | None, str | None]:
    age_val: int | None = None
    if age:
        try:
            age_val = int(age)
        except ValueError:
            age_val = None
    sex_val = sex.upper() if sex and sex.upper() in ("M", "F") else None
    return age_val, sex_val


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    age: str | None = Form(None),
    sex: str | None = Form(None),
) -> dict:
    fname = file.filename or ""
    is_nifti = fname.endswith((".nii", ".nii.gz"))
    is_dicom_zip = fname.endswith(".zip")
    if not (is_nifti or is_dicom_zip):
        raise HTTPException(400, "仅支持 .nii / .nii.gz / .zip(DICOM序列)")

    content = await file.read()
    if len(content) > 500 * 1024 * 1024:
        raise HTTPException(400, "文件过大（>500MB）")

    age_val, sex_val = _parse_age_sex(age, sex)
    work_dir = Path(tempfile.mkdtemp(prefix="brain_seg_"))
    try:
        if is_dicom_zip:
            zip_path = work_dir / "upload.zip"
            zip_path.write_bytes(content)
            nifti_path = dicom_zip_to_nifti(zip_path, work_dir)
        else:
            suffix = ".nii.gz" if fname.endswith(".nii.gz") else ".nii"
            nifti_path = work_dir / f"upload{suffix}"
            nifti_path.write_bytes(content)

        nib.load(nifti_path)  # 校验
        return run_inference(nifti_path, age=age_val, sex=sex_val)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"处理失败: {exc}") from exc
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/api/predict_longitudinal")
async def predict_longitudinal(
    baseline: UploadFile = File(...),
    followup: UploadFile = File(...),
    interval_days: str | None = Form(None),
    age: str | None = Form(None),
    sex: str | None = Form(None),
) -> dict:
    """纵向对比：同一患者两次 MRI，计算 GM/WM 年萎缩率。"""
    for f in (baseline, followup):
        fn = f.filename or ""
        if not fn.endswith((".nii", ".nii.gz", ".zip")):
            raise HTTPException(400, "两次扫描均需为 .nii/.nii.gz/.zip")

    age_val, sex_val = _parse_age_sex(age, sex)
    interval = float(interval_days) if interval_days else None

    base_content = await baseline.read()
    fu_content = await followup.read()
    for c in (base_content, fu_content):
        if len(c) > 500 * 1024 * 1024:
            raise HTTPException(400, "文件过大（>500MB）")

    work_dir = Path(tempfile.mkdtemp(prefix="brain_long_"))
    try:
        def _save(up: UploadFile, content: bytes, name: str) -> Path:
            fn = up.filename or ""
            if fn.endswith(".zip"):
                zp = work_dir / f"{name}.zip"
                zp.write_bytes(content)
                return dicom_zip_to_nifti(zp, work_dir / name)
            suffix = ".nii.gz" if fn.endswith(".nii.gz") else ".nii"
            p = work_dir / f"{name}{suffix}"
            p.write_bytes(content)
            return p

        base_path = _save(baseline, base_content, "baseline")
        fu_path = _save(followup, fu_content, "followup")
        nib.load(base_path); nib.load(fu_path)

        base_result = run_inference(base_path, age=age_val, sex=sex_val)
        # 随访像配准到基线空间后再推理（体积在同一空间下对比）
        registered_arr = register_images(base_path, fu_path)
        # 把配准后的数组写回 NIfTI（复用基线的头信息）
        base_nii = nib.load(base_path)
        reg_nii = nib.Nifti1Image(registered_arr, base_nii.affine, base_nii.header)
        reg_path = work_dir / "followup_registered.nii.gz"
        nib.save(reg_nii, reg_path)
        fu_result = run_inference(reg_path, age=age_val, sex=sex_val)

        base_vols = base_result["metrics"]["volumes_ml"]
        fu_vols = fu_result["metrics"]["volumes_ml"]

        def _pct_change(old: float, new: float) -> float:
            return ((new - old) / old * 100) if old > 0 else 0.0

        gm_change = _pct_change(base_vols["GM"], fu_vols["GM"])
        wm_change = _pct_change(base_vols["WM"], fu_vols["WM"])
        csf_change = _pct_change(base_vols["CSF"], fu_vols["CSF"])

        gm_annual = gm_change / (interval / 365.25) if interval else None
        wm_annual = wm_change / (interval / 365.25) if interval else None

        # 年萎缩率判定
        if gm_annual is not None:
            if gm_annual > -ATROPHY_RATE_THRESHOLDS["normal"]:
                rate_grade = "正常衰老范围"
                rate_color = "#33e673"
            elif gm_annual > -ATROPHY_RATE_THRESHOLDS["mci"]:
                rate_grade = "轻度认知障碍可疑"
                rate_color = "#ffc233"
            else:
                rate_grade = "阿尔茨海默病可疑"
                rate_color = "#ff4d4d"
        else:
            rate_grade = "需提供两次扫描间隔天数"
            rate_color = "#64748b"

        return {
            "baseline": base_result,
            "followup": fu_result,
            "longitudinal": {
                "gm_change_pct": round(gm_change, 2),
                "wm_change_pct": round(wm_change, 2),
                "csf_change_pct": round(csf_change, 2),
                "gm_annual_pct": round(gm_annual, 2) if gm_annual is not None else None,
                "wm_annual_pct": round(wm_annual, 2) if wm_annual is not None else None,
                "interval_days": interval,
                "rate_grade": rate_grade,
                "rate_color": rate_color,
            },
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"纵向对比失败: {exc}") from exc
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ── PDF 报告 ──────────────────────────────────────────────────────────


def _find_chinese_font() -> str | None:
    """查找系统中可用的中文字体，返回字体名（已注册）或 None。"""
    candidates = [
        ("C:/Windows/Fonts/msyh.ttc", "MSYH"),
        ("C:/Windows/Fonts/simhei.ttf", "SimHei"),
        ("C:/Windows/Fonts/simsun.ttc", "SimSun"),
    ]
    for path, name in candidates:
        if Path(path).exists():
            try:
                pdfmetrics.registerFont(TTFont(name, path))
                return name
            except Exception:
                continue
    return None


@app.post("/api/report.pdf")
async def generate_pdf_report(result: dict) -> StreamingResponse:
    """根据推理结果生成可下载的临床 PDF 报告。"""
    metrics = result.get("metrics", {})
    shape = result.get("shape", [])
    overlay_b64 = result.get("overlay_png_base64", "")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm,
        title="脑萎缩定量分析报告",
    )
    font_name = _find_chinese_font() or "Helvetica"
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=styles["Title"], fontName=font_name, fontSize=16, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName=font_name, fontSize=12, spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName=font_name, fontSize=10, leading=14)
    small = ParagraphStyle("small", parent=styles["BodyText"], fontName=font_name, fontSize=8, leading=11, textColor=colors.grey)

    story = []
    story.append(Paragraph("脑萎缩定量分析报告", title_style))
    from datetime import datetime
    story.append(Paragraph(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", small))
    age_sex = ""
    if metrics.get("age"):
        age_sex += f"年龄：{metrics['age']}岁　"
    if metrics.get("sex"):
        age_sex += f"性别：{'男' if metrics['sex']=='M' else '女'}　"
    if age_sex:
        story.append(Paragraph(age_sex, small))
    story.append(Spacer(1, 6))

    # 萎缩分级徽章
    grade = metrics.get("atrophy_grade", "-")
    color = metrics.get("atrophy_color", "#333333")
    grade_table = Table(
        [[Paragraph(f"<b>综合评估：{grade}</b>", ParagraphStyle("g", parent=body, fontSize=13, textColor=colors.white))]],
        colWidths=[170 * mm],
    )
    grade_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("ROUNDEDCORNERS", [6]),
    ]))
    story.append(grade_table)
    story.append(Spacer(1, 8))

    # 定量指标表
    story.append(Paragraph("一、定量指标", h2))
    vols = metrics.get("volumes_ml", {})
    ratios = metrics.get("ratios", {})
    rows = [
        ["指标", "数值", "参考范围", "状态"],
        ["CSF 体积 (mL)", f"{vols.get('CSF', '-')}", "—", metrics.get("reference_comparison", {}).get("status", {}).get("CSF", "-")],
        ["灰质 GM 体积 (mL)", f"{vols.get('GM', '-')}", "—", metrics.get("reference_comparison", {}).get("status", {}).get("GM", "-")],
        ["白质 WM 体积 (mL)", f"{vols.get('WM', '-')}", "—", metrics.get("reference_comparison", {}).get("status", {}).get("WM", "-")],
        ["TIV (mL)", f"{metrics.get('tiv_ml', '-')}", "—", "—"],
        ["GM/TIV (%)", f"{ratios.get('GM/TIV', 0)*100:.1f}", f"P50={metrics.get('reference_comparison',{}).get('reference',{}).get('GM_pct',{}).get('p50','-')}", metrics.get("reference_comparison", {}).get("status", {}).get("GM", "-")],
        ["WM/TIV (%)", f"{ratios.get('WM/TIV', 0)*100:.1f}", f"P50={metrics.get('reference_comparison',{}).get('reference',{}).get('WM_pct',{}).get('p50','-')}", metrics.get("reference_comparison", {}).get("status", {}).get("WM", "-")],
        ["CSF/TIV (%)", f"{ratios.get('CSF/TIV', 0)*100:.1f}", f"P50={metrics.get('reference_comparison',{}).get('reference',{}).get('CSF_pct',{}).get('p50','-')}", metrics.get("reference_comparison", {}).get("status", {}).get("CSF", "-")],
        ["BPF", f"{ratios.get('BPF', 0)*100:.1f}%", f"P50={metrics.get('reference_comparison',{}).get('reference',{}).get('BPF',{}).get('p50','-')}", metrics.get("reference_comparison", {}).get("status", {}).get("BPF", "-")],
        ["脑室 CSF (mL)", f"{metrics.get('ventricular_csf_ml', '-')}", "< 20", "—"],
        ["VBR", f"{metrics.get('vbr', '-')}", "< 0.03", "—"],
    ]
    tbl = Table(rows, colWidths=[45*mm, 35*mm, 45*mm, 45*mm])
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 8))

    # 临床解读
    story.append(Paragraph("二、临床解读", h2))
    for interp in metrics.get("interpretations", []):
        story.append(Paragraph(f"• {interp}", body))
    story.append(Spacer(1, 8))

    # 三视图
    if overlay_b64:
        story.append(Paragraph("三、分割结果（三视图）", h2))
        try:
            img_bytes = base64.b64decode(overlay_b64)
            img_buf = io.BytesIO(img_bytes)
            story.append(RLImage(img_buf, width=170 * mm, height=110 * mm))
        except Exception:
            pass
        story.append(Spacer(1, 8))

    # 免责声明
    story.append(Paragraph("免责声明", h2))
    story.append(Paragraph(
        "本报告由 AI 模型辅助生成，仅供临床参考，不构成最终诊断。"
        "模型基于 IBSR-18 数据集训练，参考范围综合 ICBM/ADNI/OASIS 文献近似值，"
        "实际临床应用需结合病史、认知评估及多模态检查综合判断。",
        small,
    ))

    doc.build(story)
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=brain_atrophy_report.pdf"},
    )


@app.get("/api/health")
def health() -> dict:
    return {
        "device": str(DEVICE),
        "model_loaded": _MODEL is not None,
        "checkpoint": CHECKPOINT_PATH.name,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent / "static"),
    name="static",
)

if __name__ == "__main__":
    # 0.0.0.0 表示监听所有网卡，同一局域网（同一 WiFi/办公网络）的设备可通过本机内网 IP 访问
    uvicorn.run(app, host="0.0.0.0", port=8000)
