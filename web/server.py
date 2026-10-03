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
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure

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


def compute_clinical_metrics(
    pred: np.ndarray,
    voxel_volume_mm3: float,
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
            # 中央 50% 区域（X、Y），中下 70%（Z）
            in_central = all(
                abs(centroid[a] - center[a]) < 0.30 * extent[a] for a in (0, 1)
            ) and (centroid[2] < center[2] + 0.2 * extent[2])
            if in_central:
                ventricular_vol += comp_size * voxel_volume_mm3 / 1000.0

    vbr = ventricular_vol / (vols["GM"] + vols["WM"]) if (vols["GM"] + vols["WM"]) > 0 else 0.0

    # 萎缩分级（针对 skull-stripped 数据调整）
    # 去颅骨后 CSF 仅含脑室+脑沟，BPF 天然偏高（~0.95-0.99）
    # 因此主要依据 CSF/TIV 比值和 VBR 判断萎缩
    csf_ratio = vols["CSF"] / tiv if tiv > 0 else 0.0
    if csf_ratio <= 0.025 and vbr <= 0.03:
        atrophy_grade = "正常范围"
        atrophy_color = "#33e673"
    elif csf_ratio <= 0.04 and vbr <= 0.05:
        atrophy_grade = "轻度萎缩"
        atrophy_color = "#ffc233"
    elif csf_ratio <= 0.06 and vbr <= 0.08:
        atrophy_grade = "中度萎缩"
        atrophy_color = "#ff8c33"
    else:
        atrophy_grade = "重度萎缩"
        atrophy_color = "#ff4d4d"

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


def run_inference(image_path: Path) -> dict:
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

    metrics = compute_clinical_metrics(native_pred, voxel_vol)
    overlay_png = render_overlay(native_image, native_pred)
    mesh = extract_3d_mesh(native_pred)

    return {
        "shape": list(native_image.shape),
        "metrics": metrics,
        "overlay_png_base64": base64.b64encode(overlay_png).decode("ascii"),
        "mesh": mesh,
    }


# ── API 路由 ──────────────────────────────────────────────────────────


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    fname = file.filename or ""
    is_nifti = fname.endswith((".nii", ".nii.gz"))
    is_dicom_zip = fname.endswith(".zip")
    if not (is_nifti or is_dicom_zip):
        raise HTTPException(400, "仅支持 .nii / .nii.gz / .zip(DICOM序列)")

    content = await file.read()
    if len(content) > 500 * 1024 * 1024:
        raise HTTPException(400, "文件过大（>500MB）")

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
        return run_inference(nifti_path)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"处理失败: {exc}") from exc
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


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
