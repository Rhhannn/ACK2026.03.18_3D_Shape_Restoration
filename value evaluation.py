import json
import random
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd
import torch
import trimesh


# ============================================================
# 0. 기본 설정
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 1. GT 데이터 로드
# ============================================================
def load_surface_points(pointcloud_npz_path: str, device=DEVICE):
    """
    [GT 표면 점군 로드]
    사용 파일: pointcloud.npz

    이 데이터는 아래 평가지표에서 사용됩니다.
    - 평가지표 2. CD (Chamfer Distance)
    - 평가지표 3. F-score
    """
    data = np.load(pointcloud_npz_path)

    points = data["points"].astype(np.float32)
    meta = {
        "loc": data["loc"].tolist() if "loc" in data else None,
        "scale": float(data["scale"]) if "scale" in data else None,
        "num_points": int(points.shape[0]),
    }

    return torch.from_numpy(points).to(device), meta


def load_query_points_and_occ(points_npz_path: str, device=DEVICE):
    """
    [GT occupancy query 로드]
    사용 파일: points.npz

    이 데이터는 아래 평가지표에서 사용됩니다.
    - 평가지표 1. Volume IoU

    구성:
    - points       : query points
    - occupancies  : 각 query point가 shape 내부인지 여부
                     bit-packed uint8 이므로 unpack 필요
    """
    data = np.load(points_npz_path)

    query_points = data["points"].astype(np.float32)
    occ_raw = data["occupancies"]

    # bit-packed occupancy 복원
    gt_occ = np.unpackbits(occ_raw)[: len(query_points)].astype(np.bool_)

    meta = {
        "loc": data["loc"].tolist() if "loc" in data else None,
        "scale": float(data["scale"]) if "scale" in data else None,
        "num_query_points": int(query_points.shape[0]),
    }

    return (
        torch.from_numpy(query_points).to(device),
        torch.from_numpy(gt_occ).to(device),
        meta,
    )


# ============================================================
# 2. 예측 mesh 로드
# ============================================================
def load_mesh(mesh_path: str) -> trimesh.Trimesh:
    """
    사용 파일:
    - Baseline.ply
    - Ours.ply

    이 예측 mesh는 아래 3개 평가지표 모두에 사용됩니다.
    - 평가지표 1. Volume IoU
    - 평가지표 2. CD
    - 평가지표 3. F-score
    """
    mesh = trimesh.load(mesh_path, force="mesh")

    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if len(meshes) == 0:
            raise ValueError(f"{mesh_path}: scene 안에 mesh가 없습니다.")
        mesh = trimesh.util.concatenate(meshes)

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{mesh_path}: mesh 로드 실패")

    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()

    try:
        mesh.remove_degenerate_faces()
    except Exception:
        pass

    try:
        mesh.remove_duplicate_faces()
    except Exception:
        pass

    return mesh


def get_mesh_info(mesh: trimesh.Trimesh):
    return {
        "num_vertices": int(len(mesh.vertices)),
        "num_faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "bbox_min": mesh.bounds[0].astype(float).tolist(),
        "bbox_max": mesh.bounds[1].astype(float).tolist(),
    }


def print_mesh_info(name: str, mesh_info: dict):
    print(f"\n[{name}]")
    print(f"  vertices           : {mesh_info['num_vertices']}")
    print(f"  faces              : {mesh_info['num_faces']}")
    print(f"  watertight         : {mesh_info['watertight']}")
    print(f"  winding_consistent : {mesh_info['winding_consistent']}")
    print(f"  bbox_min           : {mesh_info['bbox_min']}")
    print(f"  bbox_max           : {mesh_info['bbox_max']}")


# ============================================================
# 3. 예측 mesh 표면 샘플링
# ============================================================
def sample_mesh_surface(
    mesh: trimesh.Trimesh,
    n_points: int = 50000,
    seed: int = 42,
    device=DEVICE,
):
    """
    예측 mesh 표면에서 점을 샘플링

    이 샘플 점은 아래 평가지표에서 사용됩니다.
    - 평가지표 2. CD
    - 평가지표 3. F-score
    """
    np.random.seed(seed)
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    points = points.astype(np.float32)
    return torch.from_numpy(points).to(device)


# ============================================================
# 4. 거리 계산 유틸
# ============================================================
def chunked_min_distances(
    src: torch.Tensor,
    dst: torch.Tensor,
    chunk_size_src: int = 2048,
    chunk_size_dst: int = 2048,
) -> torch.Tensor:
    """
    src 각 점에서 dst까지의 최근접 거리 계산
    torch.cdist를 chunk로 나눠 메모리 사용량 제어
    """
    src = src.float()
    dst = dst.float()

    out = []

    for i in range(0, src.shape[0], chunk_size_src):
        src_chunk = src[i : i + chunk_size_src]
        min_dist = torch.full(
            (src_chunk.shape[0],),
            float("inf"),
            device=src.device,
            dtype=torch.float32,
        )

        for j in range(0, dst.shape[0], chunk_size_dst):
            dst_chunk = dst[j : j + chunk_size_dst]
            d = torch.cdist(src_chunk, dst_chunk, p=2)
            min_dist = torch.minimum(min_dist, d.min(dim=1).values)

        out.append(min_dist)

    return torch.cat(out, dim=0)


# ============================================================
# [평가지표 2] CD (Chamfer Distance)
# ============================================================
def chamfer_distance_torch(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    squared: bool = False,
    chunk_size_src: int = 2048,
    chunk_size_dst: int = 2048,
):
    """
    [평가지표 2. CD (Chamfer Distance)]

    사용 데이터:
    - pred_points : 예측 mesh 표면에서 샘플링한 점
    - gt_points   : pointcloud.npz 에서 불러온 GT 표면 점군

    계산 방식:
    1. pred -> gt 최근접 거리 평균
    2. gt -> pred 최근접 거리 평균
    3. 둘을 더해서 CD 계산

    squared=False : Chamfer-L1 스타일
    squared=True  : squared L2 스타일
    """
    d_pred_to_gt = chunked_min_distances(
        pred_points, gt_points,
        chunk_size_src=chunk_size_src,
        chunk_size_dst=chunk_size_dst,
    )
    d_gt_to_pred = chunked_min_distances(
        gt_points, pred_points,
        chunk_size_src=chunk_size_src,
        chunk_size_dst=chunk_size_dst,
    )

    if squared:
        d_pred_to_gt = d_pred_to_gt ** 2
        d_gt_to_pred = d_gt_to_pred ** 2

    cd = d_pred_to_gt.mean() + d_gt_to_pred.mean()
    return float(cd.item())


# ============================================================
# [평가지표 3] F-score
# ============================================================
def fscore_torch(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    threshold: float = 0.01,
    chunk_size_src: int = 2048,
    chunk_size_dst: int = 2048,
):
    """
    [평가지표 3. F-score]

    사용 데이터:
    - pred_points : 예측 mesh 표면에서 샘플링한 점
    - gt_points   : pointcloud.npz 에서 불러온 GT 표면 점군

    계산 방식:
    1. pred -> gt 최근접 거리 계산
    2. gt -> pred 최근접 거리 계산
    3. threshold 이하를 match로 판단
    4. precision, recall 계산
    5. F-score = 2PR / (P + R)

    threshold 예:
    - 0.01 이면 F-score@1%
    """
    d_pred_to_gt = chunked_min_distances(
        pred_points, gt_points,
        chunk_size_src=chunk_size_src,
        chunk_size_dst=chunk_size_dst,
    )
    d_gt_to_pred = chunked_min_distances(
        gt_points, pred_points,
        chunk_size_src=chunk_size_src,
        chunk_size_dst=chunk_size_dst,
    )

    precision = (d_pred_to_gt < threshold).float().mean()
    recall = (d_gt_to_pred < threshold).float().mean()

    if (precision + recall).item() == 0:
        f = torch.tensor(0.0, device=pred_points.device)
    else:
        f = 2.0 * precision * recall / (precision + recall)

    return float(f.item()), float(precision.item()), float(recall.item())


# ============================================================
# 7. Volume IoU용 occupancy 계산
# ============================================================
@nb.njit(parallel=True, fastmath=True)
def contains_points_numba(points, triangles, direction):
    """
    trimesh.contains가 안 될 때 사용하는 fallback
    """
    n = points.shape[0]
    out = np.zeros(n, dtype=np.bool_)

    dx, dy, dz = direction[0], direction[1], direction[2]
    eps = 1e-8

    for i in nb.prange(n):
        px, py, pz = points[i, 0], points[i, 1], points[i, 2]
        count = 0

        for j in range(triangles.shape[0]):
            v0x, v0y, v0z = triangles[j, 0, 0], triangles[j, 0, 1], triangles[j, 0, 2]
            v1x, v1y, v1z = triangles[j, 1, 0], triangles[j, 1, 1], triangles[j, 1, 2]
            v2x, v2y, v2z = triangles[j, 2, 0], triangles[j, 2, 1], triangles[j, 2, 2]

            e1x, e1y, e1z = v1x - v0x, v1y - v0y, v1z - v0z
            e2x, e2y, e2z = v2x - v0x, v2y - v0y, v2z - v0z

            hx = dy * e2z - dz * e2y
            hy = dz * e2x - dx * e2z
            hz = dx * e2y - dy * e2x

            a = e1x * hx + e1y * hy + e1z * hz

            if -eps < a < eps:
                continue

            f = 1.0 / a

            sx, sy, sz = px - v0x, py - v0y, pz - v0z
            u = f * (sx * hx + sy * hy + sz * hz)

            if u < -eps or u > 1.0 + eps:
                continue

            qx = sy * e1z - sz * e1y
            qy = sz * e1x - sx * e1z
            qz = sx * e1y - sy * e1x

            v = f * (dx * qx + dy * qy + dz * qz)

            if v < -eps or (u + v) > 1.0 + eps:
                continue

            t = f * (e2x * qx + e2y * qy + e2z * qz)

            if t > eps:
                count += 1

        out[i] = (count % 2 == 1)

    return out


def predict_mesh_occupancy(mesh: trimesh.Trimesh, query_points: torch.Tensor):
    """
    예측 mesh가 query point를 내부에 포함하는지 판정

    이 결과는 아래 평가지표에서 사용됩니다.
    - 평가지표 1. Volume IoU
    """
    query_points_np = np.ascontiguousarray(
        query_points.detach().cpu().numpy().astype(np.float32)
    )

    try:
        pred_occ = mesh.contains(query_points_np).astype(np.bool_)
        method = "trimesh.contains"
        return pred_occ, method
    except Exception:
        pass

    triangles = np.ascontiguousarray(mesh.triangles.astype(np.float32))
    direction = np.array([1.0, 0.12347, 0.05679], dtype=np.float32)
    direction = direction / np.linalg.norm(direction)

    pred_occ = contains_points_numba(query_points_np, triangles, direction).astype(np.bool_)
    method = "numba_ray_casting"
    return pred_occ, method


# ============================================================
# [평가지표 1] Volume IoU
# ============================================================
def volume_iou_from_mesh(mesh: trimesh.Trimesh, query_points: torch.Tensor, gt_occ: torch.Tensor):
    """
    [평가지표 1. Volume IoU]

    사용 데이터:
    - mesh         : Baseline.ply 또는 Ours.ply
    - query_points : points.npz 의 points
    - gt_occ       : points.npz 의 occupancies

    계산 방식:
    1. 예측 mesh에 대해 각 query point가 내부인지 occupancy 예측
    2. GT occupancy와 예측 occupancy 비교
    3. intersection / union 으로 IoU 계산
    """
    pred_occ_np, occ_method = predict_mesh_occupancy(mesh, query_points)
    pred_occ = torch.from_numpy(pred_occ_np).to(gt_occ.device)

    intersection = torch.logical_and(pred_occ, gt_occ).sum().float()
    union = torch.logical_or(pred_occ, gt_occ).sum().float()

    if union.item() == 0:
        return 0.0, occ_method

    return float((intersection / union).item()), occ_method


# ============================================================
# 9. 단일 모델 평가
# ============================================================
def evaluate_one_model(
    mesh_path: str,
    pointcloud_npz_path: str,
    points_npz_path: str,
    n_surface_samples: int = 50000,
    fscore_threshold: float = 0.01,
    squared_cd: bool = False,
    chunk_size_src: int = 2048,
    chunk_size_dst: int = 2048,
    seed: int = 42,
    device=DEVICE,
):
    # --------------------------------------------------------
    # GT surface points 로드
    # -> 평가지표 2. CD
    # -> 평가지표 3. F-score
    # --------------------------------------------------------
    gt_surface_points, surface_meta = load_surface_points(pointcloud_npz_path, device=device)

    # --------------------------------------------------------
    # GT query points + occupancies 로드
    # -> 평가지표 1. Volume IoU
    # --------------------------------------------------------
    query_points, gt_occ, query_meta = load_query_points_and_occ(points_npz_path, device=device)

    # --------------------------------------------------------
    # 예측 mesh 로드
    # -> 3개 지표 모두 사용
    # --------------------------------------------------------
    mesh = load_mesh(mesh_path)
    mesh_info = get_mesh_info(mesh)
    print_mesh_info(Path(mesh_path).name, mesh_info)

    # --------------------------------------------------------
    # 예측 mesh 표면 샘플링
    # -> 평가지표 2. CD
    # -> 평가지표 3. F-score
    # --------------------------------------------------------
    pred_surface_points = sample_mesh_surface(
        mesh,
        n_points=n_surface_samples,
        seed=seed,
        device=device,
    )

    # --------------------------------------------------------
    # [평가지표 2] CD 계산
    # --------------------------------------------------------
    cd = chamfer_distance_torch(
        pred_surface_points,
        gt_surface_points,
        squared=squared_cd,
        chunk_size_src=chunk_size_src,
        chunk_size_dst=chunk_size_dst,
    )

    # --------------------------------------------------------
    # [평가지표 3] F-score 계산
    # --------------------------------------------------------
    f, precision, recall = fscore_torch(
        pred_surface_points,
        gt_surface_points,
        threshold=fscore_threshold,
        chunk_size_src=chunk_size_src,
        chunk_size_dst=chunk_size_dst,
    )

    # --------------------------------------------------------
    # [평가지표 1] Volume IoU 계산
    # --------------------------------------------------------
    viou, occ_method = volume_iou_from_mesh(
        mesh,
        query_points,
        gt_occ,
    )

    result = {
        "mesh_path": mesh_path,
        "Volume IoU": viou,            # 평가지표 1
        "CD": cd,                      # 평가지표 2
        "F-score": f,                  # 평가지표 3
        "Precision": precision,        # F-score 계산용 보조값
        "Recall": recall,              # F-score 계산용 보조값
        "Occupancy Method": occ_method,
        "Watertight": mesh_info["watertight"],
        "Winding Consistent": mesh_info["winding_consistent"],
        "Num Vertices": mesh_info["num_vertices"],
        "Num Faces": mesh_info["num_faces"],
        "BBox Min": mesh_info["bbox_min"],
        "BBox Max": mesh_info["bbox_max"],
        "GT surface loc": surface_meta["loc"],
        "GT surface scale": surface_meta["scale"],
        "GT query loc": query_meta["loc"],
        "GT query scale": query_meta["scale"],
        "n_surface_samples": n_surface_samples,
        "fscore_threshold": fscore_threshold,
        "CD_type": "squared_l2" if squared_cd else "l1",
        "seed": seed,
        "device": device,
    }
    return result


# ============================================================
# 10. Baseline / Ours 비교
# ============================================================
def compare_baseline_and_ours(
    baseline_mesh_path: str,
    ours_mesh_path: str,
    pointcloud_npz_path: str,
    points_npz_path: str,
    n_surface_samples: int = 50000,
    fscore_threshold: float = 0.01,
    squared_cd: bool = False,
    chunk_size_src: int = 2048,
    chunk_size_dst: int = 2048,
    seed: int = 42,
    device=DEVICE,
):
    baseline_result = evaluate_one_model(
        baseline_mesh_path,
        pointcloud_npz_path,
        points_npz_path,
        n_surface_samples=n_surface_samples,
        fscore_threshold=fscore_threshold,
        squared_cd=squared_cd,
        chunk_size_src=chunk_size_src,
        chunk_size_dst=chunk_size_dst,
        seed=seed,
        device=device,
    )

    ours_result = evaluate_one_model(
        ours_mesh_path,
        pointcloud_npz_path,
        points_npz_path,
        n_surface_samples=n_surface_samples,
        fscore_threshold=fscore_threshold,
        squared_cd=squared_cd,
        chunk_size_src=chunk_size_src,
        chunk_size_dst=chunk_size_dst,
        seed=seed,
        device=device,
    )

    df = pd.DataFrame([
        {
            "Model": "Baseline",
            **{k: v for k, v in baseline_result.items() if k != "mesh_path"},
        },
        {
            "Model": "Ours",
            **{k: v for k, v in ours_result.items() if k != "mesh_path"},
        }
    ])

    return df, baseline_result, ours_result


# ============================================================
# 11. 실행부
# ============================================================
if __name__ == "__main__":
    set_seed(42)

    baseline_mesh_path = "Baseline.ply"
    ours_mesh_path = "Ours.ply"
    pointcloud_npz_path = "pointcloud.npz"
    points_npz_path = "points.npz"

    print(f"Using device: {DEVICE}")

    df, baseline_result, ours_result = compare_baseline_and_ours(
        baseline_mesh_path=baseline_mesh_path,
        ours_mesh_path=ours_mesh_path,
        pointcloud_npz_path=pointcloud_npz_path,
        points_npz_path=points_npz_path,
        n_surface_samples=50000,
        fscore_threshold=0.01,
        squared_cd=False,
        chunk_size_src=2048,
        chunk_size_dst=2048,
        seed=42,
        device=DEVICE,
    )

    print("\n=== Quantitative Evaluation ===")
    print(df[[
        "Model",
        "Volume IoU",   # 평가지표 1
        "CD",           # 평가지표 2
        "F-score",      # 평가지표 3
        "Precision",
        "Recall",
        "Occupancy Method"
    ]])

    df.to_csv("evaluation_results_torch_final.csv", index=False, encoding="utf-8-sig")
    print("\nSaved: evaluation_results_torch_final.csv")

    summary = {
        "device": DEVICE,
        "n_surface_samples": 50000,
        "fscore_threshold": 0.01,
        "CD_type": "l1",
        "seed": 42,
        "baseline_result": baseline_result,
        "ours_result": ours_result,
    }
    with open("evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Saved: evaluation_summary.json")