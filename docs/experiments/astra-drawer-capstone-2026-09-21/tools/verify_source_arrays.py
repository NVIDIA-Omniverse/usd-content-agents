"""Portable read-only equivalent of the retained original-array source comparison."""

import argparse, hashlib, json
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, required=True)
p.add_argument("--candidate", type=Path, required=True)
p.add_argument("--output", type=Path)
a = p.parse_args()
doc = json.loads(a.source.read_text())
buffers = [(a.source.parent / x["uri"]).read_bytes() for x in doc["buffers"]]


def accessor(i):
    x = doc["accessors"][i]
    v = doc["bufferViews"][x["bufferView"]]
    dt = np.dtype({5123: "<u2", 5125: "<u4", 5126: "<f4"}[x["componentType"]])
    width = {"SCALAR": 1, "VEC3": 3}[x["type"]]
    return np.ndarray(
        (x["count"], width),
        dt,
        buffer=buffers[v["buffer"]],
        offset=v.get("byteOffset", 0) + x.get("byteOffset", 0),
        strides=(v.get("byteStride", dt.itemsize * width), dt.itemsize),
    ).copy()


stage = Usd.Stage.Open(str(a.candidate))
assert stage
cache = UsdGeom.XformCache()
rows = []
seen = []
for prim in stage.Traverse():
    if not prim.IsA(UsdGeom.Mesh):
        continue
    node_id = prim.GetCustomDataByKey("sourceGltfNode")
    assert type(node_id) is int
    node = doc["nodes"][node_id]
    assert not any(k in node for k in ["matrix", "translation", "rotation", "scale"])
    seen.append(node_id)
    src = doc["meshes"][node["mesh"]]["primitives"][0]
    points = accessor(src["attributes"]["POSITION"])
    indices = accessor(src["indices"]).reshape(-1)
    mesh = UsdGeom.Mesh(prim)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    checks = {
        "exact_points": np.array_equal(points, np.asarray(mesh.GetPointsAttr().Get())),
        "exact_original_indices": np.array_equal(
            indices, np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        ),
        "all_triangles": bool(np.all(counts == 3) and len(counts) * 3 == len(indices)),
        "original_world_transform": np.array_equal(
            np.asarray(cache.GetLocalToWorldTransform(prim)), np.eye(4)
        ),
    }
    rows.append(
        {
            "source_node": node["name"],
            "usd_path": str(prim.GetPath()),
            "points": len(points),
            "triangles": len(indices) // 3,
            "checks": checks,
        }
    )
x = {
    "scope": "Original visual source arrays only; no physical acceptance inferred.",
    "source_sha256": hashlib.sha256(a.source.read_bytes()).hexdigest(),
    "candidate_sha256": hashlib.sha256(a.candidate.read_bytes()).hexdigest(),
    "parts": rows,
    "unique_source_nodes": len(set(seen)),
    "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
    "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
    "passed": len(rows) == len(set(seen)) == len(doc["meshes"]) == 5
    and UsdGeom.GetStageMetersPerUnit(stage) == 1
    and str(UsdGeom.GetStageUpAxis(stage)) == "Y"
    and all(all(r["checks"].values()) for r in rows),
}
text = json.dumps(x, indent=2) + "\n"
if a.output:
    a.output.write_text(text)
print(text)
raise SystemExit(0 if x["passed"] else 1)
