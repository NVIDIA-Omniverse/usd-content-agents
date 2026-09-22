"""Create separate portable USD copies; never edit retained original bundle bytes."""

from __future__ import annotations
import argparse, hashlib, json, os, re, shutil
from pathlib import Path
from pxr import Sdf, Usd, UsdUtils  # Initialize file-format plugins.


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def non_asset_digest(layer):
    return hashlib.sha256(
        re.sub(r"@[^@]*@", "@ASSET_LOCATOR@", layer.ExportToString()).encode()
    ).hexdigest()


def relocate(source, target):
    if target.exists():
        raise ValueError(
            "Use a fresh output directory; original files are never overwritten."
        )
    manifest = json.loads((source / "publication_manifest.json").read_text())
    target.mkdir(parents=True)
    selected = [
        r
        for r in manifest["files"]
        if Path(r["path"]).suffix.lower()
        in {".usd", ".usda", ".usdc", ".usdz", ".gltf", ".bin", ".png", ".jpg", ".jpeg"}
    ]
    for row in selected:
        src = source / row["path"]
        assert digest(src) == row["published_sha256"]
        dst = target / row["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    changes = []
    for row in selected:
        path = target / row["path"]
        if path.suffix not in {".usd", ".usda", ".usdc"}:
            continue
        layer = Sdf.Layer.FindOrOpen(str(path))
        assert layer
        before = non_asset_digest(layer)
        rewrites = []

        def remap(locator):
            prefixes = [
                ("/opt/astra-content-value-20260921/capstone/", ""),
                (
                    "/opt/astra-content-value-20260921/assets/01_drawer/source/",
                    "source/",
                ),
            ]
            for prefix, replacement in prefixes:
                if locator.startswith(prefix):
                    relative = replacement + locator[len(prefix) :]
                    package = relative.split("[", 1)[0]
                    if not (target / package).is_file():
                        raise ValueError("Missing published dependency: " + relative)
                    result = os.path.relpath(target / package, path.parent) + (
                        ("[" + relative.split("[", 1)[1]) if "[" in relative else ""
                    )
                    rewrites.append(
                        {"original_locator": locator, "portable_locator": result}
                    )
                    return result
            if locator.startswith("/"):
                raise ValueError("Unmapped absolute USD dependency: " + locator)
            return locator

        UsdUtils.ModifyAssetPaths(layer, remap)
        assert non_asset_digest(layer) == before, "Non-asset USD opinions changed"
        if rewrites:
            layer.Save()
        changes.append(
            {
                "path": row["path"],
                "original_sha256": row["published_sha256"],
                "portable_sha256": digest(path),
                "only_asset_locators_changed": True,
                "non_asset_opinion_digest": before,
                "rewrites": rewrites,
            }
        )
    receipt = {
        "schema_version": "capstone-portable-usd-relocation.v1",
        "scope": "Presentation/inspection relocation only, not new simulation or native acceptance.",
        "original_publication_manifest_sha256": digest(
            source / "publication_manifest.json"
        ),
        "originals_unchanged": all(
            digest(source / r["path"]) == r["published_sha256"] for r in selected
        ),
        "changes": changes,
    }
    (target / "relocation_manifest.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("bundle", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    r = relocate(a.bundle.resolve(), a.output.resolve())
    print(
        json.dumps(
            {
                "originals_unchanged": r["originals_unchanged"],
                "usd_layers": len(r["changes"]),
                "rewritten_layers": sum(bool(c["rewrites"]) for c in r["changes"]),
            }
        )
    )
