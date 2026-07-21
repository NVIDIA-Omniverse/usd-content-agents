# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from world_understanding.functions.graphics import validate_usd as vu


class GeometryRule:
    pass


class MaterialRule:
    pass


class UnknownRule:
    pass


class _Issue:
    def __init__(
        self,
        rule: object,
        severity: object,
        *,
        message: str = "message",
        at: str | None = "/World",
        suggestion: str | None = "fix it",
    ) -> None:
        self.rule = rule
        self.severity = severity
        self.message = message
        self.at = at
        self.suggestion = suggestion


class _Results:
    def __init__(self, issues: list[_Issue]) -> None:
        self._issues = issues

    def issues(self) -> list[_Issue]:
        return self._issues


def _install_validator_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    issues: list[_Issue] | None = None,
    export_ok: bool = True,
    asset: object | None = None,
    registry_raises: bool = False,
    validate_raises: Exception | None = None,
) -> None:
    vu.clear_registered_rule_categories_cache()
    issues = issues or []

    class CategoryRuleRegistry:
        @property
        def categories(self) -> list[str]:
            if registry_raises:
                raise RuntimeError("registry failed")
            return ["Geometry", "Material"]

        def get_rules(self, category: str) -> list[type]:
            return {"Geometry": [GeometryRule], "Material": [MaterialRule]}.get(
                category, []
            )

    class ValidationEngine:
        def validate(self, path: str) -> _Results:
            if validate_raises:
                raise validate_raises
            return _Results(issues)

    class _RootLayer:
        def Export(self, path: str) -> bool:
            Path(path).write_text("fixed", encoding="utf-8")
            return export_ok

    class _Asset:
        def GetRootLayer(self) -> _RootLayer | None:
            return _RootLayer()

    class IssueFixer:
        def __init__(self, path: str) -> None:
            self.asset = _Asset() if asset is None else asset

        def fix(self, fix_issues: list[_Issue]) -> list[object]:
            return [
                SimpleNamespace(
                    issue=fix_issues[0],
                    status=SimpleNamespace(name="FIXED"),
                    exception=None,
                ),
                SimpleNamespace(
                    issue=fix_issues[0],
                    status=SimpleNamespace(name="FAILED"),
                    exception=RuntimeError("not fixed"),
                ),
            ]

    monkeypatch.setitem(
        sys.modules,
        "usd_validation_nvidia",
        SimpleNamespace(
            CategoryRuleRegistry=CategoryRuleRegistry,
            ValidationEngine=ValidationEngine,
            IssueFixer=IssueFixer,
        ),
    )


def test_ensure_usd_validation_compat_existing_and_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pxr.UsdValidation", object())
    vu._ensure_usd_validation_compat()

    monkeypatch.delitem(sys.modules, "pxr.UsdValidation", raising=False)
    real_import = builtins.__import__

    def fake_import(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
        if name == "pxr" and "UsdValidation" in fromlist:
            raise TypeError("broken binding")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    vu._ensure_usd_validation_compat()
    registry = sys.modules["pxr.UsdValidation"].ValidationRegistry()
    assert registry.GetOrLoadValidatorByName("Rule") is None


def test_availability_and_registered_categories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_validator_module(monkeypatch)
    assert vu.is_available() is True
    assert vu._registered_rule_categories() == {
        "GeometryRule": "Geometry",
        "MaterialRule": "Material",
    }
    vu.clear_registered_rule_categories_cache()

    _install_validator_module(monkeypatch, registry_raises=True)
    assert vu.is_available() is False
    with pytest.raises(RuntimeError):
        vu._load_rule_categories(required_for_filtering=True)
    assert vu._load_rule_categories(required_for_filtering=False) == {}

    monkeypatch.delitem(sys.modules, "usd_validation_nvidia", raising=False)
    vu.clear_registered_rule_categories_cache()
    real_import = builtins.__import__

    def fake_missing_import(
        name: str, globals=None, locals=None, fromlist=(), level: int = 0
    ):
        if name == "usd_validation_nvidia":
            raise ImportError("missing validator")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_missing_import)
    with pytest.raises(ImportError, match="CategoryRuleRegistry"):
        vu._registered_rule_categories()


def test_validate_usd_filters_categories_and_exports_fixes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "scene.usd"
    input_path.write_text("#usda", encoding="utf-8")
    output_path = tmp_path / "fixed" / "scene.usda"
    _install_validator_module(
        monkeypatch,
        issues=[
            _Issue(GeometryRule, SimpleNamespace(name="WARNING")),
            _Issue(MaterialRule, SimpleNamespace(name="FAILURE")),
            _Issue(
                UnknownRule, SimpleNamespace(name="ERROR"), at=None, suggestion=None
            ),
        ],
    )

    result = vu.validate_usd(
        input_path,
        categories=["Omni:Geometry"],
        fix=True,
        output_path=output_path,
    )

    assert result["status"] == "success"
    assert result["categories_checked"] == ["Geometry"]
    assert result["summary"] == {
        "total_issues": 2,
        "failures": 0,
        "warnings": 1,
        "errors": 1,
        "is_valid": False,
    }
    assert [issue["rule"] for issue in result["issues"]] == [
        "GeometryRule",
        "UnknownRule",
    ]
    assert result["issues"][1]["at"] is None
    assert result["issues"][1]["suggestion"] is None
    assert result["fixes"] == [
        {"rule": "GeometryRule", "status": "fixed", "message": None},
        {"rule": "GeometryRule", "status": "failed", "message": "not fixed"},
    ]
    assert output_path.read_text(encoding="utf-8") == "fixed"


def test_validate_usd_defaults_categories_and_handles_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "scene.usd"
    input_path.write_text("#usda", encoding="utf-8")
    _install_validator_module(monkeypatch, issues=[])

    result = vu.validate_usd(input_path)
    assert result["status"] == "success"
    assert result["categories_checked"] == list(vu.DEFAULT_VALIDATION_CATEGORIES)
    assert result["summary"]["is_valid"] is True

    with pytest.raises(ValueError, match="Input file does not exist"):
        vu.validate_usd(tmp_path / "missing.usd")

    _install_validator_module(monkeypatch, validate_raises=RuntimeError("bad validate"))
    result = vu.validate_usd(input_path)
    assert result["status"] == "error"
    assert result["error"] == "bad validate"


@pytest.mark.parametrize(
    ("asset", "export_ok", "output_path", "message"),
    [
        (None, True, "fixed.usdz", "must be a writable USD layer"),
        (
            SimpleNamespace(GetRootLayer=lambda: None),
            True,
            "fixed.usda",
            "no root layer",
        ),
        (None, False, "fixed.usda", "Failed to save fixed stage"),
    ],
)
def test_validate_usd_fix_export_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    asset: object | None,
    export_ok: bool,
    output_path: str,
    message: str,
) -> None:
    input_path = tmp_path / "scene.usd"
    input_path.write_text("#usda", encoding="utf-8")
    _install_validator_module(
        monkeypatch,
        issues=[_Issue(GeometryRule, SimpleNamespace(name="WARNING"))],
        export_ok=export_ok,
        asset=asset,
    )

    result = vu.validate_usd(
        input_path,
        categories=["Geometry"],
        fix=True,
        output_path=tmp_path / output_path,
    )

    assert result["status"] == "error"
    assert message in result["error"]


def test_validate_usd_fix_asset_none_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "scene.usd"
    input_path.write_text("#usda", encoding="utf-8")

    class IssueFixerWithoutAsset:
        def __init__(self, path: str) -> None:
            pass

        def fix(self, fix_issues: list[_Issue]) -> list[object]:
            return []

    _install_validator_module(
        monkeypatch,
        issues=[_Issue(GeometryRule, SimpleNamespace(name="WARNING"))],
    )
    sys.modules["usd_validation_nvidia"].IssueFixer = IssueFixerWithoutAsset

    result = vu.validate_usd(
        input_path,
        categories=["Geometry"],
        fix=True,
        output_path=tmp_path / "fixed.usda",
    )

    assert result["status"] == "error"
    assert "did not return a fixed USD asset" in result["error"]


def test_severity_rule_and_category_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    assert vu._map_severity(SimpleNamespace(name="FAILURE")) == "failure"
    assert vu._map_severity(SimpleNamespace(name="ERROR")) == "error"
    assert vu._map_severity("notice") == "warning"
    assert vu._get_rule_name(SimpleNamespace(rule=None)) == "Unknown"
    assert vu._get_rule_name(SimpleNamespace(rule="PlainRule")) == "PlainRule"

    _install_validator_module(monkeypatch, registry_raises=True)
    vu.clear_registered_rule_categories_cache()
    assert vu._infer_category(SimpleNamespace(rule=GeometryRule), None) == "Unknown"
    assert (
        vu._infer_category(
            SimpleNamespace(rule=GeometryRule), {"GeometryRule": "Geometry"}
        )
        == "Geometry"
    )
