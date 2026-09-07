# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted filesystem evidence adapter for the static qualification lane.

This is the production :class:`~joint_agent.static_qualification.
StaticQualificationEvidenceAdapter`. It reopens retained bytes from a sealed
evidence directory and *derives* each stage verdict from the validator
documents the Joint Rigger core release gate, Gate 3A, and Gate 3B already
emit. It never reads a verdict out of the untrusted result bundle -- the caller
compares this report against the bundle field by field, so anything echoed here
would make that comparison vacuous.

Establishing correspondence
---------------------------
None of the five emitters carries ``capability_id``, ``result_id``, or
``source_evidence_bindings``, and three of them carry no profile identifier
either. So "does this report belong to the stage it is being interpreted for?"
has to be answered from what the documents *do* carry. Three independent
sources have to agree, and any gap raises:

``claim``
    The retained ``result`` bytes are a
    ``joint-agent-static-qualification-stage-claim-v1`` record naming
    ``(capability_id, stage, profile_id, result_id, status)``. This is a
    *claim*, never trusted on its own.
``authorization``
    The adapter is constructed from a run plan that has already passed
    ``_validate_static_qualification_run_plan_against_manifest``, and holds the
    resulting ``(capability_id, stage) -> StaticStageBindingV1`` table. A claim
    with no exact table entry, or one whose profile/result differ from the
    admitted pair, is rejected, and every identity field this module emits is
    then read from the table rather than from any document that also supplies
    the verdict. Because *which* row applies would otherwise be selected by the
    operator-written claim, the table is refused outright when it spans more
    than one capability -- see capability attribution below.
``corroboration``
    The validator bytes have to agree, by digest, that they describe this
    artifact. See below.

Digest corroboration
--------------------
The retained ``validator`` role is the release-gate **authoring receipt**
(``joint-rigger-core-release-gate-authoring-v3``) for every stage. It is the
one document that binds, in a single record, the adopted plan
(``plan_adoption_sha256`` and per-asset ``plan.sha256``), the authored output
(``generated_usd.sha256``, ``generated_usd.identity.root_sha256``, and the
saved-stage reopen evidence ``generated_usd.validator_identity``), and the
closeout (``closeout.sha256``). Every stage anchors to it:

* exactly one authoring fixture must carry
  ``generated_usd.sha256 == observed_artifact_identity.root_sha256``. Zero
  matches means the report is for a different artifact; more than one is
  ambiguous. Both raise -- the adapter never picks a candidate;
* the retained ``contract`` bytes must hash to that fixture's ``plan.sha256``;
* the stage's own ``raw_report`` must declare ``schema_version ==
  validation_contract``, then satisfy its stage-specific digest anchor
  (``plan_adoption_sha256`` for ``contract``, byte identity with the retained
  receipt for ``authoring``, ``closeout.sha256`` for ``readback``, and a unique
  ``artifact_sha256`` result row for ``gate3a``/``gate3b``).

The root SHA-256 is the only digest that is comparable across the three
incompatible dependency-bundle schemes in this repository (the Joint Rigger
reference bundle, ``joint-agent-usd-artifact-dependency-bundle-v3``, and the
``<root>``-bearing closure this module's caller computes). Correspondence is
therefore keyed on the root digest alone -- which is sound only while that
digest *covers* the artifact, so it is enforced rather than assumed. See the
dependency closure limit below.

Fail-closed limits
------------------
Two things the retained bytes cannot establish are refused rather than
documented. Both raise a greppable token, because ``.coveragerc`` excludes
``raise (.*)Error`` and sets no ``branch = True``, so the coverage gate cannot
see either path.

* **Capability attribution is not corroborated by the evidence bytes**, so at
  most one capability may be scored per run
  (``static_qualification_ambiguous_capability_attribution``). The digest chain
  above ties a report to an *artifact*; nothing ties that artifact to a
  *capability*. No emitter names a capability, and the release gate's
  ``asset_id`` namespace (``simready_foundation_*``) is disjoint from the
  reference corpus namespace the manifest binds evidence in (``joint_ref_*``),
  so there is no sound in-repo map to check against. With two capabilities
  scheduled, one authored artifact plus one edited claim file would score for
  either of them and emit the manifest's corpus assets for whichever row was
  claimed, even though the run never touched them. So the binding table is
  refused when it spans more than one capability. With exactly one, the table
  lookup *is* the equality check -- a claim naming any other capability cannot
  resolve -- and attribution rests on the reviewed plan rather than on the
  claim. What this does **not** establish, and cannot: scoring the same
  artifact again under a second single-capability plan qualifies that
  capability too, and no scorecard can see the other run. The refusal moves
  attribution out of an editable claim file and into the reviewed plan; it does
  not corroborate it. Admitting multi-capability runs, or detecting a reused
  artifact across runs, needs the manifest to bind release-gate asset ids to
  capabilities, which is the manifest-owning stream's decision, not this
  module's.
* **A sealed dependency closure cannot be corroborated**, so only a retained
  output whose root digest is its whole identity can qualify
  (``static_qualification_unverifiable_dependency_closure``). An unchanged root
  ``.usda`` referencing a modified sublayer keeps its root digest, and simply
  not copying the sublayer into the sealed tree would hide it entirely. Neither
  is closable by recomputation: the receipt's
  ``identity.dependency_bundle_sha256`` is keyed on the authoring machine's
  resolved layer identifiers, and the ``validator_identity`` bundle is keyed on
  per-entry ``kind`` -- neither is recoverable from a flat copied directory. So
  the sealed closure must be empty, and the receipt's own bundle *entry count*
  must equal what the retained bytes cover: exactly one entry for a raw layer,
  and for a ``.usdz`` package one entry per archive member, which is the one
  count that **is** recomputable from the retained bytes. Until the receipt and
  the retained tree share a closure scheme, a raw USD output that references
  anything cannot qualify -- that is the fail-closed outcome, not a gap in the
  guard.

Stated limits
-------------
* ``contract``, ``authoring``, and ``readback`` have no profile field in their
  emitters, so the profile is not independently corroborated for them; it rests
  on the manifest's per-stage admission plus the digest anchors above.
* Those three emitters also carry a *disposition* rather than a status token,
  so this module projects them onto the admitted status vocabulary with a
  closed, total map (``apply``/``authored``+``passed``/published saved stage ->
  ``PASS``; anything else -> ``FAIL`` with a synthesized finding). Any token
  outside the map raises.
* ``joint-rigger-core-release-gate-closeout-v1`` records saved-stage paths and
  nothing else -- no digest and no verdict. ``readback`` is therefore anchored
  through the authoring receipt that published the closeout, and its only
  derivable failure is a closeout that names the asset without publishing a
  saved stage. Until that contract emits a per-asset digest and status, that is
  the honest ceiling for this stage.
* Gate 3B must be run with ``--include-artifact-sha256``; without it no result
  row carries ``artifact_sha256``, no row matches, and the stage fails closed.
"""

from __future__ import annotations

import hashlib
import io
import json
import stat
import zipfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, JsonValue
from world_understanding.functions.physics.joint_rigger import ArtifactIdentityV1

from joint_agent.capability_manifest import (
    CapabilityManifestError,
    EvidenceBindingV1,
    loads_json_document,
)
from joint_agent.static_qualification import (
    STATIC_STAGE_NAMES,
    STATIC_STAGE_VALIDATION_CONTRACTS,
    StaticQualificationRunPlanV1,
    StaticResolvedDependencyV1,
    StaticResolvedUsdArtifactV1,
    StaticRetainedArtifactRole,
    StaticStageName,
    StaticTrustedStageReportV1,
)
from joint_agent.static_qualification_contracts import (
    require_registered_static_validation_contract,
)

STATIC_QUALIFICATION_STAGE_CLAIM_SCHEMA_VERSION: Literal[
    "joint-agent-static-qualification-stage-claim-v1"
] = "joint-agent-static-qualification-stage-claim-v1"

RELEASE_GATE_AUTHORING_CONTRACT = STATIC_STAGE_VALIDATION_CONTRACTS["authoring"]

# The runner copies the resolved dependency closure of a retained USD output
# into this sibling directory. A `.usdz` package is self-contained -- its root
# bytes already cover every member -- so its closure directory is empty.
DEPENDENCY_DIRECTORY_SUFFIX = ".dependencies"

# The scheme the release gate's saved-stage reopen evidence
# (`generated_usd.validator_identity`) is emitted under. Its `entry_count`
# counts the root layer plus every dependency, so `1` means the root digest is
# the artifact's whole identity.
ARTIFACT_DEPENDENCY_BUNDLE_SCHEMA_VERSION = (
    "joint-agent-usd-artifact-dependency-bundle-v3"
)

# Every USDZ package is a ZIP archive, so its first four bytes are the
# local-file-header signature. No raw `.usda`/`.usdc` layer starts with it.
# The signature only *selects* which entry count the receipt has to attest --
# it is never on its own a reason to skip that attestation.
USDZ_PACKAGE_MAGIC = b"PK\x03\x04"

# Stable failure tokens. `.coveragerc` excludes `raise (.*)Error` and sets no
# `branch = True`, so the coverage gate cannot see these paths; they are
# greppable instead, and each has a dedicated test.
AMBIGUOUS_CAPABILITY_ATTRIBUTION_CODE = (
    "static_qualification_ambiguous_capability_attribution"
)
UNVERIFIABLE_DEPENDENCY_CLOSURE_CODE = (
    "static_qualification_unverifiable_dependency_closure"
)
UNREADABLE_SEALED_EVIDENCE_CODE = "static_qualification_unreadable_sealed_evidence"


class _EvidenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StaticStageClaimV1(_EvidenceModel):
    """The retained ``result`` record naming the stage its bytes belong to."""

    schema_version: Literal["joint-agent-static-qualification-stage-claim-v1"]
    capability_id: str
    stage: StaticStageName
    profile_id: str
    result_id: str
    status: Literal["pass", "fail", "error"]


class StaticStageBindingV1(_EvidenceModel):
    """One ``execution="run"`` stage identity the manifest already admitted."""

    capability_id: str
    stage: StaticStageName
    validation_contract: str
    profile_id: str
    result_id: str
    command: tuple[str, ...]
    tool_id: str
    tool_version: str
    source_evidence_bindings: tuple[EvidenceBindingV1, ...]


def static_stage_bindings_from_run_plan(
    plan: StaticQualificationRunPlanV1,
) -> tuple[StaticStageBindingV1, ...]:
    """Project every scheduled stage of an already-validated run plan."""

    bindings = []
    for row in plan.rows:
        for stage in STATIC_STAGE_NAMES:
            stage_plan = getattr(row.stages, stage)
            if stage_plan.execution != "run":
                continue
            bindings.append(
                StaticStageBindingV1.model_validate(
                    {
                        "capability_id": row.capability_id,
                        "stage": stage,
                        "validation_contract": stage_plan.validation_contract,
                        "profile_id": stage_plan.profile_id,
                        "result_id": stage_plan.result_id,
                        "command": stage_plan.command,
                        "tool_id": stage_plan.tool_id,
                        "tool_version": stage_plan.tool_version,
                        "source_evidence_bindings": row.evidence_bindings,
                    }
                )
            )
    return tuple(bindings)


def canonical_stage_claim_json(claim: StaticStageClaimV1) -> str:
    """Serialize the retained ``result`` record deterministically."""

    return json.dumps(
        claim.model_dump(mode="json"),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


class _StageOutcome(NamedTuple):
    status: Literal["pass", "fail", "error"]
    raw_status: str
    raw_findings: tuple[dict[str, JsonValue], ...]
    blocking_reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class _StageContext:
    """Everything one stage interpreter is allowed to read."""

    document: Mapping[str, Any]
    receipt: Mapping[str, Any]
    fixture: Mapping[str, Any]
    artifacts: Mapping[StaticRetainedArtifactRole, bytes]
    root_sha256: str
    profile_id: str


def _require_object(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CapabilityManifestError(f"{label} must be a JSON object")
    return value


def _require_list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise CapabilityManifestError(f"{label} must be a JSON array")
    return value


def _require_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CapabilityManifestError(f"{label} must be a nonblank string")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sorted_children(directory: Path) -> list[Path]:
    """Materialize one directory listing so its errors surface at call time."""

    return sorted(directory.iterdir())


def _guard[T](operation: Callable[[], T], *, label: str) -> T:
    """Run one filesystem operation, reporting failure as a policy refusal.

    The adapter's callers catch ``CapabilityManifestError``, so an ``OSError``
    from an incomplete or unreadable evidence directory would surface as a
    traceback. ``RuntimeError`` is caught with it because that -- not
    ``OSError`` -- is what ``Path.resolve`` raises on a symlink loop.
    """

    try:
        return operation()
    except (OSError, RuntimeError) as exc:
        raise CapabilityManifestError(
            f"{UNREADABLE_SEALED_EVIDENCE_CODE}: {label}: {exc}"
        ) from exc


def _release_gate_canonical_sha256(document: Mapping[str, Any]) -> str:
    """Reproduce the release gate's own canonical receipt digest exactly."""

    encoded = json.dumps(
        document,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _basename(value: str) -> str:
    return PurePosixPath(value.replace("\\", "/")).name


def _required_basename(value: Any, *, label: str) -> str:
    """Return the file name a retained path names, or refuse the path.

    ``PurePosixPath('/').name`` and ``PurePosixPath('.').name`` are both ``''``,
    so two paths that name no file compare equal and corroborate nothing --
    the same vacuum two absent fields would leave.
    """

    name = _basename(_require_text(value, label=label))
    if not name:
        raise CapabilityManifestError(f"{label} must name a file")
    return name


def _package_member_count(root_bytes: bytes) -> int:
    """Count the archive members of a retained USDZ package, or refuse it.

    The v3 emitter writes exactly one bundle entry per archive member (files
    and directories alike), so for a package this count -- unlike either
    bundle *digest* -- is recomputable from the retained bytes and can be held
    against the receipt.
    """

    try:
        with zipfile.ZipFile(io.BytesIO(root_bytes)) as archive:
            return len(archive.infolist())
    except (OSError, zipfile.BadZipFile) as exc:
        raise CapabilityManifestError(
            f"{UNVERIFIABLE_DEPENDENCY_CLOSURE_CODE}: the retained output opens "
            "with the ZIP local-file-header signature but its package members "
            f"cannot be listed: {exc}"
        ) from exc


def _json_findings(value: Any, *, label: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        _require_object(entry, label=f"{label} entry")
        for entry in _require_list(value, label=label)
    )


def _synthesized_finding(*, rule: str, message: str) -> dict[str, JsonValue]:
    return {"severity": "ERROR", "rule": rule, "message": message}


def _unique_artifact_result(
    value: Any,
    *,
    root_sha256: str,
    label: str,
) -> dict[str, Any]:
    """Select the single result row whose artifact digest is the retained one."""

    matches = [
        entry
        for entry in (
            _require_object(item, label=f"{label} result")
            for item in _require_list(value, label=f"{label} results")
        )
        if entry.get("artifact_sha256") == root_sha256
    ]
    if len(matches) != 1:
        raise CapabilityManifestError(
            f"{label} report carries {len(matches)} results whose artifact "
            "SHA-256 equals the retained output; exactly one is required"
        )
    return matches[0]


def _authoring_receipt(payload: bytes) -> dict[str, Any]:
    receipt: dict[str, Any] = loads_json_document(
        payload,
        label="release-gate authoring receipt",
    )
    if receipt.get("schema_version") != RELEASE_GATE_AUTHORING_CONTRACT:
        raise CapabilityManifestError(
            "retained validator bytes must be a "
            f"{RELEASE_GATE_AUTHORING_CONTRACT} receipt"
        )
    return receipt


def _fixture_output_sha256(fixture: Mapping[str, Any]) -> Any:
    generated = fixture.get("generated_usd")
    if isinstance(generated, dict):
        return generated.get("sha256")
    return None


def _fixture_request_sha256(fixture: Mapping[str, Any]) -> Any:
    """Read one adoption fixture's request digest, or nothing if it has none.

    A real plan-adoption receipt lists the whole roster, and a
    ``source_data_excluded`` fixture never acquires a plan, so it carries no
    ``request`` record at all. Such a fixture is skipped exactly the way an
    unauthored fixture is skipped by `_authored_fixture` below -- it can never be the match, and
    demanding a request record from it would reject every genuine receipt.
    """

    request = fixture.get("request")
    if isinstance(request, dict):
        return request.get("sha256")
    return None


def _authored_fixture(
    receipt: Mapping[str, Any],
    *,
    root_sha256: str,
) -> dict[str, Any]:
    """Select the single authored asset whose output is the retained output."""

    matches = [
        fixture
        for fixture in (
            _require_object(item, label="release-gate authoring fixture")
            for item in _require_list(
                receipt.get("fixtures"),
                label="release-gate authoring fixtures",
            )
        )
        if _fixture_output_sha256(fixture) == root_sha256
    ]
    if len(matches) != 1:
        raise CapabilityManifestError(
            f"release-gate authoring receipt carries {len(matches)} assets whose "
            "authored output SHA-256 equals the retained output; exactly one is "
            "required"
        )
    fixture = matches[0]
    generated = _require_object(
        fixture.get("generated_usd"),
        label="release-gate authored output record",
    )
    identity = _require_object(
        generated.get("identity"),
        label="release-gate authored output identity",
    )
    if identity.get("root_sha256") != root_sha256:
        raise CapabilityManifestError(
            "release-gate authored output identity disagrees with its own file "
            "digest for the retained output"
        )
    return fixture


def _require_complete_artifact_identity(
    fixture: Mapping[str, Any],
    *,
    resolved_output: StaticResolvedUsdArtifactV1,
    root_sha256: str,
) -> None:
    """Refuse a retained output whose root digest is not its whole identity.

    Correspondence is keyed on ``root_sha256`` because no two of the three
    dependency-bundle schemes in this repository are comparable (see the module
    docstring). That is only sound while the root digest *covers* the artifact.
    A raw ``.usda`` that references a sublayer has a root digest that is
    unchanged by editing that sublayer, so an unchanged root would otherwise
    qualify a modified tree -- and simply not copying the sublayer into the
    sealed closure would hide it.

    Neither hole is closable by recomputing a digest: the release gate's
    ``identity.dependency_bundle_sha256`` is keyed on the authoring machine's
    resolved layer identifiers and the ``validator_identity`` bundle is keyed on
    per-entry ``kind``, and neither is recoverable from a flat copied directory.
    So this fails closed instead: the sealed closure must be empty, and the
    authoring receipt's bundle entry count must equal what the retained bytes
    themselves cover -- one entry for a raw layer, and one entry per archive
    member for a ``.usdz`` package, which is the one count that can be
    recomputed here. Reading the ZIP signature as a licence to skip that
    attestation would be the whole hole again: any bytes at all can be given
    that prefix.
    """

    if _sha256_bytes(resolved_output.root_bytes) != root_sha256:
        raise CapabilityManifestError(
            "retained output bytes are not the artifact its observed identity covers"
        )
    generated = _require_object(
        fixture.get("generated_usd"),
        label="release-gate authored output record",
    )
    reopened = _require_object(
        generated.get("validator_identity"),
        label="release-gate saved-stage reopen evidence",
    )
    declared_schema = _require_text(
        reopened.get("artifact_dependency_bundle_schema_version"),
        label="release-gate saved-stage dependency bundle schema version",
    )
    if declared_schema != ARTIFACT_DEPENDENCY_BUNDLE_SCHEMA_VERSION:
        raise CapabilityManifestError(
            f"{UNVERIFIABLE_DEPENDENCY_CLOSURE_CODE}: release-gate saved-stage "
            f"reopen evidence declares {declared_schema!r}, not "
            f"{ARTIFACT_DEPENDENCY_BUNDLE_SCHEMA_VERSION!r}"
        )
    if reopened.get("artifact_sha256") != root_sha256:
        raise CapabilityManifestError(
            "release-gate saved-stage reopen evidence does not match the "
            "retained output"
        )
    if resolved_output.dependencies:
        raise CapabilityManifestError(
            f"{UNVERIFIABLE_DEPENDENCY_CLOSURE_CODE}: the retained output seals "
            f"{len(resolved_output.dependencies)} dependencies, and no digest any "
            "retained document carries can be recomputed over them, so the "
            "closure cannot be corroborated"
        )
    covered_entries = (
        _package_member_count(resolved_output.root_bytes)
        if resolved_output.root_bytes.startswith(USDZ_PACKAGE_MAGIC)
        else 1
    )
    entry_count = reopened.get("artifact_dependency_bundle_entry_count")
    # Raw JSON, never a pydantic field: `True` and `1.0` both equal `1`.
    if (
        isinstance(entry_count, bool)
        or not isinstance(entry_count, int)
        or entry_count != covered_entries
    ):
        raise CapabilityManifestError(
            f"{UNVERIFIABLE_DEPENDENCY_CLOSURE_CODE}: the release gate hashed the "
            f"retained USD output over {entry_count!r} dependency bundle entries "
            f"while its retained bytes cover exactly {covered_entries}, so its "
            "root digest is not a complete artifact identity"
        )


def _interpret_contract(context: _StageContext) -> _StageOutcome:
    adopted_digest = _require_text(
        context.receipt.get("plan_adoption_sha256"),
        label="release-gate adopted plan digest",
    )
    if _release_gate_canonical_sha256(context.document) != adopted_digest:
        raise CapabilityManifestError(
            "retained plan-adoption receipt is not the receipt the authoring "
            "receipt adopted"
        )
    contract_digest = _sha256_bytes(context.artifacts["contract"])
    matches = [
        fixture
        for fixture in (
            _require_object(item, label="plan-adoption fixture")
            for item in _require_list(
                context.document.get("fixtures"),
                label="plan-adoption fixtures",
            )
        )
        if _fixture_request_sha256(fixture) == contract_digest
    ]
    if len(matches) != 1:
        raise CapabilityManifestError(
            f"plan-adoption receipt carries {len(matches)} requests whose "
            "SHA-256 equals the retained contract bytes; exactly one is required"
        )
    adopted = matches[0]
    # Both sides have to be present before they can be compared: two absent
    # fields are equal, so a receipt that names no asset would otherwise
    # corroborate any authoring fixture that also names none.
    adopted_asset_id = _require_text(
        adopted.get("asset_id"),
        label="plan-adoption adopted asset ID",
    )
    authored_asset_id = _require_text(
        context.fixture.get("asset_id"),
        label="release-gate authored asset ID",
    )
    if adopted_asset_id != authored_asset_id:
        raise CapabilityManifestError(
            "plan-adoption receipt adopted the retained contract for a different "
            "asset than the one that authored the retained output"
        )
    disposition = _require_text(
        adopted.get("disposition"),
        label="plan-adoption disposition",
    )
    if disposition == "apply":
        return _StageOutcome("pass", "PASS", (), ())
    return _StageOutcome(
        "fail",
        "FAIL",
        (
            _synthesized_finding(
                rule="plan_adoption_disposition",
                message=f"plan adoption disposition {disposition!r} is not 'apply'",
            ),
        ),
        (),
    )


def _interpret_authoring(context: _StageContext) -> _StageOutcome:
    if context.artifacts["raw_report"] != context.artifacts["validator"]:
        raise CapabilityManifestError(
            "authoring raw report must be the retained release-gate authoring "
            "receipt itself"
        )
    gate2 = _require_object(
        context.fixture.get("gate2"),
        label="release-gate Gate 2 record",
    )
    identity = _require_object(
        gate2.get("artifact_identity"),
        label="release-gate Gate 2 artifact identity",
    )
    if identity.get("root_sha256") != context.root_sha256:
        raise CapabilityManifestError(
            "release-gate Gate 2 validated a different artifact than the "
            "retained output"
        )
    disposition = _require_text(
        context.fixture.get("disposition"),
        label="release-gate authoring disposition",
    )
    gate2_status = _require_text(
        gate2.get("status"), label="release-gate Gate 2 status"
    )
    if disposition == "authored" and gate2_status == "passed":
        return _StageOutcome("pass", "PASSED", (), ())
    return _StageOutcome(
        "fail",
        "FAIL",
        (
            _synthesized_finding(
                rule="release_gate_authoring",
                message=(
                    f"release-gate authoring disposition {disposition!r} with "
                    f"Gate 2 status {gate2_status!r} is not an authored pass"
                ),
            ),
        ),
        (),
    )


def _interpret_readback(context: _StageContext) -> _StageOutcome:
    closeout_record = _require_object(
        context.receipt.get("closeout"),
        label="release-gate closeout record",
    )
    published_digest = _require_text(
        closeout_record.get("sha256"),
        label="release-gate closeout digest",
    )
    if published_digest != _sha256_bytes(context.artifacts["raw_report"]):
        raise CapabilityManifestError(
            "retained core closeout is not the closeout the authoring receipt published"
        )
    asset_id = _require_text(
        context.fixture.get("asset_id"),
        label="release-gate authored asset ID",
    )
    matches = [
        asset
        for asset in (
            _require_object(item, label="core closeout asset")
            for item in _require_list(
                context.document.get("assets"),
                label="core closeout assets",
            )
        )
        if asset.get("asset_id") == asset_id
    ]
    if len(matches) != 1:
        raise CapabilityManifestError(
            f"core closeout carries {len(matches)} records for {asset_id}; "
            "exactly one is required"
        )
    saved_stage_path = matches[0].get("generated_usd_path")
    if saved_stage_path is None:
        return _StageOutcome(
            "fail",
            "FAIL",
            (
                _synthesized_finding(
                    rule="saved_stage_not_published",
                    message=(
                        f"core closeout closed {asset_id} without publishing a "
                        "saved stage"
                    ),
                ),
            ),
            (),
        )
    generated = _require_object(
        context.fixture.get("generated_usd"),
        label="release-gate authored output record",
    )
    if _required_basename(
        saved_stage_path,
        label="core closeout saved-stage path",
    ) != _required_basename(
        generated.get("path"),
        label="release-gate authored output path",
    ):
        raise CapabilityManifestError(
            "core closeout saved-stage path does not name the retained output"
        )
    return _StageOutcome("pass", "PASS", (), ())


def _interpret_gate3a(context: _StageContext) -> _StageOutcome:
    profile = _require_object(
        context.document.get("validation_profile"),
        label="Gate 3A validation profile",
    )
    declared = _require_text(
        profile.get("name"),
        label="Gate 3A validation profile name",
    )
    if declared != context.profile_id:
        raise CapabilityManifestError(
            f"Gate 3A report ran profile {declared!r} but the plan admitted "
            f"{context.profile_id!r}"
        )
    entry = _unique_artifact_result(
        context.document.get("results"),
        root_sha256=context.root_sha256,
        label="Gate 3A",
    )
    raw_status = _require_text(entry.get("status"), label="Gate 3A result status")
    findings = _json_findings(entry.get("issues"), label="Gate 3A issues")
    if raw_status == "pass":
        return _StageOutcome("pass", raw_status, findings, ())
    if raw_status in {"fail", "warning"}:
        return _StageOutcome("fail", raw_status, findings, ())
    if raw_status == "validator_exception":
        return _StageOutcome(
            "error",
            raw_status,
            findings,
            ("gate3a_validator_exception",),
        )
    raise CapabilityManifestError(
        f"Gate 3A result status {raw_status!r} is not a completed outcome"
    )


def _foundation_findings(entry: Mapping[str, Any]) -> tuple[dict[str, JsonValue], ...]:
    """Project Foundation's severity-free message lists onto graded findings."""

    findings: list[dict[str, JsonValue]] = [
        {"severity": "ERROR", "message": _require_text(message, label="Gate 3B error")}
        for message in _require_list(entry.get("errors"), label="Gate 3B errors")
    ]
    findings.extend(
        {
            "severity": "WARNING",
            "message": _require_text(message, label="Gate 3B warning"),
        }
        for message in _require_list(entry.get("warnings"), label="Gate 3B warnings")
    )
    return tuple(findings)


def _interpret_gate3b(context: _StageContext) -> _StageOutcome:
    profile = _require_object(
        context.document.get("profile"),
        label="Gate 3B profile",
    )
    declared = _require_text(profile.get("target"), label="Gate 3B profile target")
    if declared != context.profile_id:
        raise CapabilityManifestError(
            f"Gate 3B report ran profile {declared!r} but the plan admitted "
            f"{context.profile_id!r}"
        )
    entry = _unique_artifact_result(
        context.document.get("results"),
        root_sha256=context.root_sha256,
        label="Gate 3B",
    )
    entry_profile = _require_text(
        entry.get("profile_target"),
        label="Gate 3B result profile target",
    )
    if entry_profile != context.profile_id:
        raise CapabilityManifestError(
            f"Gate 3B validated the retained output under {entry_profile!r} but "
            f"the plan admitted {context.profile_id!r}"
        )
    raw_status = _require_text(entry.get("status"), label="Gate 3B result status")
    findings = _foundation_findings(entry)
    if raw_status == "PASS":
        return _StageOutcome("pass", raw_status, findings, ())
    if raw_status == "FAIL":
        return _StageOutcome("fail", raw_status, findings, ())
    if raw_status in {"BLOCKED", "ERROR"}:
        return _StageOutcome(
            "error",
            raw_status,
            findings,
            (f"gate3b_{raw_status.lower()}",),
        )
    raise CapabilityManifestError(
        f"Gate 3B result status {raw_status!r} is not a completed outcome"
    )


_STAGE_INTERPRETERS: Mapping[
    StaticStageName,
    Callable[[_StageContext], _StageOutcome],
] = {
    "contract": _interpret_contract,
    "authoring": _interpret_authoring,
    "readback": _interpret_readback,
    "gate3a": _interpret_gate3a,
    "gate3b": _interpret_gate3b,
}


class FilesystemStaticQualificationEvidenceAdapter:
    """Reopen sealed retained bytes and derive each stage report from them.

    ``evidence_root`` seals the byte namespace: every retained URI is a
    canonical relative POSIX path under it, resolved without following
    symlinks and rejected if it escapes. Anything under it that cannot be
    resolved, listed, or read is a refusal, never a truncated closure and never
    a traceback.

    ``bindings`` may cover at most one capability. Nothing in the retained bytes
    names a capability, so a table spanning two of them would let the
    operator-written claim choose which one an artifact qualified.
    """

    def __init__(
        self,
        *,
        evidence_root: str | Path,
        bindings: Iterable[StaticStageBindingV1],
    ) -> None:
        self._evidence_root = _guard(
            lambda: Path(evidence_root).resolve(),
            label=f"sealed evidence root {evidence_root}",
        )
        table: dict[tuple[str, StaticStageName], StaticStageBindingV1] = {}
        for binding in bindings:
            key = (binding.capability_id, binding.stage)
            if key in table:
                raise CapabilityManifestError(
                    f"static stage bindings repeat {binding.capability_id}/"
                    f"{binding.stage}"
                )
            table[key] = binding
        scheduled = sorted({capability_id for capability_id, _ in table})
        if len(scheduled) > 1:
            raise CapabilityManifestError(
                f"{AMBIGUOUS_CAPABILITY_ATTRIBUTION_CODE}: the run plan schedules "
                f"{len(scheduled)} capabilities ({', '.join(scheduled)}), and no "
                "retained document names a capability, so one authored artifact "
                "would qualify whichever capability its operator-written claim "
                "names; score one capability per run"
            )
        self._bindings = table

    def _resolve(self, uri: str) -> Path:
        relative = uri.strip()
        path = PurePosixPath(relative)
        if (
            not relative
            or "\\" in relative
            or path.is_absolute()
            or path.as_posix() != relative
            or any(part in {".", ".."} for part in path.parts)
        ):
            raise CapabilityManifestError(
                f"retained artifact URI {uri!r} must be a canonical relative POSIX path"
            )
        candidate = self._evidence_root / path
        resolved = _guard(
            candidate.resolve,
            label=f"retained artifact URI {uri!r}",
        )
        if not resolved.is_relative_to(self._evidence_root):
            raise CapabilityManifestError(
                f"retained artifact URI {uri!r} escapes the sealed evidence root"
            )
        return candidate

    def _regular_file(self, path: Path, *, label: str) -> bytes:
        metadata = _guard(path.lstat, label=f"{label} at {path}")
        if not stat.S_ISREG(metadata.st_mode):
            raise CapabilityManifestError(
                f"{label} must be a non-symlink regular file: {path}"
            )
        return _guard(path.read_bytes, label=f"{label} at {path}")

    def _sealed_closure_members(self, closure_root: Path) -> list[Path]:
        """List every non-directory member of a sealed closure, or refuse.

        ``Path.rglob`` swallows a ``PermissionError`` on a sub-directory and
        returns a shorter closure, so an operator could hide a sealed member
        with ``chmod 000`` and the adapter would report the truncated tree
        without complaint. Walking it by hand turns that silent loss into a
        refusal.
        """

        members: list[Path] = []
        pending = [closure_root]
        while pending:
            directory = pending.pop()
            label = f"sealed dependency directory {directory}"
            for child in _guard(partial(_sorted_children, directory), label=label):
                metadata = _guard(child.lstat, label=f"sealed dependency {child}")
                if stat.S_ISDIR(metadata.st_mode):
                    pending.append(child)
                    continue
                members.append(child)
        return sorted(members)

    def read_bytes(self, uri: str) -> bytes:
        """Return the retained bytes of one sealed evidence artifact."""

        path = self._resolve(uri)
        return self._regular_file(path, label="retained evidence artifact")

    def reopen_usd_artifact(self, uri: str) -> StaticResolvedUsdArtifactV1:
        """Reopen a retained USD output together with its sealed closure."""

        path = self._resolve(uri)
        root_bytes = self._regular_file(path, label="retained USD output")
        closure_root = path.parent / f"{path.name}{DEPENDENCY_DIRECTORY_SUFFIX}"
        try:
            closure_metadata = closure_root.lstat()
        except OSError as exc:
            raise CapabilityManifestError(
                "retained USD output requires a sealed dependency directory at "
                f"{closure_root}"
            ) from exc
        if not stat.S_ISDIR(closure_metadata.st_mode):
            raise CapabilityManifestError(
                f"retained USD dependency closure must be a directory: {closure_root}"
            )
        dependencies = [
            StaticResolvedDependencyV1(
                path=member.relative_to(closure_root).as_posix(),
                payload=self._regular_file(member, label="retained USD dependency"),
            )
            for member in self._sealed_closure_members(closure_root)
        ]
        return StaticResolvedUsdArtifactV1(
            uri=uri,
            root_bytes=root_bytes,
            dependencies=tuple(dependencies),
        )

    def _authorized_binding(
        self,
        *,
        stage: StaticStageName,
        claim: StaticStageClaimV1,
    ) -> StaticStageBindingV1:
        if claim.stage != stage:
            raise CapabilityManifestError(
                f"retained result claims stage {claim.stage!r} while being "
                f"interpreted as {stage!r}"
            )
        binding = self._bindings.get((claim.capability_id, claim.stage))
        if binding is None:
            raise CapabilityManifestError(
                f"retained result claims {claim.capability_id}/{claim.stage}, "
                "which the run plan never scheduled"
            )
        if (
            claim.profile_id != binding.profile_id
            or claim.result_id != binding.result_id
        ):
            raise CapabilityManifestError(
                f"retained result claims profile {claim.profile_id!r} and result "
                f"{claim.result_id!r}, which the run plan did not admit for "
                f"{claim.capability_id}/{claim.stage}"
            )
        return binding

    def interpret_stage(
        self,
        *,
        stage: StaticStageName,
        validation_contract: str,
        artifacts: Mapping[StaticRetainedArtifactRole, bytes],
        resolved_output: StaticResolvedUsdArtifactV1,
        observed_artifact_identity: ArtifactIdentityV1,
    ) -> StaticTrustedStageReportV1:
        """Derive one stage report from retained validator bytes."""

        try:
            require_registered_static_validation_contract(
                stage=stage,
                value=validation_contract,
            )
        except ValueError as exc:
            raise CapabilityManifestError(
                f"{stage} evidence declares validation contract "
                f"{validation_contract!r} for a different stage"
            ) from exc
        claim = _stage_claim(artifacts["result"])
        binding = self._authorized_binding(stage=stage, claim=claim)
        if validation_contract != binding.validation_contract:
            raise CapabilityManifestError(
                f"{stage} evidence declares validation contract "
                f"{validation_contract!r}, but the run plan admitted "
                f"{binding.validation_contract!r}"
            )
        if stage in {"contract", "authoring", "readback"} and (
            validation_contract != STATIC_STAGE_VALIDATION_CONTRACTS[stage]
        ):
            raise CapabilityManifestError(
                f"{stage} validation contract {validation_contract!r} requires "
                "a lane-specific evidence interpreter"
            )
        receipt = _authoring_receipt(artifacts["validator"])
        root_sha256 = observed_artifact_identity.root_sha256
        fixture = _authored_fixture(receipt, root_sha256=root_sha256)
        _require_complete_artifact_identity(
            fixture,
            resolved_output=resolved_output,
            root_sha256=root_sha256,
        )
        plan_record = _require_object(
            fixture.get("plan"),
            label="release-gate adopted plan record",
        )
        if _require_text(
            plan_record.get("sha256"),
            label="release-gate adopted plan digest",
        ) != _sha256_bytes(artifacts["contract"]):
            raise CapabilityManifestError(
                "retained contract bytes are not the plan the release gate "
                "adopted for the retained output"
            )
        document = loads_json_document(
            artifacts["raw_report"],
            label=f"{stage} raw report",
        )
        if document.get("schema_version") != validation_contract:
            raise CapabilityManifestError(
                f"{stage} raw report does not declare {validation_contract!r}"
            )
        outcome = _STAGE_INTERPRETERS[stage](
            _StageContext(
                document=document,
                receipt=receipt,
                fixture=fixture,
                artifacts=artifacts,
                root_sha256=root_sha256,
                profile_id=binding.profile_id,
            )
        )
        if outcome.status != claim.status:
            raise CapabilityManifestError(
                f"retained result claims {claim.status!r} but the retained "
                f"{stage} report derives {outcome.status!r}"
            )
        return StaticTrustedStageReportV1(
            status=outcome.status,
            raw_status=outcome.raw_status,
            capability_id=binding.capability_id,
            source_evidence_bindings=binding.source_evidence_bindings,
            profile_id=binding.profile_id,
            result_id=binding.result_id,
            command=binding.command,
            tool_id=binding.tool_id,
            tool_version=binding.tool_version,
            observed_artifact_identity=observed_artifact_identity,
            contract_validation_status=(
                "error" if outcome.status == "error" else "pass"
            ),
            raw_findings=outcome.raw_findings,
            blocking_reason_codes=outcome.blocking_reason_codes,
        )


def _stage_claim(payload: bytes) -> StaticStageClaimV1:
    document = loads_json_document(
        payload,
        label="static qualification stage claim",
    )
    try:
        return StaticStageClaimV1.model_validate(document)
    except ValueError as exc:
        raise CapabilityManifestError(
            f"static qualification stage claim is invalid: {exc}"
        ) from exc


__all__ = [
    "AMBIGUOUS_CAPABILITY_ATTRIBUTION_CODE",
    "ARTIFACT_DEPENDENCY_BUNDLE_SCHEMA_VERSION",
    "DEPENDENCY_DIRECTORY_SUFFIX",
    "RELEASE_GATE_AUTHORING_CONTRACT",
    "STATIC_QUALIFICATION_STAGE_CLAIM_SCHEMA_VERSION",
    "UNREADABLE_SEALED_EVIDENCE_CODE",
    "UNVERIFIABLE_DEPENDENCY_CLOSURE_CODE",
    "USDZ_PACKAGE_MAGIC",
    "FilesystemStaticQualificationEvidenceAdapter",
    "StaticStageBindingV1",
    "StaticStageClaimV1",
    "canonical_stage_claim_json",
    "static_stage_bindings_from_run_plan",
]
