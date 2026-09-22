"""Deterministic, explicit-file capstone publication staging. No network or launches."""

from __future__ import annotations
import argparse, hashlib, json, re, shutil, subprocess
from pathlib import Path

PRIVATE_KEY = re.compile(
    r"(?i)^(?:.*(?:session|thread|uuid|endpoint|base_url|service_url).*|api_?key|access_?token|refresh_?token|client_?secret|authorization|credentials?|prompt|system_prompt|reasoning|analysis|rationale|thoughts|rollout|raw_output|raw_response|_workflow_command|.*_prompt|reasoning_summary)$"
)
# Mechanical endpoint fields in reviewed typed USD/articulation receipts are not
# network endpoints. Keep this small semantic allowlist; service fields stay private.
PUBLIC_MECHANICAL_KEYS = frozenset(
    {
        "endpoint_state",
        "old_static_endpoint",
        "new_static_endpoint",
        "moving_endpoint",
        "endpoint_protocol",
        "applied_joint_endpoint_owner_promotions",
        "joint_endpoint_owner_promotions",
        "identical_identity_endpoint_world_frames",
    }
)


def private_key(key):
    return key not in PUBLIC_MECHANICAL_KEYS and bool(PRIVATE_KEY.fullmatch(key))


def public_boolean_flag(key, value):
    return key == "base_url_configured" and type(value) is bool


FINAL_ASSESSMENT_SCHEMA = "content-agent-workflows.validation-coordinator-assessment.v1"


PRIVATE_VALUE = re.compile(
    r"(?i)(?:/Users/[A-Za-z0-9._-]+|/home/(?!horde(?:/|\b))[A-Za-z0-9._-]+|(?:[A-Za-z0-9-]+\.)*teleport\.[A-Za-z0-9.-]+|[\w.-]+\.horde[\w.-]*\.nvidia\.com|\b[\w]+-(?:astra-value-pair|codex-2gpu)\b|\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b|\bworkflow-[0-9a-f]{12,}\b|Bearer\s+[A-Za-z0-9._-]{12,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)"
)
BANNED_NAME = re.compile(
    r"(?i)(?:tool_calls\.private|rollout|(?:^|/)auth(?:\.|/)|\.codex|telemetry|daemon|core(?:\.|$)|events\.jsonl|(?:\.tgz|\.tar(?:\.gz)?|\.bundle|\.pyc)$)"
)

COMMON = "schema_version version scope status success validation_status sim_ready_status handoff_ready source_fidelity_tier target_runtime validation_tier workflow failures warnings unresolved_issues issue_codes error error_type native_checks_passed geometry_preserved observed_utc".split()
ARTIFACT = dict.fromkeys("kind path sha256 size_bytes description".split(), True)
CHECK = dict.fromkeys("name status summary failures warnings".split(), True)
CHECK["evidence_artifacts"] = [ARTIFACT]
CHECK["metadata"] = dict.fromkeys(
    "claim_scope final_domain_claim profile_intent raw_status required_by_profile required assessment_status issues_found_count issues_fixed_count engine duration_s sample_fps".split(),
    True,
)
VALIDATION = dict.fromkeys(COMMON, True) | {
    "asset": True,
    "checks": [CHECK],
    "evidence_artifacts": [ARTIFACT],
    "metadata": dict.fromkeys(
        "asset_sha256 engine duration_s sample_fps settle_distance trajectory_summary phase_timings_seconds ground_clearance_support_decision".split(),
        True,
    ),
}
RUNTIME = dict.fromkeys(
    "engine failures warnings diagnostics max_abs_position settle_distance phase_timings_seconds acceptance summary physics_usd scene_usd trajectory_jsonl recording_usda response_path metrics simulation_facts scenario n_bodies".split(),
    True,
)
RUNTIME["scene_info"] = dict.fromkeys(
    "body_prim_path body_pattern placement_prim_path placement_mode rest_position world_up meters_per_unit bbox_min_local_stage bbox_max_local_stage bbox_size_m gravity_magnitude_m_per_s2 drop_height_m_resolved ground_clearance_geometry ground_clearance_support_decision camera_paths".split(),
    True,
)
GEOMETRY_RESULT = dict.fromkeys(
    COMMON
    + "geometry_usd_path prepared_source_path source_usd_path source_bundle_id source_bundle_manifest_path route source_revision repair_outcome representation_artifacts".split(),
    True,
)
IDENTITY = dict.fromkeys(
    "schema_version source source_path source_sha256 output_asset output_evidence identity_digest source_dependency_bundle_sha256 source_mutated canonical_graph_digest asset_path asset_sha256 source_asset output_asset_sha256 status terminal_disposition issue_codes error source_identity_digest".split(),
    True,
)
TERMINAL = IDENTITY | dict.fromkeys(
    "authoring_receipt canonical_graph decision_ledger decision_patch identity preparation readback post_review cleanup".split(),
    True,
)
ASSESSMENT = dict.fromkeys(
    "schema_version status checked_views runtime_report rendered_frames issues_found issues_fixed unresolved_issues".split(),
    True,
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def encode(value):
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


class Builder:
    def __init__(self, capstone, output, repo):
        self.capstone, self.experiment, self.output, self.repo = (
            capstone,
            capstone.parent,
            output,
            repo,
        )
        self.records = []
        self.selected = set()
        self.references = []

    def locator(self, value):
        for prefix, replacement in [
            (str(self.capstone) + "/", ""),
            (str(self.experiment / "assets/01_drawer/source") + "/", "source/"),
            ("/opt/astra-content-value-20260921/capstone/", ""),
            ("/opt/astra-content-value-20260921/assets/01_drawer/source/", "source/"),
        ]:
            value = value.replace(prefix, replacement)
        value = re.sub(
            r'/opt/astra-content-value-20260921/(?:capstone/)?(?:repo|qualification)/[^\s"\']+',
            "[retained code path]",
            value,
        )
        return PRIVATE_VALUE.sub("[private identifier removed]", value)

    def clean(self, value):
        if isinstance(value, dict):
            return {
                self.locator(str(k)): self.clean(v)
                for k, v in value.items()
                if not private_key(str(k)) or public_boolean_flag(str(k), v)
            }
        if isinstance(value, list):
            return [self.clean(v) for v in value]
        if isinstance(value, str):
            return self.locator(value)
        return value

    def project(self, value, selector):
        if selector is True:
            return self.clean(value)
        if isinstance(selector, list):
            return (
                [self.project(v, selector[0]) for v in value]
                if isinstance(value, list)
                else []
            )
        if not isinstance(value, dict):
            return None
        return {
            k: self.project(value[k], sub) for k, sub in selector.items() if k in value
        }

    def write(self, destination, data, origin=None, mode="generated", extra=None):
        assert destination not in self.selected, destination
        assert (
            not Path(destination).is_absolute() and ".." not in Path(destination).parts
        )
        assert not BANNED_NAME.search(destination), destination
        path = self.output / destination
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        record = {
            "path": destination,
            "published_sha256": sha(data),
            "published_bytes": len(data),
            "mode": mode,
        }
        if origin:
            raw = origin.read_bytes()
            record |= {
                "original_retained_path": (
                    str(origin.relative_to(self.experiment))
                    if origin.is_relative_to(self.experiment)
                    else "[qualified repository code]"
                ),
                "original_retained_sha256": sha(raw),
                "original_retained_bytes": len(raw),
            }
        if extra:
            record.update(extra)
        self.records.append(record)
        self.selected.add(destination)

    def exact(self, origin, destination=None):
        destination = destination or str(origin.relative_to(self.capstone))
        data = origin.read_bytes()
        if origin.suffix.lower() in {
            ".json",
            ".jsonl",
            ".md",
            ".txt",
            ".py",
            ".usda",
            ".gltf",
        }:
            assert not PRIVATE_VALUE.search(
                data.decode("utf-8")
            ), "Exact file requires projection: " + str(origin)
        self.write(destination, data, origin, "exact")

    def projected(self, relative, selector, destination=None):
        origin = self.capstone / relative
        if not origin.is_file():
            raise FileNotFoundError(origin)
        payload = self.project(read(origin), selector)
        payload["_publication"] = {
            "mode": "field_projection",
            "original_retained_sha256": sha(origin.read_bytes()),
            "embedded_sha256_semantics": "References retain ORIGINAL retained-byte digests. Use publication_manifest.json to distinguish exact published targets from projections; this is not an original native receipt.",
        }
        self.write(destination or relative, encode(payload), origin, "field_projection")

    def reviewed_json(self, relative, *, typed_final_assessment=False):
        """For explicitly selected structured receipts only: exact if privacy-clean."""
        origin = self.capstone / relative
        value = read(origin)
        allowed_paths = set()
        if typed_final_assessment:
            if value.get("schema_version") != FINAL_ASSESSMENT_SCHEMA:
                raise ValueError("Unknown typed final-assessment schema")
            for group in ["gates", "findings"]:
                for i, item in enumerate(value.get(group, [])):
                    if isinstance(item.get("rationale"), str):
                        allowed_paths.add(f"/{group}/{i}/rationale")

        def has_private_key(node, pointer=""):
            if isinstance(node, dict):
                return any(
                    (
                        private_key(str(k))
                        and not public_boolean_flag(str(k), v)
                        and pointer + "/" + str(k) not in allowed_paths
                        and v not in (None, "", [], {})
                    )
                    or has_private_key(v, pointer + "/" + str(k))
                    for k, v in node.items()
                )
            if isinstance(node, list):
                return any(
                    has_private_key(v, pointer + f"/{i}") for i, v in enumerate(node)
                )
            return False

        if has_private_key(value) or PRIVATE_VALUE.search(origin.read_text()):
            self.projected(relative, True)
        else:
            self.exact(origin)
            if allowed_paths:
                self.records[-1]["public_typed_final_assessment_rationale_paths"] = (
                    sorted(allowed_paths)
                )
                self.records[-1][
                    "privacy_semantics"
                ] = "These schema-required rationale strings are manually reviewed final evidence-based assessment explanations intended for readers, not raw model reasoning. No other private fields are exempted."

    def optional_projected(self, relative, selector):
        if (self.capstone / relative).is_file():
            self.projected(relative, selector)


def build(capstone, output, repo):
    if output.exists():
        if (
            not (output / "publication_manifest.json").is_file()
            and not (output / ".build_in_progress").is_file()
        ):
            raise ValueError("Refusing to replace an unrecognized output directory.")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / ".build_in_progress").write_text(
        "Owned deterministic builder partial output; not publishable.\n"
    )
    b = Builder(capstone, output, repo)
    # The five original files form the complete CC0 glTF source dependency closure.
    source = b.experiment / "assets/01_drawer/source"
    for relative in [
        "drawer_cabinet_1k.gltf",
        "drawer_cabinet.bin",
        "textures/drawer_cabinet_diff_1k.jpg",
        "textures/drawer_cabinet_arm_1k.jpg",
        "textures/drawer_cabinet_nor_gl_1k.jpg",
    ]:
        b.exact(source / relative, "source/" + relative)
    for rel in ["manifest.json", "evidence/license_provenance.json"]:
        p = b.experiment / "assets/01_drawer" / rel
        payload = b.clean(read(p))
        payload["_publication"] = {
            "mode": "field_projection",
            "original_retained_sha256": sha(p.read_bytes()),
        }
        b.write("source/" + Path(rel).name, encode(payload), p, "field_projection")
    # Completed Geometry04: source output and dependency textures, original source proof.
    geo = "runs/drawer_geometry_04/"
    b.projected(geo + "geometry_workflow_result.json", GEOMETRY_RESULT)
    b.projected(geo + "geometry_validation_evidence.json", VALIDATION)
    for name in [
        "source_preserved.geometry.usdc",
        "source_prep/source_preserved.usdc",
        "source_prep/textures/image_0000.jpg",
        "source_prep/textures/image_0001.jpg",
        "source_prep/textures/image_0002.jpg",
        "render_evidence/geometry_six_view.png",
    ]:
        b.exact(capstone / geo / name)
    for name in [
        "geometry_audit_report.json",
        "asset_audit_summary.json",
        "usd_validation_report.json",
        "render_audit_report.json",
        "source_prep/lossless_gltf_receipt.json",
    ]:
        b.projected(
            geo + name,
            dict.fromkeys(
                COMMON
                + "checks findings counts metrics source output input source_sha256 output_sha256 mesh_count triangle_count topology geometry_preservation passed signals codes signal_count blocking_signal_count authoring_contract metadata".split(),
                True,
            ),
        )
    b.projected(
        geo + "render_evidence/geometry_render_evidence_six_view.json",
        dict.fromkeys(
            "schema_version status backend requested_backend renderer renderer_identity_verified preset source_usd_path source_usd_sha256 source_usd_sha256_after_render ovrtx_render_mode ovrtx_num_sensor_updates active_aov image_paths image_bindings presentation_image_path shared_render_status shared_render_issues image_validation".split(),
            True,
        ),
    )
    for p in sorted((capstone / geo / "render_evidence/renders").glob("*/*.png")):
        b.exact(p)
    # Joint02 native accepted terminal/readback, preserving the original packaged USD.
    joint = "runs/drawer_joint_02/"
    b.exact(capstone / joint / "joint_rigger/rigged.usdz")
    b.projected(joint + "standalone_articulation_terminal_receipt.json", TERMINAL)
    for name in [
        "standalone_articulation_identity.json",
        "standalone_articulation_authoring_receipt.json",
        "standalone_articulation_readback.json",
        "standalone_articulation_output_evidence.json",
        "standalone_articulation_post_review.json",
        "standalone_articulation_cleanup_receipt.json",
    ]:
        b.projected(
            joint + name,
            TERMINAL
            | dict.fromkeys(
                "checks observations articulation_graph bodies joints verdict accepted issues warnings failures source_snapshot output_snapshot source_asset_path output_asset_path actual_joint_count exact_co_rigid_disposition_match exact_membership_match exact_topology_match groups joint_ids ledger_sha256 memberships rigid_link_operations topology validation".split(),
                True,
            ),
        )
    for name in [
        "articulation_decision_patch.json",
        "canonical_articulation_graph.json",
        "validation_evidence.json",
    ]:
        b.projected(
            joint + name,
            VALIDATION
            | dict.fromkeys(
                "source_identity_digest identity_digest source_graph_digest candidate_id root_prim_path articulation_root_path bodies joints decisions components base_body root_body graph_digest checks source_asset output_asset authoritative_owner_prims candidate_ids graph_id groups memberships required_fact_blockers source_dependency_bundle_sha256 source_member_prims source_sha256".split(),
                True,
            ),
        )
    b.exact(capstone / "joint02_canonical.png", "evidence/joint02_canonical.png")
    # Completed Joint03: original native receipts are exact when their reviewed
    # fields are safe. Private metadata/decision reasoning remains projected.
    joint03 = "runs/drawer_joint_03/"
    terminal03 = read(
        capstone / joint03 / "standalone_articulation_terminal_receipt.json"
    )
    assert terminal03["terminal_disposition"] == "accept"
    assert terminal03["status"] == "completed"
    assert terminal03["source_mutated"] is False
    for name in [
        "standalone_articulation_terminal_receipt.json",
        "standalone_articulation_identity.json",
        "standalone_articulation_authoring_receipt.json",
        "standalone_articulation_readback.json",
        "standalone_articulation_output_evidence.json",
        "standalone_articulation_post_review.json",
        "standalone_articulation_cleanup_receipt.json",
        "standalone_articulation_preparation.json",
        "articulation_decision_ledger.json",
        "canonical_articulation_graph.json",
        "approved_articulation_candidates.json",
        "articulation_decision_patch.json",
        "authoring_request.json",
        "authoring_result.json",
        "validation_evidence.json",
        "final_summary.json",
        "outer_post_review_patch.json",
        "joint_rigger/authoring_attempt.json",
        "joint_rigger/diagnostics.json",
        "joint_rigger/result.json",
    ]:
        b.reviewed_json(joint03 + name)
    b.exact(capstone / joint03 / "joint_rigger/rigged.usdz")
    for directory, names in {
        "articulation_preparation_03": [
            "articulation_preparation_publication.json",
            "embedded_articulation_preparation.json",
        ],
        "drawer_packaging_04": [
            "articulation_preparation_readback.json",
            "inspector_config.json",
            "inspection_snapshot.json",
            "packaged_render.json",
            "packaging_fidelity.json",
            "saved_geometry_observation.json",
            "save_package.json",
        ],
        "drawer_topology_03": [
            "applied_topology.json",
            "execution.json",
            "physics_topology_plan.json",
            "source_topology.json",
            "prepared_source/prepared_topology_report.json",
        ],
        "drawer_packaging_05": [
            "inspection_snapshot.json",
            "packaged_render.json",
            "packaging_fidelity.json",
            "save_package.json",
        ],
        "drawer_joint_03_visual": [
            "canonical_visual_payload.json",
            "canonical_visual_render_report.json",
            "canonical_visual_request.json",
            "verified_operation_envelope.json",
            "verified_operation_projection.json",
            "usd_cli/renders/view-000_camera.json",
            "usd_cli/renders/view-000_response.json",
        ],
    }.items():
        for name in names:
            b.reviewed_json("runs/" + directory + "/" + name)
    for name in [
        "runs/drawer_packaging_04/source.usdz",
        "runs/drawer_packaging_04/packaged_preview.png",
        "runs/drawer_packaging_04/_render_stage_392456_4fee1d07.usdc",
        "runs/drawer_packaging_04/prepare_static_mesh_articulation_readback.py",
        "runs/drawer_topology_03/prepared_source/prepared.usda",
        "runs/drawer_topology_03/prepared_source/source.usdz.dscsave392456-f86b5496.usdz.stage.usdc",
        "runs/drawer_topology_03/prepared_source/0/image_0000.jpg",
        "runs/drawer_topology_03/prepared_source/0/image_0001.jpg",
        "runs/drawer_topology_03/prepared_source/0/image_0002.jpg",
        "runs/drawer_packaging_05/source.usdz",
        "runs/drawer_packaging_05/packaged_preview.png",
        "runs/drawer_packaging_05/_render_stage_417769_0b938142.usdc",
        "runs/drawer_joint_03_visual/usd_cli/renders/view-000.png",
        "runs/drawer_joint_03_visual/usd_cli/renders/_render_stage_410510_4339c622.usdc",
        "evidence/joint03_canonical_view.png",
        "publish_joint03_preparation.py",
        "publish_joint03_preparation_retry01.py",
    ]:
        b.exact(capstone / name)
    # Keep the original helper digest, replacing only its private GPU selector
    # with an explicitly caller-supplied environment value. This is a derivative.
    packaging_helper = capstone / "package_for_joint_03.py"
    packaging_code, replaced = re.subn(
        r"CUDA_VISIBLE_DEVICES='GPU-[0-9a-f-]+'",
        "CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES']",
        packaging_helper.read_text(),
    )
    assert replaced == 1 and not PRIVATE_VALUE.search(packaging_code)
    compile(packaging_code, "package_for_joint_03.py", "exec")
    b.write(
        "package_for_joint_03.py",
        packaging_code.encode(),
        packaging_helper,
        "executable_environment_projection",
        extra={
            "change": "Only hardcoded private GPU UUID replaced with caller-supplied CUDA_VISIBLE_DEVICES; original helper retained separately. Not executed by publication builder.",
        },
    )
    b.exact(capstone / "prepare_static_mesh_articulation_readback.py")
    for name in [
        "evidence/joint03_authored_readback_before_visual.json",
        "evidence/joint03_bound_output_evidence.json",
        "evidence/joint03_export_verification.json",
    ]:
        b.reviewed_json(name)
    # Accepted articulation is not measured Physics/task acceptance. No active10
    # output is read by this completed-stage allowlist.
    # Honest physical failures and complete measured state traces;08 remains unresolved.
    for case in ["03", "06", "07", "08", "09"]:
        prefix = "runs/drawer_physics_" + case + "/"
        b.projected(prefix + "validation_evidence.json", VALIDATION)
        if case != "06":
            b.projected(prefix + "physics_behavior_assessment.json", ASSESSMENT)
            b.projected(
                prefix + "physics_assignments.json",
                dict.fromkeys(
                    "schema_version source_usd source_sha256 prepared_usd prepared_sha256 source_asset_sha256 prepared_asset_sha256 authored_usd physics_usd assignments decisions components physics_config validation_evidence_path asset prepared_asset candidate_count component_count decision_count mobility_intent path_space unresolved_components validation_status physics_behavior_assessment simulation_report validation_evidence visual_validation_frames vomp_mass".split(),
                    True,
                ),
            )
            b.projected(
                prefix + "raw/physics_decision_patch.json",
                dict.fromkeys(
                    "schema_version source_identity_digest decisions assignments components asset source_digest unresolved_components".split(),
                    True,
                ),
            )
            b.exact(capstone / prefix / "physics.usda")
            b.exact(capstone / prefix / "inputs/source/source.usdz")
        for name in [
            "runtime/runtime_validation_report.json",
            "runtime/usd_cli_simulation/runtime_validation_report.json",
        ]:
            b.projected(prefix + name, RUNTIME)
        b.projected(
            prefix + "runtime/simulation_response.json",
            dict.fromkeys(
                "engine status n_bodies n_steps trajectory_sample_count usd_cli_report_path".split(),
                True,
            ),
        )
        for name in [
            "runtime/drop_settle_scene.usda",
            "runtime/usd_cli_simulation/recording.usda",
            "runtime/usd_cli_simulation/trajectory.jsonl",
        ]:
            b.exact(capstone / prefix / name)
        for p in sorted(
            (capstone / prefix / "runtime/drop_settle_scene.usda_assets").glob(
                "*/*.jpg"
            )
        ):
            b.exact(p)
        if case in {"07", "08", "09"}:
            for iteration in ([1, 2] if case in {"08", "09"} else [1]):
                for p in sorted(
                    (
                        capstone / prefix / f"runtime/visual_review_frames_{iteration}"
                    ).glob("review_frame_*.png")
                ):
                    b.exact(p)
                b.projected(
                    prefix + f"raw/physics_render_frame_receipt_{iteration}.json",
                    dict.fromkeys(
                        "schema_version iteration simulation_report render_inputs render_input_bundle runtime_asset_provenance render_request render_response response_artifact frames".split(),
                        True,
                    ),
                )
        if case in {"08", "09"}:
            # Review1 predates the final runtime report at the same reused path.
            # Preserve its actual digest without claiming those superseded bytes exist.
            first = next(
                r
                for r in b.records
                if r["path"] == prefix + "raw/physics_render_frame_receipt_1.json"
            )
            original_ref = read(capstone / first["path"])["simulation_report"]
            assert (
                original_ref["sha256"]
                == {
                    "08": "8c6c3310e49e1a3da75c81bc30d6018857f1f0e5db738fd48dfdf39fab3019c5",
                    "09": "dc39bc80da769d3f8d7e58e30f90b2539d5991e25582888809cb49b648228218",
                }[case]
            )
            first["unprovided_intermediate_references"] = [
                {
                    "path": prefix + "runtime/runtime_validation_report.json",
                    "sha256": original_ref["sha256"],
                    "bytes_provided": False,
                    "scope": "Native review1 receipt attests this intermediate digest; final review2 superseded the file at the same path. No exact-byte or runtime-closure verification is claimed for this intermediate report.",
                }
            ]
            b.projected(
                prefix + "workflow_run_manifest.json",
                dict.fromkeys(
                    "schema_version workflow status failure source_sha256 source_path created_at updated_at".split(),
                    True,
                ),
            )
    # Actual completed native10 pass: exact safe producer evidence, and labeled
    # projections only where runtime identity or private decision fields occur.
    native10 = "runs/drawer_physics_10/"
    native10_asset = capstone / native10 / "physics.usda"
    assert (
        sha(native10_asset.read_bytes())
        == "0ef7038845d569a2f502f38af9fb5b1e4e99cf06c5f4e04035e4c259b1b07156"
    )
    native10_manifest = read(capstone / native10 / "workflow_run_manifest.json")
    native10_validation = read(capstone / native10 / "validation_evidence.json")
    native10_assessment = read(capstone / native10 / "physics_behavior_assessment.json")
    assert native10_manifest["status"] == "pass"
    assert (
        native10_validation["sim_ready_status"]
        == native10_assessment["status"]
        == "pass"
    )
    assert len(native10_validation["checks"]) == 4
    assert all(x["status"] == "pass" for x in native10_validation["checks"])
    assert not native10_assessment["unresolved_issues"]
    for name in [
        "physics_behavior_assessment.json",
        "validation_evidence.json",
        "workflow_run_manifest.json",
        "physics_assignments.json",
        "runtime/runtime_validation_report.json",
        "runtime/usd_cli_simulation/runtime_validation_report.json",
        "runtime/simulation_response.json",
        "raw/staged_input_source.json",
        "raw/physics_finalize_result_1.json",
        "raw/physics_render_frame_receipt_1.json",
        "raw/physics_decision_patch.json",
    ]:
        b.reviewed_json(native10 + name)
    for name in [
        "physics.usda",
        "inputs/source/source.usdz",
        "runtime/drop_settle_scene.usda",
        "runtime/usd_cli_simulation/recording.usda",
        "runtime/usd_cli_simulation/trajectory.jsonl",
    ]:
        b.exact(capstone / native10 / name)
    for p in sorted(
        (capstone / native10 / "runtime/drop_settle_scene.usda_assets").glob("*/*.jpg")
    ):
        b.exact(p)
    for i in range(8):
        base = native10 + f"runtime/visual_review_frames_1/review_frame_{i:04d}"
        b.exact(capstone / (base + ".png"))
        b.reviewed_json(base + ".camera.json")
    for name in [
        "evidence/original_source_drawer_physics_10.json",
        "evidence/authored_contract_drawer_physics_10.json",
        "evidence/native10_cooked_clearance_v1.json",
        "evidence/native10_critical_cooking_log_audit.json",
        "evidence/native10_cooking_log_review.json",
        "evidence/native10_postflight_execution.json",
        "evidence/completed_native_leaf_accounting.json",
        "evidence/physics_native_execution_10.json",
        "evidence/native10_pre_visual_runtime/capture.json",
        "evidence/native10_pre_visual_runtime/runtime_validation_report.json",
        "evidence/native10_pre_visual_runtime/validation_evidence.json",
        "evidence/native10_pre_visual_runtime/physics_decision_patch.json",
        "evidence/native10_task_bindings.json",
        "evidence/native10_supplement_export_manifest.json",
    ]:
        b.reviewed_json(name)
    for name in [
        "evidence/native10_cooked_clearance_v1.log",
        "evidence/native10_source_audit.log",
        "evidence/native10_numeric_audit.log",
        "evidence/native10_cooking_log_audit.log",
        "audit_cooking_log.py",
        "audit_native_drawer_contract_v5.py",
        "probe_cooked_drawer_clearance.py",
        "review_native10_cooking_log.py",
        "verify_original_drawer_geometry.py",
    ]:
        b.exact(capstone / name)
    # Qualified read-only audit + original unsuccessful attempts are explicit
    # artifacts bound by its fixed qualification; no arbitrary directory glob.
    task_audit_root = capstone / "task_evidence_audit_v1"
    task_audit_qualification = read(task_audit_root / "qualification.json")
    assert (
        sha((task_audit_root / "qualification.json").read_bytes())
        == "0f2b6651efd37386b6a52b782dc6a9bd6412543c7fcec1dc9916394b1fbc8002"
    )
    assert task_audit_qualification["tests"]["passed"] == 20
    assert (
        task_audit_qualification["full_path_negative_task"]["reported_task_status"]
        == "FAIL"
    )
    b.exact(task_audit_root / "qualification.json")
    b.records[-1]["reference_base"] = "task_evidence_audit_v1"
    for row in task_audit_qualification["artifact_bindings"]:
        rel = Path(row["path"])
        assert not rel.is_absolute() and ".." not in rel.parts
        origin = task_audit_root / rel
        assert sha(origin.read_bytes()) == row["sha256"]
        if origin.suffix == ".json":
            b.reviewed_json(str(origin.relative_to(capstone)))
        else:
            b.exact(origin)
    # Two small retained fixtures are required by three of the audit's20 tests.
    for name in ["scene.usda", "request.json"]:
        p = capstone / "evaluations/native09_source_clear_v1/seed_11" / name
        if p.suffix == ".json":
            b.reviewed_json(str(p.relative_to(capstone)))
        else:
            b.exact(p)
    # Independent native review and completed trace audit are original safe
    # receipts. Their scope remains measured consistency, not fabricated acceptance.
    native10_audit = read(capstone / "task_evidence_audit_v1/native10/audit.json")
    assert native10_audit["audit_consistent"] is True
    assert native10_audit["reported_task_pass"] is True
    assert native10_audit["task_acceptance_attested"] is False
    for name in [
        "evidence/native10_independent_review/review.json",
        "evidence/native10_independent_review/provenance_note.json",
        "evidence/native10_independent_review/native_producer_bundle.json",
        "task_evidence_audit_v1/native10/audit.json",
        "evidence/native10_task_export_manifest.json",
        "evidence/native10_outcome_before_validation.json",
    ]:
        b.reviewed_json(name)
    for name in [
        "evidence/native10_independent_review/review.py",
        "evidence/native10_independent_review/execution.log",
    ]:
        b.exact(capstone / name)
    # Actual completed native Validation. These are explicit terminal/operation
    # receipts only: no native private logs, prompts, or child-agent transcripts.
    validation = "runs/drawer_validation_01_prepared/"
    native_validation = validation + "native_run/"
    final_conjunction = read(capstone / "evidence/final_capstone_conjunction.json")
    assert (
        sha((capstone / "evidence/final_capstone_conjunction.json").read_bytes())
        == "44fc44580c58ca68a489cdf9f818508b5704fe6052223399cdb42b821c92fa0c"
    )
    assert final_conjunction["final_bounded_capstone_conjunction"] is True
    for field in [
        "final_asset",
        "native_terminal",
        "task_report",
        "independent_task_audit",
        "geometry_evidence",
    ]:
        binding = final_conjunction[field]
        assert sha((capstone / binding["path"]).read_bytes()) == binding["sha256"]
    for binding in final_conjunction["verified_native_terminal_bindings"]:
        assert sha((capstone / binding["path"]).read_bytes()) == binding["sha256"]
    terminal_validation = read(
        capstone / native_validation / "validation_terminal_receipt.json"
    )
    assert terminal_validation["receipt_status"] == "completed"
    assert terminal_validation["review_disposition"] == "accept"
    assert terminal_validation["terminal_disposition"] == "pass"
    assert terminal_validation["source_mutated"] is False
    assert terminal_validation["gate_dispositions"] == {
        "cross_stage_integrity": "not_evaluated",
        "package_integrity": "pass",
        "runtime_validation": "pass",
        "static_validation": "pass",
        "visual_quality": "pass",
    }
    assert final_conjunction["final_asset"]["sha256"] == sha(
        native10_asset.read_bytes()
    )
    for name in [
        "collect.execution.json",
        "assess.execution.json",
        "review.execution.json",
        "run.execution.json",
        "independent_review.json",
        "native_physics_bundle.json",
        "commands.json",
        "review.schema.json",
        "preparation_receipt.json",
        "policy.json",
    ]:
        b.reviewed_json(validation + name)
    b.exact(capstone / validation / "assessment.schema.json")
    b.records[-1]["public_json_schema_definition"] = True
    b.reviewed_json(validation + "outer_assessment.json", typed_final_assessment=True)
    b.reviewed_json(
        native_validation + "canonical_validation_assessment.json",
        typed_final_assessment=True,
    )
    for name in [
        "validation_terminal_receipt.json",
        "validation_operation_index.json",
        "standalone_validation_execution.json",
        "validation_result.json",
        "final_summary.json",
        "validation_coordinator_plan_patch.json",
        "validation_operation_preparation.json",
        "validation_plan.json",
        "validation_coordinator_accepted_plan.json",
        "validation_evidence.json",
        "validation_checkpoint.json",
        "validation_coordinator_execution_receipt.json",
        "validation_request.json",
        "standalone_validation_evidence.json",
        "validation_coordinator_preparation.json",
    ]:
        b.reviewed_json(native_validation + name)
    for operation in ["physical_behavior", "physics_sane", "render_valid"]:
        for name in ["operation_result.json", "template_result.json"]:
            b.reviewed_json(native_validation + "operations/" + operation + "/" + name)
    b.exact(
        capstone
        / native_validation
        / "operations/render_valid/renders/000_physics_c491b4e7/000_physics_c491b4e7_plus_xplus_yplus_z_0000.png"
    )
    for name in [
        "evidence/final_capstone_conjunction.json",
        "evidence/final_conjunction_independent_review.json",
        "evidence/validation01_terminal_export.json",
        "evidence/validation01_leaf_accounting.json",
    ]:
        b.reviewed_json(name)
    # Explicit independently produced source/failure diagnostics.
    for name in [
        "original_source_drawer_geometry_04.json",
        "joint02_original_geometry_and_joint_readback.json",
        "original_source_drawer_physics_03.json",
        "original_source_drawer_physics_07.json",
        "original_floor_probe.json",
        "mounted06_independent_review.json",
        "physics07_controlled_decision_readback.json",
        "physics07_launcher_bookkeeping.json",
        "native03_cooking_probe.json",
    ]:
        p = capstone / "evidence" / name
        if p.exists():
            b.projected(
                "evidence/" + name,
                dict.fromkeys(
                    "schema_version scope status observed_utc passed source_gltf_sha256 source_bin_sha256 source_mesh_count output_mesh_count meters_per_unit up_axis parts candidate candidate_sha256 checks measurements observed expected conclusion limitations changes controlled_comparison claims physics_usd_sha256 joint_readback joints body_paths source_floor_y_m cooked_floor_y_m first_step_displacement_m source output output_sha256 source_fidelity joint_path joint_type axis limits_m articulation_roots physics_task_accepted started_utc completed_utc asset sha256 backend initial_pose initial_velocity tensor_types floor_rays interior samples asset_unchanged recorded_utc issue recovery limitation physics03_archive_sha256 retained_original03_launch_sha256 retained_original03_execution_sha256 corrected07_launcher_sha256 completion_recovery_pending recovery_completed_utc recovery_hashes_verified".split(),
                    True,
                ),
            )
    # Exact sanitized independent read-only scans and the runtime option overclaim correction.
    for name in [
        "publication_review/usd_privacy_review.json",
        "publication_review/usd_privacy_review_08.json",
        "publication_review/usd_privacy_review_09.json",
        "publication_review/usd_privacy_review_joint03.json",
        "publication_review/usd_privacy_review_10.json",
        "publication_review/usd_privacy_review_native10_replay.json",
        "publication_review/task_audit_portable_tests.json",
        "publication_review/task_audit_portable_tests.log",
        "publication_review/original_arrays_joint03.json",
        "publication_review/original_arrays_joint03_package05.json",
        "evidence/native08_independent_preflight_review.json",
    ]:
        b.exact(capstone / name)
    # Read-only portability qualification uses a separate derivative namespace.
    for name in [
        "portable_public_bundle_r2_decoded_review_plugin.json",
        "portable_public_bundle_r2_portable_closure.json",
        "portable_relocation_manifest.json",
        "portable_public_rebuild_review.json",
        "public_bundle_r2_source_geometry_04_source_preserved_geometry_usdc.json",
        "public_bundle_r2_source_joint_02_joint_rigger_rigged_usdz.json",
        "public_bundle_r2_source_physics_03_physics_usda.json",
        "public_bundle_r2_source_physics_07_physics_usda.json",
        "public_bundle_r2_source_physics_08_physics_usda.json",
    ]:
        b.exact(capstone / "publication_review" / name)
        if name == "portable_public_bundle_r2_portable_closure.json":
            b.records[-1]["reference_namespace"] = "portable_derivative_not_in_bundle"
    # Preserve the full unchanged frozen evaluator/solver contract plus exact qualification.
    fixture = capstone / "evaluator/source_clear_v1"
    freeze = read(fixture / "source_clear_freeze.json")
    for name, expected in sorted(freeze["files_sha256"].items()):
        p = fixture / name
        assert sha(p.read_bytes()) == expected, (name, "frozen hash mismatch")
        if name == "original_drawer_freeze.json":
            b.projected(
                str(p.relative_to(capstone)),
                dict.fromkeys(
                    "protocol_id frozen_utc status files evidence synthetic_test_count synthetic_tests_pass native_convex_mesh_contact_witness_pass force_units_and_gravity_witness_pass source_outward_direction_verified scored_source_cabinet_runs solver claim_limit pre_score_hardening additional_source_parenting_tests source_parenting_tests_pass".split(),
                    True,
                ),
            )
        else:
            b.exact(p)
    for name in [
        "source_clear_freeze.json",
        "qualification.json",
        "original_drawer_freeze.json",
        "prepare_source_clear_fixture.py",
        "drawer_mesh_collision_probe.py",
        "source_direction_probe.py",
        "drawer_force_units_probe.py",
        "static_binding_test_evidence.json",
    ]:
        p = fixture / name
        if p.exists() and str(p.relative_to(capstone)) not in b.selected:
            b.exact(p)
    for name in [
        "test_report.json",
        "run_good/trace.jsonl",
        "run_good/trial_report.json",
        "run_no_motion/trace.jsonl",
        "run_no_motion/trial_report.json",
        "run_filled/trace.jsonl",
        "run_filled/trial_report.json",
    ]:
        p = fixture / "selftest" / name
        if p.exists():
            if p.suffix == ".json":
                b.projected(
                    str(p.relative_to(capstone)),
                    dict.fromkeys(
                        "version seed status passed accepted checks measurements metrics result errors failures step_count steps completed n_steps runtime engine timeout timed_out tests".split(),
                        True,
                    ),
                )
            else:
                b.exact(p)
    b.exact(
        capstone / "fixture_build_input/evaluator/drawer_acceptance.json",
        "evaluator/source_clear_v1/original_spec/drawer_acceptance.json",
    )
    b.exact(capstone / "SOURCE_CLEAR_FIXTURE.md")
    b.exact(capstone / "prepare_source_clear_fixture.py")
    # Preserve every completed task, including failures, without changing gates.
    task_reports = {}
    for case in ["08", "09", "10"]:
        task_dir = f"evaluations/native{case}_source_clear_v1"
        plots_dir = f"task_plots/native{case}_source_clear_v1"
        task_report = read(capstone / task_dir / "report.json")
        expected_pass = case == "10"
        assert task_report["status"] == ("PASS" if expected_pass else "FAIL")
        assert task_report["pass"] is expected_pass
        assert len(task_report["trials"]) == 5
        assert all(t["pass"] is expected_pass for t in task_report["trials"])
        assert task_report["inputs"]["usd_sha256"] == sha(
            (capstone / f"runs/drawer_physics_{case}/physics.usda").read_bytes()
        )
        for name in ["report.json", "structural_report.json"]:
            b.exact(capstone / task_dir / name)
        summary = read(capstone / plots_dir / "summary.json")
        assert summary["all_five_pass"] is expected_pass and len(summary["trials"]) == 5
        export = read(capstone / plots_dir / "export_verification.json")
        export_rows = export["trials"] if case == "10" else export["checks"]
        if case == "10":
            assert export["all_five_exact"] is True
            assert all(r["exact_roundtrip"] is True for r in export_rows)
        sizes = {row["seed"]: row for row in export_rows}
        assert set(sizes) == {11, 23, 47, 83, 131}
        for row in summary["trials"]:
            seed = row["seed"]
            report = capstone / task_dir / f"seed_{seed}/trial_report.json"
            assert sha(report.read_bytes()) == row["report_sha256"]
            b.exact(report)
            for suffix, key in [
                ("measurements.csv", "measurements_sha256"),
                ("trace.jsonl.gz", "compressed_trace_sha256"),
            ]:
                path = capstone / plots_dir / f"seed_{seed}_{suffix}"
                assert sha(path.read_bytes()) == row[key]
                b.exact(path)
                if suffix.endswith(".gz"):
                    b.records[-1]["compression"] = {
                        "format": "gzip",
                        "mtime": 0,
                        "uncompressed_sha256": row["trace_sha256"],
                        "uncompressed_bytes": sizes[seed]["uncompressed_bytes"],
                        "original_trace_path": task_dir + f"/seed_{seed}/trace.jsonl",
                        "scope": "Full lossless trace, no clipping; original uncompressed digest is verified after decoding.",
                    }
        for name in [
            "summary.json",
            "export_verification.json",
            "drawer_task.png",
            "drawer_task.svg",
        ]:
            b.exact(capstone / plots_dir / name)
        task_reports[case] = {
            "status": task_report["status"],
            "pass": task_report["pass"],
            "trials_passed": sum(t["pass"] for t in task_report["trials"]),
            "trials_total": 5,
            "asset_sha256": task_report["inputs"]["usd_sha256"],
            "report_sha256": sha((capstone / task_dir / "report.json").read_bytes()),
        }
    for name in ["plot_task_trials.py", "source_contact_diagnostic.py"]:
        b.exact(capstone / name)
    for name in [
        "native08_task_bindings.json",
        "native09_task_bindings.json",
        "source_contact_diagnostic_08.json",
        "authored_contract_drawer_physics_08.json",
        "original_source_drawer_physics_08.json",
    ]:
        b.exact(capstone / "evidence" / name)
    b.projected(
        "evidence/convex_options_bounds_v2/qualification.json",
        dict.fromkeys(
            "schema_version qualified_utc status commit base_commit code_files patch_sha256 correction tests native8_witness prior64_witness setup_attempts_preserved commands preserved08_asset_sha256 remote_active_checkout_mutated scored_or_frozen_files_modified native_drawer_acceptance_claim limitations artifacts".split(),
            True,
        ),
    )
    b.reviewed_json("evidence/static_endpoint_repair_protocol_v1.json")
    # Read-only independent09 and static-endpoint hypothesis/qualification.
    for name in [
        "evidence/native09_startup_review/review.json",
        "evidence/static_joint_endpoint_v1/review_receipt.json",
        "evidence/static_joint_endpoint_v1/result.json",
        "evidence/static_joint_endpoint_v1/prepared.json",
        "evidence/static_endpoint_route_review/review.json",
        "evidence/public_repo_required_checks_e640.json",
    ]:
        b.projected(name, True)
    for name in [
        "evidence/static_joint_endpoint_v1/qualify.py",
        "evidence/static_endpoint_route_review/check_route.py",
        "evidence/native09_startup_review/review.py",
        "evidence/static_joint_endpoint_v1/static_xform_collide.usda",
        "evidence/static_joint_endpoint_v1/static_mesh_disabled.usda",
        "evidence/static_joint_endpoint_v1/static_xform_disabled.usda",
        "evidence/static_joint_endpoint_v1/static_mesh_collide.usda",
        "promote_joint03_upper.py",
        "evidence/public_repo_required_tests_e640.log",
        "evidence/public_repo_skill_sync_e640.log",
    ]:
        b.exact(capstone / name)
    for name in [
        "evidence/ovphysx_0_4_13_registry_excerpt.json",
        "evidence/native09_critical_cooking_log_audit.json",
        "evidence/native09_cooking_log_review.json",
        "evidence/native09_cooked_clearance_v1.json",
        "evidence/authored_contract_drawer_physics_09.json",
        "evidence/original_source_drawer_physics_09.json",
        "evidence/native09_outcome.json",
    ]:
        b.projected(name, True)
    for name in [
        "evidence/native09_cooked_clearance_v1.log",
        "evidence/static_joint_endpoint_v1/native.log",
    ]:
        b.exact(capstone / name)
    # Presentation-only measured replay: three stills, no auto-GIF or invented motion.
    replay = "task_replay/native10_seed11_v2/"
    replay_receipt = read(capstone / replay / "presentation_receipt.json")
    assert (
        replay_receipt["input_bytes_unchanged"]
        and replay_receipt["dependency_bytes_unchanged"]
    )
    assert len(replay_receipt["pose_readback_checks"]) == 6
    assert all(
        r["position_error_m"] == 0 for r in replay_receipt["pose_readback_checks"]
    )
    assert (
        sha((capstone / "render_task_replay.py").read_bytes())
        == replay_receipt["script_sha256"]
    )
    b.exact(capstone / "render_task_replay.py")
    for step in [179, 1230, 2459]:
        name = f"frames/usd_cam__1024x1024__f{step:04d}.png"
        assert (
            sha((capstone / replay / name).read_bytes())
            == replay_receipt["outputs"][name]
        )
        b.exact(capstone / replay / name)
    b.reviewed_json(replay + "root_visual_review.json")
    for name in ["presentation_receipt.json", "probe.json", "render.json"]:
        b.projected(replay + name, True)
    for name in ["request.json", "scene.usda", "replay.usda"]:
        path = capstone / "evaluations/native10_source_clear_v1/seed_11" / name
        if path.suffix == ".json":
            b.reviewed_json(str(path.relative_to(capstone)))
        else:
            b.exact(path)
    # Current corrected launcher is hash-bound independently of its earlier review.
    launcher_qualification = read(capstone / "validation_launch/qualification.json")
    assert launcher_qualification["provider_free_tests"]["passed"] == 22
    for row in launcher_qualification["artifacts"]:
        rel = row["path"]
        assert rel.startswith("validation_launch/") and ".." not in Path(rel).parts
        origin = capstone / rel
        assert sha(origin.read_bytes()) == row["sha256"]
        if rel not in b.selected:
            if origin.suffix == ".json":
                b.reviewed_json(rel)
            else:
                b.exact(origin)
    b.reviewed_json("validation_launch/qualification.json")
    # Earlier qualification still points to the then-current code paths. Preserve
    # its exact bytes and explicitly redirect those digests to provided history.
    historical = "validation_launch/history/geometry_handoff_assumption/"
    old_qualification = read(capstone / historical / "qualification.json")
    old_record = next(
        r for r in b.records if r["path"] == historical + "qualification.json"
    )
    aliases, absent = [], []
    for item in old_qualification["artifacts"]:
        target = historical + Path(item["path"]).name
        origin = capstone / target
        if origin.is_file() and sha(origin.read_bytes()) == item["sha256"]:
            aliases.append(
                {
                    "path": item["path"],
                    "sha256": item["sha256"],
                    "published_path": target,
                }
            )
        elif (capstone / item["path"]).is_file() and sha(
            (capstone / item["path"]).read_bytes()
        ) != item["sha256"]:
            absent.append(
                {
                    "path": item["path"],
                    "sha256": item["sha256"],
                    "bytes_provided": False,
                    "scope": "Historical qualification attests superseded documentation bytes; current file differs. No exact-byte verification of those omitted historical bytes is claimed.",
                }
            )
    old_record["reference_redirects"] = aliases
    old_record["unprovided_intermediate_references"] = absent
    # Code changes are a separate patch from the public base, not rewritten run outputs.
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    patch = subprocess.check_output(
        ["git", "diff", "a96faf9cb2f5c1f655fe0d60c0ccf57e3477b1aa", commit, "--"],
        cwd=repo,
    )
    assert not PRIVATE_VALUE.search(
        patch.decode()
    ), "Code patch has private identifiers"
    b.write(
        "code/capstone_changes.patch",
        patch,
        extra={
            "base_commit": "a96faf9cb2f5c1f655fe0d60c0ccf57e3477b1aa",
            "final_commit": commit,
        },
    )
    for name in [
        "qualification.json",
        "independent_review.json",
        "retained_qualification.json",
        "integration_commit.json",
    ]:
        p = capstone / "validation_contract_review" / name
        b.projected(
            str(p.relative_to(capstone)),
            dict.fromkeys(
                "schema_version observed_utc status no_model_solver_or_renderer_execution isolated_test_baseline_commit local_physics_commit_separate tests lint diff_check scope legacy_comparison files source_bindings retained_cases all_retained_failed_evidence_rejected checks blockers limitations conclusion qualification patch commit parent_commit active_remote_checkout_updated independent_review".split(),
                True,
            ),
        )
    for name in ["tests_lint_final.log", "POLICY_EXAMPLE.md"]:
        b.exact(capstone / "validation_contract_review" / name)
    # Publication tools are portable and contain no runtime secrets.
    for name in [
        "build_public_bundle.py",
        "verify_public_bundle.py",
        "verify_source_arrays.py",
        "relocate_usd_paths.py",
        "rebuild_source_clear_fixture.py",
        "test_publication_tools.py",
    ]:
        p = capstone / "publication_tools" / name
        if p.exists():
            b.exact(p, "tools/" + name)
    outcome = {
        "scope": "Unscored native Content Agents capstone; frozen pilot unchanged.",
        "code_commit": commit,
        "stages": {
            "geometry04": {
                "success": read(capstone / geo / "geometry_workflow_result.json").get(
                    "success"
                ),
                "validation_status": read(
                    capstone / geo / "geometry_workflow_result.json"
                ).get("validation_status"),
                "physical_acceptance": False,
            },
            "joint02": {
                "terminal_disposition": read(
                    capstone / joint / "standalone_articulation_terminal_receipt.json"
                )["terminal_disposition"],
                "physical_acceptance": False,
            },
            "joint03": {
                "terminal_disposition": terminal03["terminal_disposition"],
                "source_mutated": terminal03["source_mutated"],
                "asset_sha256": sha(
                    (capstone / joint03 / "joint_rigger/rigged.usdz").read_bytes()
                ),
                "terminal_receipt_sha256": sha(
                    (
                        capstone
                        / joint03
                        / "standalone_articulation_terminal_receipt.json"
                    ).read_bytes()
                ),
                "physical_acceptance": False,
            },
            "physics10": {
                "workflow_status": native10_manifest["status"],
                "validation_status": native10_validation["sim_ready_status"],
                "behavior_assessment_status": native10_assessment["status"],
                "asset_sha256": sha(native10_asset.read_bytes()),
                "physical_acceptance": False,
                "scope": "Three-second mounted-rest native behavior; independent five-seed task and completed native Validation pass separately on the same asset.",
            },
            "independent_native10_task": {
                "status": "PASS",
                "trials_passed": 5,
                "trials_total": 5,
                "audit_consistent": native10_audit["audit_consistent"],
                "audit_sha256": sha(
                    (
                        capstone / "task_evidence_audit_v1/native10/audit.json"
                    ).read_bytes()
                ),
                "asset_sha256": sha(native10_asset.read_bytes()),
                "native_validation_requirement_satisfied_separately": True,
            },
            "physics03": {"native_status": "fail"},
            "physics06": {"native_status": "fail"},
            "physics07": {"native_status": "fail"},
            "physics08": {
                "workflow_status": read(
                    capstone / "runs/drawer_physics_08/workflow_run_manifest.json"
                )["status"],
                "validation_status": read(
                    capstone / "runs/drawer_physics_08/validation_evidence.json"
                )["sim_ready_status"],
                "behavior_assessment_status": read(
                    capstone / "runs/drawer_physics_08/physics_behavior_assessment.json"
                )["status"],
                "asset_sha256": sha(
                    (capstone / "runs/drawer_physics_08/physics.usda").read_bytes()
                ),
                "physical_acceptance": False,
                "cooking_option_limit": "Authored hullVertexLimit128 was warned invalid by the native plugin (supported8..64); authored readback does not prove effective use.",
            },
            "native_validation": {
                "status": "PASS",
                "terminal_acceptance": True,
                "source_mutated": terminal_validation["source_mutated"],
                "gate_dispositions": terminal_validation["gate_dispositions"],
                "terminal": final_conjunction["native_terminal"],
                "asset_sha256": final_conjunction["final_asset"]["sha256"],
            },
            "physics09": {
                "workflow_status": read(
                    capstone / "runs/drawer_physics_09/workflow_run_manifest.json"
                )["status"],
                "validation_status": read(
                    capstone / "runs/drawer_physics_09/validation_evidence.json"
                )["sim_ready_status"],
                "behavior_assessment_status": read(
                    capstone / "runs/drawer_physics_09/physics_behavior_assessment.json"
                )["status"],
                "asset_sha256": sha(
                    (capstone / "runs/drawer_physics_09/physics.usda").read_bytes()
                ),
                "physical_acceptance": False,
            },
            "independent_five_seed_tasks": task_reports,
            "native_validation_launcher_qualification": {
                "qualified": True,
                "qualification_itself_executes_native_workflow": False,
                "qualification_sha256": sha(
                    (capstone / "validation_launch/qualification.json").read_bytes()
                ),
            },
        },
        "final_task_success": True,
        "final_conjunction_receipt": {
            "path": "evidence/final_capstone_conjunction.json",
            "sha256": sha(
                (capstone / "evidence/final_capstone_conjunction.json").read_bytes()
            ),
        },
        "final_scope_limits": final_conjunction["limitations"],
        "required_final_conjunction": "native Validation AND five-seed source-clear task acceptance on exactly the same authored USD SHA",
        "source_clear_fixture": {
            "freeze_sha256": sha((fixture / "source_clear_freeze.json").read_bytes()),
            "qualification_sha256": sha((fixture / "qualification.json").read_bytes()),
            "asset_acceptance": False,
        },
        "evidence_semantics": "Native JSONs are exact or explicitly labeled field projections per manifest; measured trajectory/source/evaluator bytes are exact where marked. Original referenced hashes are not silently replaced with hashes of projected receipts.",
    }
    b.write("outcome.json", encode(outcome))
    b.write("README.md", README.encode())
    b.write("EVIDENCE_NOTES.md", EVIDENCE_NOTES.encode())
    b.write(".gitattributes", GITATTRIBUTES.encode())
    manifest = {
        "schema_version": "capstone-curated-publication-manifest.v1",
        "deterministic_builder": True,
        "status": "CURATED_BOUNDED_NATIVE_VALIDATION_AND_TASK_CONJUNCTION",
        "files": sorted(b.records, key=lambda r: r["path"]),
        "excluded": "No archives, credentials, raw tool/event streams, private reasoning, rollouts, daemon/session logs, runtime environments or GPU/service identities are selected.",
    }
    (output / ".build_in_progress").unlink()
    (output / "publication_manifest.json").write_bytes(encode(manifest))
    return {
        "files": len(b.records),
        "bytes": sum(r["published_bytes"] for r in b.records),
        "manifest_sha256": sha((output / "publication_manifest.json").read_bytes()),
    }


GITATTRIBUTES = """# Preserve SHA256-bound evidence exactly in every checkout.
* -text
*.bin binary
*.png binary
*.jpg binary
*.jpeg binary
*.usdc binary
*.usdz binary
*.gz binary
# Original writer line endings and USD export EOF spacing are evidence.
*.csv whitespace=cr-at-eol
*.usda whitespace=-blank-at-eof
# Preserve generated plot whitespace without changing its digest.
task_plots/native08_source_clear_v1/drawer_task.svg whitespace=-blank-at-eol
task_plots/native09_source_clear_v1/drawer_task.svg whitespace=-blank-at-eol
task_plots/native10_source_clear_v1/drawer_task.svg whitespace=-blank-at-eol
"""

EVIDENCE_NOTES = """# Evidence notes: scope, history, privacy and reproduction

This bundle is a dated subset of a separate follow-up. Frozen pilot assets,
evaluators and scores are unchanged. Geometry04 preserves all five original
CC0 source meshes and has a conditional handoff; Joint02 has an accepted native
articulation receipt. Native Physics03,06 and07 fail. Physics08 also completed
with workflow status `fail`, ValidationEvidence `conditional`, and the typed
visual assessment `unresolved_issues`; its measured native smoke gates passed.
The independent five-seed source-clear task also failed all five seeds. Physics09 also ended unresolved and failed all five independent task trials.
Physics10 passed its native mounted-rest workflow and all five independent source-clear tasks on the same asset. The later actual native Validation terminal passed all four required gates and accepted the independent review. This completes the bounded conjunction; original Geometry conditions and optional unsupported cross-stage integrity remain unchanged.

The [complete task report](evaluations/native08_source_clear_v1/report.json) binds
the unchanged08 asset, bindings, specification and evaluator. The [full-range
plot](task_plots/native08_source_clear_v1/drawer_task.png), five CSVs and five
losslessly compressed full traces show failed opening, penetration and closed
hold gates. The [export receipt](task_plots/native08_source_clear_v1/export_verification.json)
and manifest bind both compressed and original uncompressed trace hashes.
The [source contact diagnostic](evidence/source_contact_diagnostic_08.json) measures
distance to original source surfaces at a recorded pose; proximity does not identify
a contact actor, prove solid overlap or establish the failure cause.

The [09 task report](evaluations/native09_source_clear_v1/report.json),
[09 full-range plot](task_plots/native09_source_clear_v1/drawer_task.png),
[09 exact trace export verification](task_plots/native09_source_clear_v1/export_verification.json),
and [independent09 startup review](evidence/native09_startup_review/review.json)
preserve the separate failed outcome and its unchanged asset. The [brief installed
runtime documentation excerpt](evidence/ovphysx_0_4_13_registry_excerpt.json)
binds package version/document SHA and exact lines137–138 supporting one narrow
nonfatal initialization-message exception. The full SDK document is not distributed.
The [initial strict audit](evidence/native09_critical_cooking_log_audit.json) remains
false and the [actual cooking log](evidence/native09_cooked_clearance_v1.log)
preserves every warning; this exception grants no workflow/task acceptance. A later
[static endpoint repair protocol](evidence/static_endpoint_repair_protocol_v1.json)
and [synthetic qualification](evidence/static_joint_endpoint_v1/review_receipt.json)
support the bounded static Mesh endpoint hypothesis. The separately completed
[Joint03 native terminal receipt](runs/drawer_joint_03/standalone_articulation_terminal_receipt.json)
is accepted; this establishes neither Physics10 nor physical-task success. The [native Validation launcher](validation_launch/README.md)
and [qualification](validation_launch/qualification.json) qualified the strict
consumer and staged native closure before execution. That qualification itself
launched no native workflow. Its task precheck
binds reports, not a new independent verification of solver traces. The launcher
later rejected actual Geometry04 because it assumed the wrong handoff field. The
[real refusal](validation_launch/history/geometry_handoff_assumption/original_assumption_refusal.json)
and old source/review remain exact in history. The corrected launcher consumes
Geometry's actual typed schema and conditional handoff, qualified by22 tests;
no Geometry receipt or policy was changed. The earlier independent review applies
only to its explicitly preserved old code, not this correction. Historical path
redirects in the publication manifest point to exact old code bytes. Any superseded
historical documentation that is not provided is labeled digest-attested only.

The [independent08 preflight review](evidence/native08_independent_preflight_review.json)
retains an actual plugin warning: requested hullVertexLimit128 lies outside the
observed supported8..64 range. Authored USD readback does not prove that128 took
effect. Clear initial cooked space and measured smoke stability do not establish
loaded motion or payload retention. Previous code qualification is preserved as
a historical result and does not qualify every cooking option at runtime. A
[later range correction](evidence/convex_options_bounds_v2/qualification.json)
restricts future authoring to8..64. The code patch includes that later correction;
it was not applied to, or used to reinterpret, the unchanged08 asset or trials.

The [outcome](outcome.json) and [publication manifest](publication_manifest.json)
separate exact bytes from explicitly selected/projected reports. Selected Joint03
terminal, identity, readback and preparation receipts remain exact where their
reviewed fields contain no private data. Mechanical endpoint fields are retained;
service/session metadata and decision reasoning are projected out when present. Embedded native
receipt hashes identify original retained files; consult the manifest to see
whether the target is published exactly or as a projection. Projected receipts
are not original native inputs to the strict Validation adapter. The native10
assessment, ValidationEvidence, runtime reports, scene/recording/trajectory, and
review images remain exact. Its render receipt and camera metadata need privacy
projection because nested commands contain workflow identifiers. Therefore the
public subset can verify exact measured/typed bytes and frame hashes but does not
recreate every original strict native producer hash binding. Original digest
attestation is explicitly distinct from exact public bytes.

[Source-clear initialization](SOURCE_CLEAR_FIXTURE.md) documents the predeclared
source-only payload placement correction and original source/proxy-floor limit.
Fixture qualification is not authored-asset acceptance. Native smoke stability
and visual review also do not establish opening or loaded-payload retention.
Physics10 separately completed its native workflow with all four checks passing
and a typed visual pass for three seconds of mounted rest. Its exact assessment
explicitly does not establish opening, closing, cavity fidelity or payload retention.
[Postflight](evidence/native10_postflight_execution.json) and the
[actual warnings](evidence/native10_cooking_log_review.json) preserve the initial
strict cooking audit failure, the exact documented nonfatal-message exception,
and CPU collision fallback limitations. No GPU, particle or deformable collision
qualification is inferred. The later final receipt verifies both native Validation
and independent five-seed task acceptance bound to the same authored USD digest.

The [original source](source/drawer_cabinet_1k.gltf) includes its exact BIN and
three texture dependencies. [License provenance](source/license_provenance.json)
retains the CC0 source attribution. Geometry's [native six-view render](runs/drawer_geometry_04/render_evidence/geometry_six_view.png)
and the [Joint02 canonical render](evidence/joint02_canonical.png) are evidence
of the retained geometry, not physical success. Physics07's eight retained
review frames accompany its failed native measured trace. Physics08 retains both
bounded review-frame sets and its unchanged unresolved assessment. Earlier frame
receipts may bind intermediate files subsequently superseded within that run;
the manifest distinguishes published bytes from original retained references.

Run `python tools/verify_public_bundle.py .` for checksum, privacy and reference
review; add `--decode-usd` with `usd-core` installed for decoded layer/dependency
review. The verified CPU environment used OpenUSD25.5; tools import `Usd` before
`Sdf` to initialize file-format plugins. The independently retained [25-candidate decoded scan](publication_review/usd_privacy_review.json)
[08 extension](publication_review/usd_privacy_review_08.json), and
[09/static-fixture extension](publication_review/usd_privacy_review_09.json), and
[Joint03/package/topology extension](publication_review/usd_privacy_review_joint03.json), and
[native10/retained09-test-fixture extension](publication_review/usd_privacy_review_10.json) cover source-host
closure, not portable dependency resolution. Run `python -m unittest discover -s
tools -p test_publication_tools.py` for provider-free negative controls.
`tools/verify_source_arrays.py` compares the original glTF arrays against
a selected USD. The [rebuild qualification](publication_review/portable_public_rebuild_review.json)
reproduced13source/evaluator/specification digests; the historical freeze remains
an explicitly disclosed exception. The [portable dependency qualification](publication_review/portable_public_bundle_r2_portable_closure.json)
found no missing or external dependencies across17USD/package files. Its relative
paths and hashes identify the separate relocated derivatives, not original files
at matching paths in this bundle. Five source-array checks on those derivatives
verified Geometry04, Joint02 and Physics03/07/08 against the original glTF.
Exact native USDs can retain generic absolute runtime locators;
`tools/relocate_usd_paths.py` creates separate portable copies without rewriting
original bytes. A relocated scene is a derivative and its new digest is reported.
A retained trajectory replay is presentation evidence, not a new simulation.
The [native10 replay receipt](task_replay/native10_seed11_v2/presentation_receipt.json)
checks three recorded steps, both drawer/payload positions and
quaternion agreement, and unchanged input/dependency bytes. The native render
command reports OVRTX and its probe is ready, but per-frame renderer identities
are null. Three PNGs are provided separately; the automatically generated
three-frame GIF is deliberately excluded. The exact seed11 request, scene and
replay are included; decompress its complete published trace to restore the
fourth input expected by [render_task_replay.py](render_task_replay.py). Generic
absolute native paths require the same separate relocation or original filesystem
layout described above. Running that script with rendering enabled is a new
presentation render, never a new physical trial or native Validation evidence.
[Decoded replay review](publication_review/usd_privacy_review_native10_replay.json)
retains the source-host dependency and privacy limits.
The Joint03 decoded scan likewise found only generic absolute locators across
its eight selected USD/package/render-stage inputs, with no unresolved source-host
dependencies. The original helper `package_for_joint_03.py` contained a private
GPU identifier: its published executable derivative changes only that selector
to the caller's `CUDA_VISIBLE_DEVICES` environment value. The manifest binds both
helper digests and labels the transformation. It has not been executed as part of
publication checks. All other selected exact helpers retain their original bytes.
The source-preserving Joint03 canonical image has been viewed; it does not display
joint motion or establish physics. Terminal receipts may reference intentionally
omitted private request/trace files; their digests remain retained-only references,
not a claim that the public bundle contains the whole original runtime context.

For a new independent task execution, first verify the bundle and create relocated
assets. Preserve the original binding and write a separate binding that changes
only its filesystem locator, then use the unchanged evaluator with an explicitly
chosen compatible ovphysx Python environment:

```sh
python tools/verify_public_bundle.py . --decode-usd
python tools/relocate_usd_paths.py . ../portable-native
python - <<'PYCODE'
import json
from pathlib import Path
binding = json.loads(Path("evidence/native08_task_bindings.json").read_text())
binding["final_usd"] = str(Path("../portable-native/runs/drawer_physics_08/physics.usda").resolve())
with Path("../portable-native/native08.replay-bindings.json").open("x") as f:
    json.dump(binding, f, indent=2)
PYCODE
python evaluator/source_clear_v1/drawer_evaluate.py \
  --bindings ../portable-native/native08.replay-bindings.json \
  --output ../fresh-native08-evaluation --solver /path/to/ovphysx-env/bin/python
```

This launches a new simulation only when explicitly executed. The source-only
rebuild and publication checks do not launch simulation. The copied USD retains
all non-asset opinions; relocation creates a different digest, so new results
must report that digest and must not be substituted for the historical08 receipt.
The evaluator requires its scientific Python dependencies and a compatible native
ovphysx installation; no backend or credential is redistributed here.

The historical `original_drawer_freeze.json` is a privacy projection: its original
digest is attested by the publication manifest but its original bytes are not
provided. The source-clear freeze remains exact. Reconstruct the old12-file input
from11unchanged files plus the exact `original_spec/drawer_acceptance.json` with
`python tools/rebuild_source_clear_fixture.py . /fresh/output`. The helper verifies
all original file hashes and runs the unchanged source-only builder. Its new
freeze differs in timestamp and projected historical-receipt hash; compare the
actual regenerated specification, source and evaluator bytes, not overall freeze
digests. This performs no simulation or authored-asset acceptance.

The [qualified task evidence auditor](task_evidence_audit_v1/README.md) independently
recomputes recorded formulas without launching a solver. Its [20-test qualification](task_evidence_audit_v1/qualification.json)
and [retained09 audit](task_evidence_audit_v1/native09_qualified/audit.json) corroborate
all five historical09 failures; audit consistency is not acceptance. Historical
failed auditor attempts and their code snapshots are retained, not rewritten.
The [independent native10 review](evidence/native10_independent_review/review.json)
and [completed native10 trace audit](task_evidence_audit_v1/native10/audit.json)
corroborate the separate five-seed PASS on the unchanged10 asset. Neither is a
native Validation terminal result. Only seed11's small09 request/scene fixtures are included for the three retained-USD
tests; a full audit rerun additionally needs every seed's exact request, scene,
replay and native log, as well as the decompressed full traces. Missing retained
inputs cannot be replaced with generated records. The test suite can use a
separate relocated copy arranged as `<test-root>/capstone` via `TASK_AUDIT_ROOT`;
its qualification receipt records testing on original retained bytes. A separate
[public-copy check](publication_review/task_audit_portable_tests.json) reproduced
all20 tests using independently relocated published assets, with no dependency
outside that portable directory. From the public bundle, with compatible OpenUSD,
NumPy and pytest installed:

```sh
python tools/relocate_usd_paths.py . ../audit-test-root/capstone
python - <<'PYTESTFILES'
from pathlib import Path
import shutil
for relative in [
    "task_evidence_audit_v1/audit.py", "task_evidence_audit_v1/test_audit.py",
    "evaluator/source_clear_v1/drawer_acceptance.json",
    "evidence/native09_task_bindings.json",
    "evaluations/native09_source_clear_v1/seed_11/request.json",
]:
    destination = Path("../audit-test-root/capstone") / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(relative, destination)
PYTESTFILES
TASK_AUDIT_ROOT="$(cd ../audit-test-root && pwd)" python -m pytest -q ../audit-test-root/capstone/task_evidence_audit_v1/test_audit.py
```

These are analytical and copied-scene tests only. They do not rerun physical trials.

[Completed native leaf accounting](evidence/completed_native_leaf_accounting.json)
records only its named native workflow invocations. It excludes outer development
and coordinator work, independent reviewers, native Validation, replay, tests,
probes and GPU cost. Dollar cost is unknown; missing times and counts are not
filled in. This partial accounting is not an end-to-end cost or workflow-efficiency
comparison.


The [actual native Validation terminal](runs/drawer_validation_01_prepared/native_run/validation_terminal_receipt.json)
now records completed/pass/review-accept. Its four required gates pass; optional
standalone cross-stage integrity remains not_evaluated. The [final assessment](runs/drawer_validation_01_prepared/native_run/canonical_validation_assessment.json)
and [independent native review](runs/drawer_validation_01_prepared/independent_review.json)
are exact safe final records. Schema-required gate/finding rationale strings are
reviewed final evidence-based explanations intended for readers; this narrow
exception does not publish private prompts or raw model reasoning. A boolean
base_url_configured only states whether configuration exists and contains no URL.
The accepted plan and plan patch remove the private child session identifier and
are explicit projections. The exact terminal retains original hashes for those
targets: the public verifier labels them ORIGINAL_RETAINED_BYTES_TARGET_IS_PROJECTION,
not exact public reproduction of the entire original native input closure. The
[export manifest](evidence/validation01_terminal_export.json) attests the retained
original42-file export, not that every export member is selected for publication.
The [final conjunction](evidence/final_capstone_conjunction.json) and its
[independent review](evidence/final_conjunction_independent_review.json) bind the
same authored asset, native terminal and separately audited five-seed task.
Geometry's original conditional handoff and formal SimReady not_evaluated remain
unchanged. Original topology warnings, estimated mass/materials, ideal connected-actor
collision exclusion, CPU-only scope and missing initial native pose/contact actor
identifiers remain limits. Extensive repairs are outside the frozen pilot budget.
The original Physics03 root console was overwritten by the later Physics07
launcher; surviving native receipts are retained and that missing console is not
reconstructed.

[Validation leaf accounting](evidence/validation01_leaf_accounting.json) is a
separate partial counter receipt. Outer development, assessor/reviewer work and
compute dollars remain outside these named native-leaf counters; it is not a total
cost or workflow-efficiency comparison.

[Code changes](code/capstone_changes.patch) are separate from run outputs.
[Validation policy example](validation_contract_review/POLICY_EXAMPLE.md) keeps
native smoke and the independently frozen physical task gate distinct.
"""

README = """# Native drawer capstone: bounded Validation and loaded-task checks passed

This separate, unscored follow-up preserves the original CC0 drawer geometry and
all completed outcomes. **Native Validation passed and all five independent
loaded-task trials passed on the exact same authored asset.** This is a bounded
CPU rigid-body, ideal-joint result; Geometry remains conditional and standalone
cross-stage integrity was not evaluated. The frozen pilot and its results are
unchanged. [Final conjunction](evidence/final_capstone_conjunction.json),
[independent review](evidence/final_conjunction_independent_review.json),
[machine-readable outcome](outcome.json).

| Stage | Retained outcome | Evidence |
| --- | --- | --- |
| Geometry04 | Source-preserving, conditional handoff | [Native result](runs/drawer_geometry_04/geometry_workflow_result.json), [six views](runs/drawer_geometry_04/render_evidence/geometry_six_view.png) |
| Joint02 | Native articulation accepted | [Terminal receipt](runs/drawer_joint_02/standalone_articulation_terminal_receipt.json) |
| Physics03 / 06 / 07 | Failed | [03](runs/drawer_physics_03/validation_evidence.json), [06](runs/drawer_physics_06/validation_evidence.json), [07](runs/drawer_physics_07/validation_evidence.json) |
| Physics08 | Native workflow failed; visual review unresolved; independent task **0/5** | [Assessment](runs/drawer_physics_08/physics_behavior_assessment.json), [task report](evaluations/native08_source_clear_v1/report.json) |
| Physics09 | Native workflow failed; visual review unresolved; independent task **0/5** | [Assessment](runs/drawer_physics_09/physics_behavior_assessment.json), [task report](evaluations/native09_source_clear_v1/report.json) |
| Joint03 | Native articulation accepted with source-bound static Mesh endpoint | [Exact terminal receipt](runs/drawer_joint_03/standalone_articulation_terminal_receipt.json), [readback](runs/drawer_joint_03/standalone_articulation_readback.json), [canonical image](evidence/joint03_canonical_view.png) |
| Physics10 | **Native PASS** for three-second mounted-rest behavior | [Exact assessment](runs/drawer_physics_10/physics_behavior_assessment.json), [exact four-check evidence](runs/drawer_physics_10/validation_evidence.json), [runtime report](runs/drawer_physics_10/runtime/runtime_validation_report.json) |
| Physics10 independent five-seed task | **5/5 PASS**, corroborated by a separate read-only trace audit | [Exact task report](evaluations/native10_source_clear_v1/report.json), [independent audit](task_evidence_audit_v1/native10/audit.json), [same-asset bindings](evidence/native10_task_bindings.json) |
| Native Validation01 | **Terminal PASS / review accept**, four required gates pass; optional cross-stage integrity not evaluated | [Exact terminal](runs/drawer_validation_01_prepared/native_run/validation_terminal_receipt.json), [exact final assessment](runs/drawer_validation_01_prepared/native_run/canonical_validation_assessment.json), [independent review](runs/drawer_validation_01_prepared/independent_review.json) |

![Physics10: five independent source-clear task trials, full displayed ranges](task_plots/native10_source_clear_v1/drawer_task.png)

The [three retained stills](task_replay/native10_seed11_v2/presentation_receipt.json)
show recorded seed11 poses at [settled](task_replay/native10_seed11_v2/frames/usd_cam__1024x1024__f0179.png),
[held open](task_replay/native10_seed11_v2/frames/usd_cam__1024x1024__f1230.png), and
[held closed](task_replay/native10_seed11_v2/frames/usd_cam__1024x1024__f2459.png).
They are presentation of the measured replay, not a continuous animation or new
simulation. Native OVRTX is reported by probe/render commands; individual frame
renderer identities are null, so the images grant no additional acceptance.

Native10's pass is a narrower milestone: its [exact typed assessment](runs/drawer_physics_10/physics_behavior_assessment.json)
limits that native claim to constrained mounted rest. The [independent loaded-task report](evaluations/native10_source_clear_v1/report.json)
passes all five seeds on the same authored USD. Its [trace audit](task_evidence_audit_v1/native10/audit.json)
checks all5×2460 recorded samples and retains the missing-initial-pose and contact
attribution limits. The completed native Validation terminal separately accepts
its static, runtime, visual and package gates without waivers. The [fresh native
render](runs/drawer_validation_01_prepared/native_run/operations/render_valid/renders/000_physics_c491b4e7/000_physics_c491b4e7_plus_xplus_yplus_z_0000.png)
is visual evidence, not a substitute for the measured task.
[10 export verification](task_plots/native10_source_clear_v1/export_verification.json)
binds all complete compressed traces to the original uncompressed hashes.

The [09 report](evaluations/native09_source_clear_v1/report.json) and
[08 report](evaluations/native08_source_clear_v1/report.json) bind each unchanged
asset and frozen task. [A separately qualified trace auditor](task_evidence_audit_v1/qualification.json)
corroborates09's failures without re-simulation. Each plot folder includes five CSVs and five **complete,
lossless** gzip traces. [09 export verification](task_plots/native09_source_clear_v1/export_verification.json)
binds compressed and original uncompressed hashes. Smoke stability, a render, or
accepted articulation does not establish loaded opening or payload retention.
The [final receipt](evidence/final_capstone_conjunction.json) verifies native
Validation **and** five independent task passes on exactly the same authored USD
SHA256. It does not confer universal SimReady, hardware realism or a causal
workflow cost or effectiveness claim.

Joint03 changes the cabinet joint target to the actual static Mesh and preserves
the upper drawer owner. The later [topology promotion](runs/drawer_topology_03/physics_topology_plan.json)
adds a rigid-body owner only to the upper drawer; it does not author masses or
colliders. The declared ideal-joint contract excludes cabinet-to-upper-drawer
contacts; independent payload contact and retention are still required. [Preparation](runs/articulation_preparation_03/articulation_preparation_publication.json),
[packaging04](runs/drawer_packaging_04/packaging_fidelity.json), and
[packaging05](runs/drawer_packaging_05/packaging_fidelity.json) preserve the source
chain. Independent CPU readback verifies all five original point/index arrays,
world transforms, units and axis for [Joint03](publication_review/original_arrays_joint03.json)
and [package05](publication_review/original_arrays_joint03_package05.json).
These are completed preparation/articulation stages, not physical acceptance.

The [original source](source/drawer_cabinet_1k.gltf), BIN and three textures are
exact CC0 source bytes. [License provenance](source/license_provenance.json).
[Source-clear fixture](SOURCE_CLEAR_FIXTURE.md) explains the predeclared source-only
payload placement correction and source/proxy-floor limitation.

From this bundle directory, checksum/privacy/link checks and synthetic negative
controls need Python; image checks need Pillow. No model, renderer or solver runs:

```sh
python tools/verify_public_bundle.py .
python -m unittest discover -s tools -p test_publication_tools.py
python tools/rebuild_source_clear_fixture.py . ../fresh-source-clear-fixture
```

With compatible OpenUSD installed, decode layers and create separately labeled
portable copies; original USD bytes remain unchanged:

```sh
python tools/verify_public_bundle.py . --decode-usd
python tools/relocate_usd_paths.py . ../portable-native
python tools/verify_source_arrays.py --help
```

[Evidence notes](EVIDENCE_NOTES.md) retain the full historical results, exact versus
projected hash semantics, intermediate receipt limits, original fixture
reconstruction, cooking warnings, portability qualifications and an explicit
command for a **new** native task execution. [Publication manifest](publication_manifest.json)
identifies every selected file. Exact receipts are retained when safe; projections
are labeled and are not interchangeable with original native Validation inputs.
[Code patch](code/capstone_changes.patch) is separate from authored outputs.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capstone", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.capstone.resolve(), args.output.resolve(), args.repo.resolve()),
            indent=2,
        )
    )
