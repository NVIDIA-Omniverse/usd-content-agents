"""Private session capture and metadata-only Codex UI-effort readback.

Wire xhigh cannot establish Ultra. Contexts and child CLI/config provenance must
also be audited. Polling can miss a very short deleted child, so completeness is
explicitly unknown until response IDs are reconciled with the gateway ledger.
"""
import hashlib
import json
import os
import stat
import threading
from pathlib import Path

from common import EFFORT, MODEL, sha256, write_new


class Capture:
    def __init__(self, roots, destination, uid):
        self.roots = [Path(x) for x in roots]
        self.destination = Path(destination)
        self.destination.mkdir(mode=0o700)
        self.uid = uid
        self.files = {}
        self.errors = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def start(self): self.thread.start()

    def scan(self):
        for root in self.roots:
            for _parent, _dirs, filenames, directory in os.fwalk(root, follow_symlinks=False):
                for name in filenames:
                    if not (name.endswith(".jsonl") and (name.startswith("rollout-") or name.startswith("session"))): continue
                    fd = None
                    try:
                        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
                        info = os.fstat(fd)
                        if not stat.S_ISREG(info.st_mode) or info.st_uid != self.uid: continue
                        identity = (info.st_dev, info.st_ino)
                        old = self.files.get(identity)
                        if old is None or info.st_size < old["offset"]:
                            key = f"capture_{len(self.files):06d}_{len(self.errors):03d}"
                            if old: self.errors.append({"type": "session_truncated", "capture": old["capture"]})
                            old = {"offset": 0, "capture": key + ".jsonl"}; self.files[identity] = old
                        os.lseek(fd, old["offset"], os.SEEK_SET)
                        # Append complete lines only; a partial final line is
                        # retried, not silently consumed as malformed JSON.
                        data = os.read(fd, 16 * 1024 * 1024)
                        split = data.rfind(b"\n")
                        if split >= 0:
                            with (self.destination / old["capture"]).open("ab") as out: out.write(data[:split+1])
                            old["offset"] += split + 1
                    except OSError as exc:
                        if not isinstance(exc, (FileNotFoundError, PermissionError)):
                            self.errors.append({"type": type(exc).__name__})
                    finally:
                        if fd is not None: os.close(fd)

    def loop(self):
        while not self.stop_event.is_set():
            try: self.scan()
            except OSError as exc: self.errors.append({"type": type(exc).__name__})
            self.stop_event.wait(.25)

    def stop(self):
        self.stop_event.set()
        if self.thread.ident is not None: self.thread.join(timeout=5)
        previous = None
        while previous != sum(x["offset"] for x in self.files.values()):
            previous = sum(x["offset"] for x in self.files.values())
            self.scan()
        return summarize(self.destination, self.errors)


def summarize(destination, errors=()):
    sessions = {}
    capture_errors = list(errors)
    for p in sorted(Path(destination).glob("*.jsonl")):
        own = None
        file_hash = sha256(p)
        for line in p.read_text(errors="replace").splitlines():
            try: record = json.loads(line)
            except ValueError:
                capture_errors.append({"type": "invalid_json", "capture": p.name})
                continue
            payload = record.get("payload", {})
            if not isinstance(payload, dict): continue
            # Fork files begin with their own identity and may then contain
            # copied ancestor metadata. Never relabel the child as its parent.
            if record.get("type") == "session_meta" and own is None:
                own = payload.get("id")
                if own:
                    metadata = {key: payload[key] for key in ("session_id", "parent_thread_id") if key in payload}
                    session = sessions.setdefault(own, {"contexts": [], "response_ids": set(), "captures": {},
                        "own_metadata": metadata, "own_turn_anchors": [], "response_records": [], "identity_conflicts": []})
                    if session["own_metadata"] != metadata:
                        session["identity_conflicts"].append("Conflicting own session metadata")
            if not own: continue
            session = sessions[own]
            session["captures"][p.name] = file_hash
            kind = record.get("type")
            if kind == "turn_context":
                context = {"model": payload.get("model"), "ui_reasoning_effort": payload.get("effort", payload.get("model_reasoning_effort"))}
                for key in ("turn_id", "root_turn_id", "parent_turn_id", "thread_id"):
                    if key in payload: context[key] = payload[key]
                if context not in session["contexts"]: session["contexts"].append(context)
            if kind in ("token_usage_record", "event_msg") and payload.get("thread_id") == own and payload.get("turn_id"):
                anchor = {key: payload[key] for key in ("session_id", "thread_id", "turn_id", "root_turn_id", "parent_turn_id") if key in payload}
                anchor["record_type"] = kind
                if anchor not in session["own_turn_anchors"]: session["own_turn_anchors"].append(anchor)
            if kind == "token_usage_record" and payload.get("thread_id") == own and payload.get("response_id"):
                session["response_ids"].add(payload["response_id"])
                association = {key: payload[key] for key in ("response_id", "session_id", "thread_id", "turn_id", "root_turn_id") if key in payload}
                if association not in session["response_records"]: session["response_records"].append(association)
    result = []
    for own, s in sessions.items():
        result.append({"session_id": own, "thread_id": own, **{k: s[k] for k in ("contexts", "own_metadata", "own_turn_anchors", "response_records", "identity_conflicts")},
                       "response_ids": sorted(s["response_ids"]), "private_captures_sha256": s["captures"],
                       "observed_ui_matches": bool(s["contexts"]) and all(x["model"] == MODEL and x["ui_reasoning_effort"] == EFFORT for x in s["contexts"])})
    return {"schema_version": "codex-ui-audit.v2", "sessions": result,
            "all_observed_ui_matches": bool(result) and all(x["observed_ui_matches"] for x in result), "capture_errors": capture_errors,
            "complete": None, "completeness_note": "Must reconcile every admitted session/thread/turn with an independently captured own-thread anchor and Ultra turn context; cross-check terminal response IDs when present. Historical missing admission identities remain incomplete.",
            "publication": "Metadata only after privacy projection; full captured sessions are private and contain model content."}


def reconcile(audit, usage):
    """Exact association only. UI completeness and terminal billing are separate."""
    from common import object_hash, require
    from request_identity import REQUIRED, OPTIONAL, canonical_uuid, validate_projection
    sessions = {s.get("thread_id", s["session_id"]): s for s in audit["sessions"]}
    rows = []
    covered_threads = set()
    direct_count = admission_count = 0
    for request in usage["requests"]:
        reason = None
        mode = None
        try:
            identity = validate_projection(request.get("request_identity"))
            own = identity["thread_id"]
            require(own in sessions, "Admitted thread not captured")
            session = sessions[own]
            require(not session.get("identity_conflicts"), "Conflicting own session metadata")
            metadata = session.get("own_metadata", {})
            for value in metadata.values(): canonical_uuid(value)
            require(metadata.get("session_id") == identity["session_id"], "Admitted session differs from own thread metadata")
            for key in OPTIONAL:
                if key in identity and key in metadata:
                    require(identity[key] == metadata[key], "Own parent/root metadata conflict")
            contexts = [x for x in session["contexts"] if x.get("turn_id") == identity["turn_id"]]
            require(bool(contexts), "Admitted turn context not captured")
            anchors = [x for x in session.get("own_turn_anchors", []) if x.get("turn_id") == identity["turn_id"]]
            require(bool(anchors), "Turn has no own-thread anchor; copied ancestor context is insufficient")
            for key in OPTIONAL:
                values = {context[key] for context in contexts if key in context}
                require(len(values) <= 1, "Conflicting contexts for one turn")
            for context in contexts:
                for key in REQUIRED + OPTIONAL:
                    if key in context: canonical_uuid(context[key])
                require(context.get("model") == MODEL and context.get("ui_reasoning_effort") == EFFORT, "Turn model/UI effort mismatch")
                require(context.get("thread_id", own) == own, "Context belongs to a different thread")
                for key in OPTIONAL:
                    if key in identity and key in context:
                        require(identity[key] == context[key], "Turn parent/root context conflict")
            for anchor in anchors:
                for key in REQUIRED + OPTIONAL:
                    if key in anchor: canonical_uuid(anchor[key])
                    if key in anchor and key in identity:
                        require(anchor[key] == identity[key], "Own-thread event/usage identity conflict")
            response_id = request.get("response_id")
            matched = [(thread, record) for thread, s in sessions.items() for record in s.get("response_records", [])
                       if response_id and record.get("response_id") == response_id]
            for thread, record in matched:
                require(thread == own, "Terminal response observed in another thread")
                for key in REQUIRED:
                    canonical_uuid(record.get(key))
                    require(record.get(key) == identity[key], "Terminal response association contradicts admission")
                for key in OPTIONAL:
                    if key in record: canonical_uuid(record[key])
                    if key in record and key in identity:
                        require(record[key] == identity[key], "Terminal parent/root association conflict")
            mode = "terminal_response_id_and_admission" if matched else "admission_and_own_turn_only"
            if matched: direct_count += 1
            else: admission_count += 1
            covered_threads.add(own)
        except (ValueError, KeyError, TypeError) as exc:
            reason = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        rows.append({"request_id": request["id"], "associated": reason is None, "association": mode, "reason": reason})
    expected = {r["response_id"] for r in usage["requests"] if r.get("response_id")}
    observed = {x for s in sessions.values() for x in s["response_ids"]}
    unledgered = sorted(observed - expected)
    uncovered = sorted(set(sessions) - covered_threads)
    complete = bool(rows) and all(r["associated"] for r in rows) and not unledgered and not uncovered and not audit["capture_errors"] and audit["all_observed_ui_matches"]
    return {"schema_version": "codex-request-ui-association.v2", "complete": complete,
            "audit_sha256": object_hash(audit), "usage_sha256": object_hash(usage), "requests": rows,
            "terminal_response_id_associations": direct_count, "admission_identity_associations": admission_count,
            "unledgered_response_ids": unledgered, "uncovered_threads": uncovered,
            "terminal_billing_complete": usage["complete"], "timing_inference_used": False,
            "scope": "Client UUID association plus captured own-thread turn/UI evidence. Not a cryptographic attestation of client behavior. Pinned client/process/config provenance remains required."}
