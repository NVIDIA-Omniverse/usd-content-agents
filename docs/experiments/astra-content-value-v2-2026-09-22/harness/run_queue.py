#!/usr/bin/env python3
"""Single-controller, resumeless paired queue; one sequential queue per GPU lane.

The shared controller DB remains authoritative across both worker hosts. A lost
connection leaves its lease occupied. No timeout automatically frees a lane.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import secrets
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

from common import (ARMS, identifier, object_hash, read, require, sha256,
                    validate_pair, validate_protocol, write_new)
from ledger import Ledger
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "controller"))
from retain_run import retain


def failure_details(exc, stage, process=None):
    """Useful fixed diagnostics; arbitrary exception content stays excluded."""
    known = {'Worker did not return a trusted reap receipt',
             'Retention proof missing or mismatched',
             'Retention export failed; preserve remote run and stop lane',
             'Wrong reap binding', 'Conflicting reap receipt', 'Unknown lease',
             'A deadline or lost SSH connection does not prove the worker is reaped'}
    reason = str(exc) if type(exc) in (ValueError, RuntimeError) and str(exc) in known else None
    if isinstance(exc, json.JSONDecodeError): reason = 'Trusted operation returned malformed JSON'
    elif isinstance(exc, subprocess.TimeoutExpired): reason = 'Trusted operation exceeded its fixed timeout'
    elif isinstance(exc, OSError): reason = 'Trusted operation failed with OS errno ' + str(exc.errno)
    return {'exception_type': type(exc).__name__, 'stage': stage,
            'safe_reason': reason or 'Trusted operation rejected or failed; inspect private process output',
            'worker_returncode': getattr(process, 'returncode', None)}


def retain_then_reap(db, cfg, protocol, job, lease, receipt, retention=retain):
    # Cancellation stops new model requests but deliberately leaves the lane
    # occupied. A verified local archive and exact worker cleanup are required
    # before another case can inherit that lane.
    db.cancel(job["run_id"])
    retained = retention(dict(cfg, protocol=protocol), job)
    require(retained.get("verified") is True and retained.get("run_id") == job["run_id"], "Retention proof missing or mismatched")
    db.reap(job["run_id"], lease, receipt)
    return retained


def validate_jobs(protocol, jobs):
    require(isinstance(jobs, list) and jobs, "No jobs")
    cases = {}
    for job in jobs:
        identifier(job["case_id"])
        identifier(job["run_id"])
        require(job["arm"] in ARMS, "Unknown arm")
        require(job["lane_id"] in {x["id"] for x in protocol["lanes"]}, "Unknown lane")
        cases.setdefault(job["case_id"], []).append(job)
    require(len({x["run_id"] for x in jobs}) == len(jobs), "Run IDs must be fresh and unique")
    for case, pair in cases.items():
        require(len(pair) == 2, f"Case {case} needs exactly two arms")
        validate_pair(*pair)
        expected = ARMS if int(case.split("_", 1)[0]) % 2 else tuple(reversed(ARMS))
        require(tuple(x["arm"] for x in pair) == expected, "Pair order differs from predeclared counterbalance")
    return True


def launch(config_file, jobs_file):
    cfg = read(config_file)
    protocol = validate_protocol(read(cfg["protocol_file"]))
    jobs = read(jobs_file)["jobs"]
    validate_jobs(protocol, jobs)
    require(all(x["protocol_sha256"] == sha256(cfg["protocol_file"]) for x in jobs), "Job protocol differs")
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=False)
    db = Ledger(cfg["database"])
    db.configure(protocol)
    db.db.close()
    routes_file = Path(cfg["routes_file"])
    require(not routes_file.exists(), "Routing file exists; no inherited run credentials")
    write_new(routes_file, {"routes": []})
    routes_file.chmod(0o600)
    lock = threading.Lock()

    def worker(lane):
        db = Ledger(cfg["database"])
        reports = []
        for job in (x for x in jobs if x["lane_id"] == lane["id"]):
            # Token has no upstream authority and is valid only for this run.
            token = secrets.token_urlsafe(32)
            lease_requested_unix = time.time()
            lease = db.acquire(job, protocol["wall_seconds"])
            allocation_monotonic = time.monotonic()
            timing = {"schema_version":"trusted-run-timing.v2","run_id":job['run_id'],"lane_id":lane['id'],
                      "lease_requested_unix":lease_requested_unix,"lease_acquired_unix":time.time(),
                      "dispatch_started_unix":None,"budget_started_unix":None,"deadline_unix":None,
                      "worker_reap_received_unix":None,"retention_started_unix":None,"reaped_after_retention_unix":None,
                      "author_elapsed_seconds":None,"allocated_lane_seconds":None,
                      "allocation_released":False,"clock":"Controller monotonic elapsed; controller Unix timestamps; worker author elapsed separately reported"}
            release_monotonic = None
            process = None
            stage = 'worker_dispatch'
            with lock:
                routes = read(routes_file)
                routes["routes"].append({"run_id": job["run_id"], "token_sha256": hashlib.sha256(token.encode()).hexdigest()})
                temporary = routes_file.with_suffix(".next")
                temporary.write_text(json.dumps(routes))
                temporary.chmod(0o600)
                temporary.replace(routes_file)
            request = dict(job, lease=lease, gateway_token=token)
            host = cfg["hosts"][lane["host"]]
            # The shell receives only frozen supervisor/config paths. Prompts,
            # code, credentials and author commands never appear in SSH argv.
            remote = "sudo -n " + shlex.join([host["python"], host["supervisor"], "--config", host["worker_config"]])
            argv = ["tsh", "ssh", "--proxy=" + cfg["teleport_proxy"], "horde@" + host["hostname"], remote]
            try:
                db.start(job["run_id"], lease)
                budget = db.db.execute("SELECT started,deadline FROM runs WHERE id=?", (job["run_id"],)).fetchone()
                request["deadline_unix"] = budget['deadline']
                timing.update(budget_started_unix=budget['started'],deadline_unix=budget['deadline'],dispatch_started_unix=time.time())
                process = subprocess.run(argv, input=json.dumps(request), text=True, capture_output=True,
                                         timeout=protocol["wall_seconds"] + 120)
                require(process.returncode == 0, "Worker did not return a trusted reap receipt")
                receipt = json.loads(process.stdout)
                timing.update(worker_reap_received_unix=time.time(),author_elapsed_seconds=receipt.get('elapsed_seconds'))
                timing['retention_started_unix']=time.time()
                stage = 'verified_retention_and_reap'
                retained = retain_then_reap(db, cfg, protocol, job, lease, receipt)
                release_monotonic=time.monotonic()
                timing.update(reaped_after_retention_unix=time.time(),allocation_released=True,
                              allocated_lane_seconds=release_monotonic-allocation_monotonic)
                usage = db.usage(job["run_id"])
                stage = 'controller_receipt_publication'
                write_new(output / job["run_id"] / "worker_reap.json", receipt)
                write_new(output / job["run_id"] / "usage.json", usage)
                write_new(output / job["run_id"] / "retention.json", retained)
                # An ordinary exhausted budget is a valid experimental outcome.
                # Missing final upstream usage affects cost completeness, not an
                # otherwise isolated task comparison. Setup/transport failures
                # receive separate adjudication; no missing usage becomes zero.
                eligible = receipt["source_unchanged"]
                reports.append({"run_id": job["run_id"], "preliminary_protocol_eligible": eligible,
                                "author_outcome": "budget_exhausted" if receipt["timed_out"] else "completed",
                                "cost_accounting_complete": usage["complete"],
                                "physical_acceptance": "NOT_ASSESSED", "author_protocol_audit": "REQUIRED"})
            except Exception as exc:
                db.cancel(job["run_id"])
                private = output / job['run_id'] / 'private'
                private.mkdir(parents=True, exist_ok=True, mode=0o700)
                streams = {}
                for name in ('stdout', 'stderr'):
                    value = getattr(process, name, None) if process is not None else getattr(exc, name, None)
                    if value is not None:
                        path = private / ('worker_' + name + '.txt')
                        with path.open('xb') as stream:
                            stream.write(value.encode() if isinstance(value, str) else value)
                        path.chmod(0o600)
                        streams[name + '_sha256'] = sha256(path)
                write_new(output / job["run_id"] / "controller_failure.json",
                          {"run_id": job["run_id"], **failure_details(exc, stage, process), **streams,
                           "lane_reusable": False, "instruction": "Inspect trusted worker, kill/reap its exact lane, retain proof, then explicitly reconcile. Never infer reap from elapsed time."})
                break  # Do not dispatch this lane again after ambiguous failure.
            finally:
                if release_monotonic is None:
                    timing['allocation_observed_seconds_without_release']=time.monotonic()-allocation_monotonic
                write_new(output / job['run_id'] / 'timing.json',timing)
        db.db.close()
        return reports

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(worker, protocol["lanes"]))
    write_new(output / "queue_result.json", {"lanes": results, "protocol_sha256": sha256(cfg["protocol_file"]), "jobs_sha256": sha256(jobs_file)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    if args.validate_only:
        cfg = read(args.config)
        validate_jobs(validate_protocol(read(cfg["protocol_file"])), read(args.jobs)["jobs"])
        print('{"configuration_valid":true,"runtime_qualified":false,"launched":false}')
    else:
        launch(args.config, args.jobs)
