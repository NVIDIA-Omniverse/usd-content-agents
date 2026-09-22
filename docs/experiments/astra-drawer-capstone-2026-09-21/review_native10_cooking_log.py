import hashlib,json
from pathlib import Path
cap=Path('/opt/astra-content-value-20260921/capstone');doc=Path('/opt/astra-content-value-20260921/ovphysx-venv/lib/python3.12/site-packages/ovphysx/docs/ovphysx_overview.md');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
initial=cap/'evidence/native10_critical_cooking_log_audit.json';data=json.loads(initial.read_text());messages=[m for log in data['logs'] for m in log['messages']]
exact='[Error] [omni.physx.cooking.plugin] registry is false.'
assert '`[Error] [omni.physx.cooking.plugin] registry is false` — a non-fatal initialization' in doc.read_text()
def blocking(m):return m['classification']=='critical_configuration_or_runtime_error' and m['verbatim']!=exact
assert not blocking({'classification':'critical_configuration_or_runtime_error','verbatim':exact})
assert blocking({'classification':'critical_configuration_or_runtime_error','verbatim':'[Error] [omni.convexdecomposition.plugin] Invalid max hull vertices(128) must be between 8 and 64'})
assert blocking({'classification':'critical_configuration_or_runtime_error','verbatim':exact+' additional failure'})
assert blocking({'classification':'critical_configuration_or_runtime_error','verbatim':'[Error] arbitrary cooking failure'})
clear_path=cap/'evidence/native10_cooked_clearance_v1.json';clear=json.loads(clear_path.read_text());assert clear['query_positive_control'] and clear['asset_unchanged'] and clear['simulation_steps']==0
result={'asset_sha256':data['asset_sha256'],'initial_strict_audit':str(initial),'initial_strict_audit_sha256':sha(initial),'original_audit_unchanged':True,'installed_runtime_documentation':str(doc),'documentation_sha256':sha(doc),'documented_exact_nonfatal_message':exact,'documentation_lines':[137,138],'blocking_messages':[m for m in messages if blocking(m)],'all_warnings_retained':data['logs'],'query_positive_control_receipt':str(clear_path),'query_positive_control_sha256':sha(clear_path),'boundary':'Only the exact initialization error documented non-fatal by the installed runtime is scoped as non-fatal. Remaining startup/service/thread warnings stay disclosed; effective CPU collision is measured by actual populated positive-control queries and task trials. No GPU collision, particle/deformable collision, task pass or native workflow pass is inferred.','classification_regressions_passed':4,'cpu_task_probe_permitted':not any(blocking(m) for m in messages)}
out=cap/'evidence/native10_cooking_log_review.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'cpu_task_probe_permitted':result['cpu_task_probe_permitted'],'remaining_blocking':result['blocking_messages']}))
