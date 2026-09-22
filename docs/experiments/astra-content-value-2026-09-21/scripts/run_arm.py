#!/usr/bin/env python3
"""Launch one budgeted Astra Ultra arm on its assigned Horde GPU.
No acceptance decision is made here. Evaluator runs separately after outputs freeze.
"""
import argparse, hashlib, json, os, signal, subprocess, time, shutil, threading
from pathlib import Path
MODEL='gpt-6-astra'
EFFORT='ultra'
ROOT=Path(os.environ.get('ASTRA_EXPERIMENT_ROOT','/opt/astra-content-value-20260921')).resolve()
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);ap.add_argument('--arm',choices=['plain_astra','content_agents'],required=True);ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--budget-seconds',type=int,default=2400);ap.add_argument('--task-file',type=Path,required=True);ap.add_argument('--run-id',required=True)
    a=ap.parse_args(); task=a.task_file.resolve(); case=json.loads(task.read_text())
    if not case.get('frozen'):raise SystemExit('Case must be frozen before a scored run')
    if not 1 <= a.budget_seconds <= 2400:raise SystemExit('Budget must be between1 and2400 seconds')
    run=ROOT/'runs'/a.run_id/a.case/a.arm
    run.mkdir(parents=True,exist_ok=False)
    private=ROOT/'private-runs'/a.run_id/a.case/a.arm;private.mkdir(parents=True,mode=0o700,exist_ok=False)
    shutil.copy2(ROOT/'private-codex/auth.json',private/'auth.json');os.chmod(private/'auth.json',0o600)
    (private/'config.toml').write_text(f'model = "{MODEL}"\nmodel_reasoning_effort = "{EFFORT}"\napproval_policy = "never"\n')
    source=ROOT/case['source_root'];repo=ROOT/'repo'
    gpu_id=str(a.gpu)
    try:
        gpu_id=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).strip()
    except (OSError,subprocess.SubprocessError):pass
    (run/'.usd-cli').mkdir()
    (run/'.usd-cli/config.toml').write_text('[server]\nallowed_read_roots = '+json.dumps([str(run),str(source),str(repo)])+'\nallowed_write_roots = '+json.dumps([str(run)])+'\n[render]\nrenderer = "ovrtx"\novrtx_auto_install = false\n')
    common=f'''You are performing a frozen, paired asset experiment on this Horde GPU host. Use only {MODEL} with {EFFORT} reasoning. The task and budget are identical across arms. You have {a.budget_seconds} seconds total including tool use and all child-model work. Up to two repair iterations after your first candidate are allowed; preserve the first candidate and every repair separately. No human assistance is available during this run.
Work only under {run}. Original source closure is read-only at {source}. Do not modify it. Preserve actual authored source geometry and assembly identity; replacing the online asset with a newly modeled proxy is forbidden. Physics collision approximations are allowed when source visual geometry remains intact and the physical task stays valid.
Low-level tools available identically in both arms: Python/OpenUSD, usd-cli scene operations, pinned OVRTX and OvPhysX runtimes, usd-convert-cad0.2.0, OCP7.8.1.1/VTK9.3.1/rtree for native CAD source reading, shell, and your Astra image inspection. Renderer/physics work must run on this Horde host and preserve the assigned CUDA_VISIBLE_DEVICES={gpu_id}. No other LLM/VLM/model provider is permitted. If launching a repository coding-agent child, explicitly pass --model gpt-6-astra --model-reasoning-effort ultra, preserve its logs, and include its cost and time within your total budget. Never use a default or substitute model.
Do not read any other case's output, the opposite arm, evaluator implementation, hidden evaluation evidence, or other agents' logs. Read only this source closure, task, tool documentation, and your own outputs. Do not change installed packages, repository source, or evaluator. Missing capabilities must be recorded rather than worked around by changing the frozen experiment.
The independent evaluator will open your saved scene and simulate it itself. Do not write files into the evaluator directory. A visually appealing animation or a schema-only check is not physical task acceptance. Declare your own acceptance claim honestly, before seeing evaluator results.
Task: {case['task']}
Submission contract:\n{case['submission_contract']}\n
Acceptance criteria provided equally to both arms:\n{json.dumps(case['public_acceptance'],indent=2)}\n
Save final.usd, bindings.json, submission.json, first candidate, repairs, and your own evidence below this run directory. submission.json must contain claimed_accepted (boolean), final_scene (relative path or null), bindings (relative path or null), reason (string), repairs_used (integer 0..2), and limitations (list of strings). If blocked or budget-limited, preserve partial work and set claimed_accepted=false.
'''
    if a.arm=='plain_astra':
        directive=f'''Arm: plain Astra. Directly solve the task using the common low-level tools. Do not read or invoke Content Agents workflow skills, workflow coordinators, or workflow decision helpers. The common USD tool documentation at {repo}/apps/usd_cli/README.md and corresponding low-level APIs are available. You may write task-specific authoring/simulation code. The difference being tested is the Content Agents workflow, not your effort or input assets.'''
    else:
        directive=f'''Arm: Astra with Content Agents workflows. Use the repository's public Content Agents workflow skills and documented entrypoints at {repo}/.agents/skills. Start with quickstart, content-workflow-asset, content-workflow-articulation, content-workflow-physics and relevant Geometry/Validation guidance. Complete Geometry -> Joint -> Physics -> Validation with durable workflow receipts where supported. Honor native failed and conditional dispositions. Do not bypass a failed workflow by switching to an ad hoc standalone implementation and calling it a workflow success. You may produce a partial submission with explicit limitations. All model calls, including workflow children, must be Astra Ultra. The repository commit is a96faf9cb2f5c1f655fe0d60c0ccf57e3477b1aa; only its Codex executable dependency was upgraded to0.154.0 identically for both arms to support the specified model.'''
    prompt=common+'\n'+directive+'\n';(run/'prompt.txt').write_text(prompt)
    env={k:v for k,v in os.environ.items() if k in ['HOME','USER','LOGNAME','LANG','LC_ALL','TERM','LD_LIBRARY_PATH','SSL_CERT_FILE','REQUESTS_CA_BUNDLE','HTTP_PROXY','HTTPS_PROXY','NO_PROXY']}
    env.update(PATH=str(repo/'agentic/packages/content_workflow_cli/node_modules/.bin')+':'+str(repo/'.venv/bin')+':/home/horde/.local/bin:/usr/local/bin:/usr/bin:/bin',CODEX_HOME=str(private),CUDA_VISIBLE_DEVICES=gpu_id,WU_OVRTX_VENV_DIR=str(ROOT/'ovrtx-venv'),WU_OVPHYSX_VENV_DIR=str(ROOT/'ovphysx-venv'),WU_OVRTX_AUTO_PROVISION='0',UV_NO_CACHE='1',TMPDIR=str(ROOT/'tmp'),USD_CLI_SESSION='trial-'+a.run_id+'-'+a.case+'-'+a.arm,CONTENT_AGENT_MODEL=MODEL)
    cmd=[str(repo/'agentic/packages/content_workflow_cli/node_modules/.bin/codex'),'exec','--ignore-user-config','--skip-git-repo-check','-C',str(run),'-m',MODEL,'-c',f'model_reasoning_effort="{EFFORT}"','-c','approval_policy="never"','-s','danger-full-access','--json','-o',str(run/'last_message.txt'),'-']
    launch={'case':a.case,'arm':a.arm,'model':MODEL,'reasoning_effort':EFFORT,'gpu_index':a.gpu,'gpu_uuid':gpu_id,'hostname':os.uname().nodename,'budget_seconds':a.budget_seconds,'task_file':str(task),'task_sha256':digest(task),'prompt_sha256':digest(run/'prompt.txt'),'argv':cmd,'start_unix':time.time(),'scored':True}
    write(run/'launch.json',launch)
    began=time.monotonic();timed_out=False
    stop_monitor=threading.Event()
    observed_sessions={};session_offsets={};observed_contexts={}
    def sample_model_sessions():
        candidates=list(private.glob('sessions/**/*.jsonl'))+list(run.rglob('rollout-*.jsonl'))
        for session in candidates:
            if session.is_symlink():continue
            try:
                with session.open('r',errors='replace') as f:
                    f.seek(session_offsets.get(str(session),0))
                    while True:
                        offset=f.tell();line=f.readline()
                        if not line or not line.endswith('\n'):f.seek(offset);break
                        try:record=json.loads(line)
                        except Exception:continue
                        payload=record.get('payload',{})
                        if record.get('type')=='turn_context':
                            context={'session_file':session.name,'model':payload.get('model'),'reasoning_effort':payload.get('effort',payload.get('model_reasoning_effort'))}
                            observed_contexts[(session.name,str(context['model']),str(context['reasoning_effort']))]=context
                        if record.get('type')=='event_msg' and payload.get('type')=='token_count' and payload.get('info'):
                            counter=payload['info'].get('total_token_usage')
                            if counter is not None:observed_sessions[session.name]=counter
                    session_offsets[str(session)]=f.tell()
            except OSError:continue
    def monitor():
        with (run/'gpu_samples.jsonl').open('w') as log:
            while not stop_monitor.is_set():
                try:
                    sample=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=timestamp,uuid,utilization.gpu,utilization.memory,memory.used,power.draw','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
                    log.write(json.dumps({'unix':time.time(),'csv':sample})+'\n');log.flush()
                except (OSError,subprocess.SubprocessError):pass
                sample_model_sessions()
                stop_monitor.wait(10)
    monitor_thread=threading.Thread(target=monitor,daemon=True);monitor_thread.start()
    with (run/'events.jsonl').open('w') as out,(run/'stderr.log').open('w') as err:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=out,stderr=err,env=env,cwd=run,start_new_session=True,text=True)
        proc.stdin.write(prompt);proc.stdin.close()
        try:proc.wait(timeout=a.budget_seconds)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    elapsed=time.monotonic()-began;usage=[];threads=[];errors=[]
    stop_monitor.set();monitor_thread.join(timeout=12)
    for line in (run/'events.jsonl').read_text(errors='replace').splitlines():
        try:e=json.loads(line)
        except Exception:continue
        if e.get('type')=='thread.started':threads.append(e.get('thread_id'))
        if e.get('type')=='turn.completed':usage.append(e.get('usage',{}))
        if e.get('type') in ['turn.failed','error']:errors.append(e)
    sample_model_sessions()
    rollout_usage=[{'session_file':k,'total_token_usage':v} for k,v in observed_sessions.items()]
    write(run/'model_usage_audit.json',{'requested_model':MODEL,'requested_effort':EFFORT,'observed_turn_contexts':list(observed_contexts.values()),'session_usage':rollout_usage,'note':'Samples include primary and transient workflow shadow homes while alive. Sum each distinct session once; final bridge usage records take precedence over last periodic sample for deleted shadow sessions. Audit any uncaptured children from workflow launch and usage receipts.'})
    write(run/'execution.json',{'elapsed_seconds':elapsed,'gpu_allocation_seconds':elapsed,'timed_out':timed_out,'exit_code':proc.returncode,'usage_events':usage,'thread_ids':threads,'errors':errors,'nested_usage_requires_separate_aggregation':a.arm=='content_agents','evaluation':'pending_independent_evaluator','human_interventions':0,'human_review_seconds':None,'human_review_timing_note':'No human review occurred during autonomous run; report review labour separately if measured.'})
    files=[]
    for p in sorted(run.rglob('*')):
        if p.is_file() and not p.is_symlink() and '.usd-cli' not in p.parts and p.name!='output_manifest.json':
            try:files.append({'path':str(p.relative_to(run)),'size':p.stat().st_size,'sha256':digest(p)})
            except OSError:pass
    write(run/'output_manifest.json',{'files':files,'frozen_unix':time.time()})
    print(json.dumps({'run':str(run),'elapsed_seconds':elapsed,'exit_code':proc.returncode,'timed_out':timed_out}))
if __name__=='__main__':main()
