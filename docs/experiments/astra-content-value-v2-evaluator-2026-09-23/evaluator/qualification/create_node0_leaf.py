"""Create only parent-authorized own-container evaluator child; no process moves."""
import json
from pathlib import Path
from datetime import datetime,timezone
root=Path('/opt/astra-content-value-20260922-rerun');known=json.loads((root/'harness/qualification/node0_cgroups.json').read_text())
line=next(x for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'));current=Path('/sys/fs/cgroup'+line[3:])
assert current.name=='astra-v2-supervisor' and str(current.parent)==known['container_scope']
parent=current.parent;assert parent.name.endswith('.scope');assert {'cpu','memory','pids'}<=set((parent/'cgroup.subtree_control').read_text().split())
child=parent/'evaluator-v2';child.mkdir(exist_ok=False)
for key,value in {'cpu.max':'200000 100000','memory.max':'8589934592','memory.swap.max':'0','pids.max':'512'}.items():(child/key).write_text(value)
receipt={'created_at':datetime.now(timezone.utc).isoformat(),'scope':'Only root-authorized evaluator leaf inside own container; no parent caps or processes changed','path':str(child),'limits':{name:(child/name).read_text().strip() for name in ['cpu.max','memory.max','memory.swap.max','pids.max','cgroup.events']}}
(root/'evaluator/qualification/node0_evaluator_cgroup.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
