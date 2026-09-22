#!/usr/bin/env python3
"""Plot independent measured trajectories; contact impulses are converted to newtons."""
import argparse,csv,gzip,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 base=a.root/'evaluations/pilot-v1/01_drawer/plain_astra';out=a.root/'report/drawer';out.mkdir(parents=True,exist_ok=True)
 colors=['#137b63','#297fbb','#9c63bb','#b47926','#bf5d71'];fig,axs=plt.subplots(2,2,figsize=(11.6,6.8),sharex=True);records=[]
 for seed,color in zip([11,23,47,83,131],colors):
  trace=base/f'seed_{seed}/trace.jsonl';data=[json.loads(x) for x in trace.read_text().splitlines()];report=json.loads((base/f'seed_{seed}/trial_report.json').read_text())
  t=np.array([r['time_s'] for r in data]);q=np.array([r['q_m'] for r in data]);force=np.array([np.linalg.norm(np.array(r['payload_drawer_contact_force_n']))*240 for r in data]);relative=np.array([r['payload_pose'][2]-r['drawer_pose'][2] for r in data]);pen=np.array([max([0]+[-c['separation'] for c in r['contacts']]) for r in data])
  axs[0,0].plot(t,q*100,color=color,lw=1.5,label=f'Seed {seed}');axs[0,1].plot(t,relative*100,color=color,lw=1.2);axs[1,0].plot(t,force,color=color,lw=1.0);axs[1,1].plot(t,pen*1000,color=color,lw=1.0);axs[1,1].scatter(t[0],pen[0]*1000,color=color,s=18,zorder=4)
  with (out/f'seed_{seed}_measurements.csv').open('w',newline='') as f:
   w=csv.writer(f);w.writerow(['time_s','drawer_displacement_m','payload_relative_z_m','payload_contact_force_n_from_impulse','max_contact_penetration_m']);w.writerows(zip(t,q,relative,force,pen))
  compressed=out/f'seed_{seed}_trace.jsonl.gz'
  with compressed.open('wb') as f:
   with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as z:z.write(trace.read_bytes())
  records.append({'seed':seed,'status':report['status'],'metrics':report['metrics'],'checks':report['checks'],'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),'compressed_trace_sha256':hashlib.sha256(compressed.read_bytes()).hexdigest()})
 axs[0,0].axhline(20,color='#666',ls='--',lw=1,label='Minimum opening');axs[0,0].set_ylabel('Drawer displacement (cm)')
 axs[0,1].set_ylabel('Payload offset in drawer (cm)');axs[0,1].set_title('Outward-axis coordinate; free payload',fontsize=10)
 axs[1,0].set_ylabel('Payload contact force (N)');axs[1,0].axhline(.5*9.81,color='#666',ls='--',lw=1);axs[1,0].set_ylim(0,7)
 axs[1,1].set_ylabel('Contact penetration (mm)');axs[1,1].axhline(5,color='#666',ls='--',lw=1);axs[1,1].set_ylim(-.15,5.5);axs[1,1].annotate('Initial maximum: 3.5 mm',xy=(.0042,3.5),xytext=(1.0,3.0),fontsize=9,arrowprops={'arrowstyle':'-','color':'#555'})
 for ax in axs.flat:
  ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15);ax.set_xlim(0,11)
 for ax in axs[1]:ax.set_xlabel('Simulated time (s)')
 fig.suptitle('Drawer pull: five independent native-physics trials passed',x=.07,ha='left',fontsize=16,fontweight='bold')
 fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.51,.935),ncol=6,frameon=False,fontsize=9)
 fig.text(.07,.01,'Original Poly Haven cabinet geometry retained • direct Astra arm • 0.5 kg free payload • force-driven motion\nContact values converted from impulse using Δt = 1/240 s. This is a simulation task, not real-hardware validation.',fontsize=9,color='#444')
 fig.tight_layout(rect=(.02,.07,1,.93));fig.savefig(out/'drawer_validation.png',dpi=170);fig.savefig(out/'drawer_validation.svg');plt.close(fig)
 (out/'summary.json').write_text(json.dumps({'case':'01_drawer','arm':'plain_astra','all_five_pass':all(r['status']=='PASS' for r in records),'source':'independent frozen evaluator native CPU PhysX trajectories','contact_units':'raw misnamed payload_drawer_contact_force_n field is impulse in N*s; plotted force = vector norm / (1/240 s)','trials':records},indent=2)+'\n')
 print(out/'drawer_validation.png')
if __name__=='__main__':main()
