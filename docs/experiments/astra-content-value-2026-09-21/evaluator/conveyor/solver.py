"""Independent contact transfer trial; writes forces/torques, never poses."""
import json, math, sys, traceback
from pathlib import Path
import numpy as np
from ovphysx import PhysX,TensorType
from common import dump,pose_matrix,joint_measure,sha

def run(request):
 out=Path(request['output']);out.mkdir(parents=True,exist_ok=True)
 report={'status':'INCONCLUSIVE','accepted':False,'seed':request['seed']};sdk=None;handles=[]
 try:
  assert sha(request['scenario'])==request['scene_sha256'],'Evaluation scene changed'
  config=request['config'];spec=request['spec'];roles=config['body_roles'];bodies=config['bodies'];joint=config['joints'][config['joint_roles']['rail_rotation']]
  sdk=PhysX(device='cpu');_,op=sdk.add_usd(request['scenario']);sdk.wait_op(op)
  states={};forces={};wrenches={};mass={};inertia={}
  def binding(path,kind):
   b=sdk.create_tensor_binding(prim_paths=[path],tensor_type=getattr(TensorType,kind),raise_if_empty=True);handles.append(b);return b
  def read(b):
   a=np.zeros(b.shape,np.float32);b.read(a);return a.reshape(-1).astype(float)
  for role in ['rail','carrier','payload']:
   path=roles[role];states[path]=(binding(path,'RIGID_BODY_POSE'),binding(path,'RIGID_BODY_VELOCITY'))
   forces[path]=binding(path,'RIGID_BODY_FORCE');wrenches[path]=binding(path,'RIGID_BODY_WRENCH')
   mass[role]=read(binding(path,'RIGID_BODY_MASS')).tolist();inertia[role]=read(binding(path,'RIGID_BODY_INERTIA')).tolist()
  witness=(binding(config['gravity_witness'],'RIGID_BODY_POSE'),binding(config['gravity_witness'],'RIGID_BODY_VELOCITY'))
  contacts={}
  for key,sensor,target in [('carrier_rail','carrier','rail'),('payload_carrier','payload','carrier')]:
   b=sdk.create_contact_binding(sensor_patterns=[roles[sensor]],filter_patterns=[roles[target]],filters_per_sensor=1,max_contact_data_count=256);handles.append(b)
   assert b.sensor_count==1 and b.filter_count==1,'Unresolved contact identity';contacts[key]=b
  def poses():return {p:(read(states[p][0]).tolist() if p in states else b['pose']) for p,b in bodies.items()}
  current=poses();initial=current.copy();q0=joint_measure(joint,current)['q'];old=q0;unwrapped=q0
  center=np.array(spec['center_world_m']);points={role:np.array(config['tracked_local_points'][role]) for role in ['carrier','payload']}
  def point(role,ps):return (pose_matrix(ps[roles[role]])@np.r_[points[role],1])[:3]
  p0={role:point(role,current) for role in points};angles0={r:math.atan2(p[1]-center[1],p[0]-center[0]) for r,p in p0.items()};angles=dict(angles0);previous_angles=dict(angles0)
  radii={r:float(np.linalg.norm((p-center)[:2])) for r,p in p0.items()}
  rng=np.random.default_rng(request['seed']);dt=spec['dt_s'];total=spec['duration_s'];count=round(total/dt)
  metrics={'max_penetration_m':0.,'max_linear_speed_m_s':0.,'max_angular_speed_rad_s':0.,'max_joint_closure_m':0.,'max_axis_error_rad':0.,'max_torque_nm':0.,'max_carrier_radial_error_m':0.,'max_payload_radial_error_m':0.,'max_carrier_height_error_m':0.,'max_payload_height_error_m':0.,'min_payload_above_carrier_m':1e9,'max_payload_from_carrier_m':0.,'carrier_rail_contact_frames':0,'payload_carrier_contact_frames':0,'transport_frames':0,'finite':True}
  witness0=read(witness[0]);witness_end=None;oldq=q0;rows=[]
  with (out/'trace.jsonl').open('w') as log:
   for step in range(count):
    t=step*dt;current=poses();m=joint_measure(joint,current);delta=(m['q']-old+math.pi)%(2*math.pi)-math.pi;unwrapped+=delta;old=m['q'];qdot=(unwrapped-oldq)/dt;oldq=unwrapped
    if t<1:target=q0;target_v=0.
    elif t<11:
     u=(t-1)/10;target=q0+spec['target_angle_rad']*(.5-.5*math.cos(math.pi*u));target_v=spec['target_angle_rad']*.5*math.pi/10*math.sin(math.pi*u)
    else:target=q0+spec['target_angle_rad'];target_v=0.
    torque=0. if t<1 else float(np.clip(.1*(target-unwrapped)+.02*(target_v-qdot),-spec['max_torque_nm'],spec['max_torque_nm']))
    for path in states:
     f=np.zeros(forces[path].shape,np.float32);w=np.zeros(wrenches[path].shape,np.float32)
     if path==roles['rail']:w.reshape(-1)[3:6]=np.array(m['axis_world'])*torque
     if path==roles['payload'] and 6<=t<6.5:f.reshape(-1)[:2]=[.0005,float(rng.uniform(-.0001,.0001))]
     forces[path].write(f);wrenches[path].write(w)
    sdk.step_sync(dt,t);current=poses();velocities={p:read(v[1]) for p,v in states.items()};ps={r:point(r,current) for r in points};cf={}
    for key,b in contacts.items():
     a=np.zeros((1,1,3),np.float32);b.read_force_matrix(a);cf[key]=(a.reshape(-1)/dt).tolist()
    native=sdk.get_contact_report();penetration=max([0.]+[-float(native['points'][i].separation) for i in range(native['num_points'])]);metrics['max_penetration_m']=max(metrics['max_penetration_m'],penetration)
    metrics['max_joint_closure_m']=max(metrics['max_joint_closure_m'],m['closure_m']);metrics['max_axis_error_rad']=max(metrics['max_axis_error_rad'],m['axis_error_rad']);metrics['max_torque_nm']=max(metrics['max_torque_nm'],abs(torque))
    for v in velocities.values():
     metrics['max_linear_speed_m_s']=max(metrics['max_linear_speed_m_s'],float(np.linalg.norm(v[:3])));metrics['max_angular_speed_rad_s']=max(metrics['max_angular_speed_rad_s'],float(np.linalg.norm(v[3:])))
    finite=all(np.isfinite(v).all() for v in list(velocities.values())+[np.array(p) for p in current.values()]+[np.array(v) for v in cf.values()]);metrics['finite'] &= bool(finite)
    for role,p in ps.items():
     a=math.atan2(p[1]-center[1],p[0]-center[0]);angles[role]+=(a-previous_angles[role]+math.pi)%(2*math.pi)-math.pi;previous_angles[role]=a
     if t>=1:
      metrics['max_'+role+'_radial_error_m']=max(metrics['max_'+role+'_radial_error_m'],abs(float(np.linalg.norm((p-center)[:2]))-radii[role]));metrics['max_'+role+'_height_error_m']=max(metrics['max_'+role+'_height_error_m'],abs(float(p[2]-p0[role][2])))
    if t>=1:
     metrics['min_payload_above_carrier_m']=min(metrics['min_payload_above_carrier_m'],float(ps['payload'][2]-ps['carrier'][2]));metrics['max_payload_from_carrier_m']=max(metrics['max_payload_from_carrier_m'],float(np.linalg.norm(ps['payload']-ps['carrier'])))
    if 2<=t<11:
     metrics['transport_frames']+=1
     for k in contacts:metrics[k+'_contact_frames']+=int(np.linalg.norm(cf[k])>1e-5)
    if witness_end is None and (step+1)*dt>=.1:witness_end={'t':(step+1)*dt,'pose':read(witness[0]).tolist(),'velocity':read(witness[1]).tolist()}
    row={'step':step,'time_s':(step+1)*dt,'poses':current,'velocities':{p:v.tolist() for p,v in velocities.items()},'points_world_m':{r:p.tolist() for r,p in ps.items()},'rail_angle_rad':unwrapped-q0,'carrier_angle_rad':angles['carrier']-angles0['carrier'],'payload_angle_rad':angles['payload']-angles0['payload'],'torque_nm':torque,'contact_forces_n':cf,'max_penetration_m':penetration}
    log.write(json.dumps(row,allow_nan=False)+'\n')
    if step%24==0 or step==count-1:rows.append(row)
    if not finite or metrics['max_linear_speed_m_s']>1 or metrics['max_angular_speed_rad_s']>20:break
  checks={'completed':step+1==count,'finite':metrics['finite'],'rail_rotated':unwrapped-q0>=spec['minimum_angle_rad'],'carrier_transferred':angles['carrier']-angles0['carrier']>=spec['minimum_angle_rad'],'payload_transferred':angles['payload']-angles0['payload']>=spec['minimum_angle_rad'],'penetration':metrics['max_penetration_m']<=.0015,'joint_closed':metrics['max_joint_closure_m']<=.001,'joint_axis':metrics['max_axis_error_rad']<=.02,'linear_speed':metrics['max_linear_speed_m_s']<=1,'angular_speed':metrics['max_angular_speed_rad_s']<=20,'carrier_stayed_on_track':metrics['max_carrier_radial_error_m']<=.007 and metrics['max_carrier_height_error_m']<=.004,'payload_retained':metrics['max_payload_radial_error_m']<=.012 and metrics['max_payload_height_error_m']<=.008 and metrics['min_payload_above_carrier_m']>.003 and metrics['max_payload_from_carrier_m']<=.025,'force_limited':metrics['max_torque_nm']<=spec['max_torque_nm']+1e-7}
  for k in contacts:checks[k+'_contact_fraction']=metrics['transport_frames']>0 and metrics[k+'_contact_frames']/metrics['transport_frames']>=.9
  checks['independent_gravity']=witness_end is not None and abs(witness_end['velocity'][2]+9.81*witness_end['t'])<.03 and abs(witness_end['pose'][2]-witness0[2]+.5*9.81*witness_end['t']**2)<.004
  for role,v in mass.items():checks[role+'_native_mass']=bool(np.isfinite(v).all() and spec['mass_ranges_kg'][role][0]<=v[0]<=spec['mass_ranges_kg'][role][1])
  checks={k:bool(v) for k,v in checks.items()}
  report.update(status='PASS' if all(checks.values()) else 'FAIL',accepted=all(checks.values()),checks=checks,metrics=metrics,final_rail_angle_rad=unwrapped-q0,final_carrier_angle_rad=angles['carrier']-angles0['carrier'],final_payload_angle_rad=angles['payload']-angles0['payload'],native_mass=mass,native_inertia=inertia,scene_sha256=request['scene_sha256'],gravity_witness=witness_end,trace_samples=rows)
 except Exception as exc:report.update(error=repr(exc),traceback=traceback.format_exc())
 finally:
  for h in reversed(handles):
   try:h.destroy()
   except Exception:pass
  if sdk:sdk.release()
 dump(out/'report.json',report);return report
if __name__=='__main__':
 result=run(json.loads(Path(sys.argv[1]).read_text()));print(json.dumps({'status':result['status'],'accepted':result['accepted']}))
