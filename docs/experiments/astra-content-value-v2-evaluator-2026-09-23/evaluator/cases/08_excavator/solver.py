from v2_control import clamp_effort
"""Fresh native PhysX experiment. Run ONLY with ovphysx Python; no pxr imports.

Only force/wrench tensors are written. Poses, velocities, joint coordinates and
targets are never overwritten. Traces are generated here, not read from arms.
"""
import argparse
import json
import math
from pathlib import Path
import traceback
import numpy as np
from ovphysx import PhysX, TensorType
from common import dump, joint_measure, frame, pose_matrix

class State:
    def __init__(self, sdk, path, moving):
        self.path, self.moving = path, moving
        self.bindings = {}
        names = ['RIGID_BODY_POSE','RIGID_BODY_VELOCITY','RIGID_BODY_MASS','RIGID_BODY_INERTIA','RIGID_BODY_COM_POSE']
        if moving:names += ['RIGID_BODY_FORCE','RIGID_BODY_WRENCH']
        for name in names:
            self.bindings[name] = sdk.create_tensor_binding(prim_paths=[path], tensor_type=getattr(TensorType,name),raise_if_empty=True)
    def read(self, name):
        b=self.bindings[name];arr=np.zeros(b.shape,np.float32);b.read(arr)
        return arr.reshape(-1).astype(float)
    def apply(self, force, torque):
        b=self.bindings['RIGID_BODY_FORCE'];v=np.zeros(b.shape,np.float32);v.reshape(-1)[:3]=force;b.write(v)
        b=self.bindings['RIGID_BODY_WRENCH'];v=np.zeros(b.shape,np.float32)
        if v.shape[-1]!=9:raise RuntimeError('unsupported wrench tensor layout '+str(v.shape))
        v.reshape(-1)[3:6]=torque;b.write(v)
    def close(self):
        for b in self.bindings.values():b.destroy()

def schedule(contract, role, q0, t, joint):
    observer=contract.get('observer')
    if observer=='unsupported':observer=contract.get('partial_observer')
    if observer=='door':target=q0+1.22
    elif observer=='engine':target=q0+2*math.pi+0.25
    elif observer=='arm':target=q0+contract['target_offsets_rad'][contract['required_joint_roles'].index(role)]
    elif observer=='slider':target=q0+0.03
    elif observer=='axes':target=q0+0.025
    else:return q0
    if observer=='engine':
        return q0+(target-q0)*np.clip((t-1)/7,0,1)
    if t<1:return q0
    if t<5:return q0+(target-q0)*(t-1)/4
    if t<7:return target
    if t<9:return target+(q0-target)*(t-7)/2
    if t<10:return q0
    # Deliberate bounded outward effort against a real authored upper limit.
    if joint.get('upper') is not None and joint['upper']>=joint.get('lower',-1e30):
        return q0+(joint['upper']+(0.3 if joint['kind']=='revolute' else 0.01)-q0)*min((t-10)/2,1) if observer in ('arm','axes') else joint['upper']+(0.3 if joint['kind']=='revolute' else 0.01)
    return q0

def run(config, seed, output, device='cpu'):
    contract=config['contract'];common=contract['common'];dt=common['dt_s']
    count=round(common['duration_s']/dt);rng=np.random.default_rng(seed)
    sdk=PhysX(device=device)
    states={};trace=[]
    try:
        _,op=sdk.add_usd(config['scenario']);sdk.wait_op(op)
        for path,b in config['bodies'].items():
            if b['moving']:states[path]=State(sdk,path,True)
        witness=State(sdk,config['gravity_witness'],False)
        def poses_now():
            return {path:(states[path].read('RIGID_BODY_POSE').tolist() if path in states else b['pose']) for path,b in config['bodies'].items()}
        poses=poses_now()
        initial_poses={p:list(q) for p,q in poses.items()}
        initial={p:joint_measure(j,poses)['q'] for p,j in config['joints'].items()}
        previous=initial.copy();unwrapped=initial.copy();qdot={p:0. for p in initial}
        native_mass={path:states[path].read('RIGID_BODY_MASS').tolist() for path in states}
        native_inertia={path:states[path].read('RIGID_BODY_INERTIA').tolist() for path in states}
        native_com={path:states[path].read('RIGID_BODY_COM_POSE').tolist() for path in states}
        controller_scales={};controller_descendants={}
        for path,joint in config['joints'].items():
            if joint['kind']=='fixed':continue
            reached={joint['body1']};front=[joint['body1']]
            while front:
                node=front.pop()
                for other_path,other in config['joints'].items():
                    if other_path==path:continue
                    for a,b in [(other['body0'],other['body1']),(other['body1'],other['body0'])]:
                        if a==node and b not in reached and b in states and b!=joint['body0']:reached.add(b);front.append(b)
            origin=frame(joint,poses,0)[:3,3];axis=np.asarray(joint_measure(joint,poses)['axis_world']);effective=0.
            for body in reached:
                if body not in states:continue
                matrix=pose_matrix(poses[body]);com=matrix[:3,:3]@np.asarray(native_com[body][:3])+matrix[:3,3];delta=com-origin;perp=delta-axis*np.dot(axis,delta)
                inertia=np.asarray(native_inertia[body]);inertia=inertia.reshape(3,3) if len(inertia)==9 else np.diag(inertia)
                effective+=float(np.max(np.linalg.eigvalsh(inertia)))+native_mass[body][0]*float(np.dot(perp,perp))
            controller_scales[path]=max(effective,1e-12);controller_descendants[path]=reached
        witness_initial=witness.read('RIGID_BODY_POSE').tolist()
        witness_at_point_one=None
        max_linear=max_angular=max_penetration=max_axis=max_closure=0.
        all_finite=True;contacts_observed=0;limit_extrema={p:[q,q] for p,q in initial.items()}
        control_roles=list(contract.get('control',{}))
        if not control_roles and contract.get('control_default'):control_roles=contract['required_joint_roles']
        effort_jitter=float(rng.uniform(.98,1.02))
        for step in range(count):
            t=step*dt;poses=poses_now();forces={p:np.zeros(3) for p in states};torques={p:np.zeros(3) for p in states}
            measures={}
            for path,joint in config['joints'].items():
                measure=joint_measure(joint,poses);raw=measure['q']
                delta=raw-previous[path]
                if joint['kind']=='revolute':delta=(delta+math.pi)%(2*math.pi)-math.pi
                unwrapped[path]+=delta;previous[path]=raw;qdot[path]=delta/dt
                measure['q_unwrapped']=unwrapped[path];measure['qdot']=qdot[path]
                measures[path]=measure
                limit_extrema[path][0]=min(limit_extrema[path][0],unwrapped[path]);limit_extrema[path][1]=max(limit_extrema[path][1],unwrapped[path])
                max_closure=max(max_closure,measure['closure_m']);max_axis=max(max_axis,measure['axis_error_rad'])
                role=joint.get('role')
                if role not in control_roles:continue
                setting=contract.get('control',{}).get(role,contract.get('control_default',{}))
                target=schedule(contract,role,initial[path],t,joint)
                gain_scale=controller_scales.get(path,1.) if setting.get('gain_mode')=='inertia_scaled' else 1.
                gravity_ff=0.
                if setting.get('gravity_compensate'):
                    origin=frame(joint,poses,0)[:3,3];axis=np.asarray(measure['axis_world'])
                    for body in controller_descendants.get(path,set()):
                        if body not in states:continue
                        mat=pose_matrix(poses[body]);com=mat[:3,:3]@np.asarray(native_com[body][:3])+mat[:3,3];weight=np.array([0.,0.,-9.81*native_mass[body][0]])
                        gravity_ff-=float(np.dot(axis,weight if joint['kind']=='prismatic' else np.cross(com-origin,weight)))
                effort=float(np.clip(gravity_ff+gain_scale*(setting['kp']*(target-unwrapped[path])-setting['kd']*qdot[path]),-setting['max_effort'],setting['max_effort']))*effort_jitter
                effort=clamp_effort(effort,setting['max_effort'])
                external=0.
                load=.01*setting['max_effort']
                if setting.get('gain_mode')=='inertia_scaled':load=min(load,gain_scale*1.0)
                if 5.5<=t<6.5:external=-load*float(config.get('external_load_scale',1.0))
                # Small reproducible initial external perturbation across five seeds.
                if t<.05:external+=float(rng.uniform(-.005,.005))*min(setting['max_effort'],gain_scale*1.0 if setting.get('gain_mode')=='inertia_scaled' else setting['max_effort'])
                axis=np.asarray(measure['axis_world']);total=(effort+external)*axis
                destination=forces if joint['kind']=='prismatic' else torques
                if joint['body1'] in destination:destination[joint['body1']]+=total
                if joint['body0'] in destination:destination[joint['body0']]-=total
                measure.update({'target':float(target),'effort':effort,'external_effort':external})
            for path,state in states.items():state.apply(forces[path],torques[path])
            sdk.step_sync(dt,t)
            report=sdk.get_contact_report();contacts_observed+=report['num_headers']
            for i in range(report['num_points']):
                point=report['points'][i]
                max_penetration=max(max_penetration,max(0.,-float(point.separation)))
                all_finite &= bool(np.isfinite(list(point.position)+list(point.normal)+list(point.impulse)+[point.separation]).all())
            velocities={p:s.read('RIGID_BODY_VELOCITY').tolist() for p,s in states.items()}
            for v in velocities.values():max_linear=max(max_linear,float(np.linalg.norm(v[:3])));max_angular=max(max_angular,float(np.linalg.norm(v[3:])))
            all_finite &= all(np.isfinite(p).all() for p in poses.values()) and all(np.isfinite(v).all() for v in velocities.values())
            if witness_at_point_one is None and (step+1)*dt>=.1:witness_at_point_one={'t':(step+1)*dt,'pose':witness.read('RIGID_BODY_POSE').tolist(),'velocity':witness.read('RIGID_BODY_VELOCITY').tolist()}
            if step%12==0 or step==count-1:
                trace.append({'t':t,'pose_time_s':t,'velocity_time_s':(step+1)*dt,'control_time_s':t,'poses':poses,'velocities':velocities,'joints':measures,'contact_headers':report['num_headers']})
            if not all_finite:break
            # A recorded mandatory bound failure is irreversible; retain witness and trace.
            if witness_at_point_one is not None and (max_linear>contract['common']['max_linear_speed_m_s'] or max_angular>contract['common']['max_angular_speed_rad_s'] or max_closure>contract['common']['max_joint_closure_m']):break
        result={'seed':seed,'completed_steps':step+1,'expected_steps':count,'direct_solver':'ovphysx.PhysX','device':device,
                'only_control_writes':['RIGID_BODY_FORCE','RIGID_BODY_WRENCH'],'native_mass':native_mass,'native_inertia':native_inertia,'native_com':native_com,'controller_effective_inertia':controller_scales,
                'native_initial_poses':initial_poses,'trace_timing':'poses/joints/control at t; velocities/contacts after step at t+dt',
                'gravity_witness':{'initial_pose':witness_initial,'at_point_one_s':witness_at_point_one},
                'max_linear_speed_m_s':max_linear,'max_angular_speed_rad_s':max_angular,
                'max_joint_closure_m':max_closure,'max_joint_axis_error_rad':max_axis,'max_contact_penetration_m':max_penetration,
                'finite_states_and_contacts':bool(all_finite),'contact_headers_total':contacts_observed,'joint_extrema':limit_extrema,'trace':trace}
        dump(output,result)
        witness.close()
    finally:
        for s in states.values():s.close()
        sdk.release()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
    try:run(json.loads(a.config.read_text()),a.seed,a.output,a.device)
    except Exception as exc:
        dump(a.output,{'seed':a.seed,'infrastructure_error':str(exc),'traceback':traceback.format_exc()})
        raise
