"""Independent native gripper contact trial; no pxr, author code or trajectories."""
import argparse,json,math,traceback
from pathlib import Path
import numpy as np
from ovphysx import PhysX,TensorType
from common import dump,joint_measure

class Body:
    def __init__(self,sdk,path,moving=True):
        self.path=path;self.bs={};self.moving=moving
        names=['POSE','VELOCITY','MASS','INERTIA']+(['WRENCH'] if moving else [])
        for name in names:
            b=sdk.create_tensor_binding(pattern=path,tensor_type=getattr(TensorType,'RIGID_BODY_'+name))
            if b.shape[0]!=1:raise RuntimeError('Body did not resolve exactly once: '+path)
            self.bs[name]=b
    def get(self,name):
        b=self.bs[name];a=np.zeros(b.shape,np.float32);b.read(a);return a.reshape(-1).astype(float)
    def apply(self,force,torque):
        b=self.bs['WRENCH'];a=np.zeros(b.shape,np.float32)
        if a.shape[-1]!=9:raise RuntimeError('Unsupported wrench tensor layout')
        a.reshape(-1)[:3]=force;a.reshape(-1)[3:6]=torque
        # API defines a WORLD application point. Payload/toy fingers explicitly
        # have COM at the body origin. The real mechanism receives pure torque.
        a.reshape(-1)[6:9]=self.get('POSE')[:3]
        b.write(a)
    def close(self):
        for b in self.bs.values():b.destroy()

def run(config,seed,output,device='cpu'):
    c=config['case04'];dt=c['dt_s'];ph=c['phases_s'];n=round(ph['end']/dt)
    sdk=PhysX(device=device);states={};extra=[];trace=[];rng=np.random.default_rng(seed)
    try:
        _,op=sdk.add_usd(config['scenario']);sdk.wait_op(op)
        for path,b in config['bodies'].items():
            if b['moving']:states[path]=Body(sdk,path)
        payload=config['payload_path'];states[payload]=Body(sdk,payload)
        left=config['body_roles']['left_finger'];right=config['body_roles']['right_finger']
        witness=Body(sdk,config['gravity_witness'],False);extra.append(witness)
        contact=sdk.create_contact_binding(sensor_patterns=[payload],filter_patterns=[left,right],filters_per_sensor=2,max_contact_data_count=256)
        extra.append(contact);cf=np.zeros((contact.sensor_count,contact.filter_count,3),np.float32)
        if cf.shape!=(1,2,3):raise RuntimeError('Bilateral contact binding unresolved')
        filters=list(contact.filter_paths)
        # Preserve the API path list and resolve columns by identity, never order.
        if len(filters)==1 and isinstance(filters[0],list):filters=filters[0]
        idx=[filters.index(p) for p in [left,right]]
        for path in [payload,left,right]:
            for name,values in [('SHAPE_FRICTION_AND_RESTITUTION',[c['payload']['friction'],c['payload']['friction'],0]),('CONTACT_OFFSET',c['payload']['contact_offset_m']),('REST_OFFSET',0)]:
                b=sdk.create_tensor_binding(pattern=path,tensor_type=getattr(TensorType,'RIGID_BODY_'+name));extra.append(b)
                a=np.zeros(b.shape,np.float32);a[...]=values
                if config.get('_synthetic_variant')=='no_friction' and name=='SHAPE_FRICTION_AND_RESTITUTION':a[...]=0
                b.write(a)
        def poses_now():return {p:(states[p].get('POSE').tolist() if p in states else b['pose']) for p,b in config['bodies'].items()}
        initial_poses=poses_now();payload_initial=states[payload].get('POSE');witness_initial=witness.get('POSE');witness_at=None
        q0={p:joint_measure(j,initial_poses)['q'] for p,j in config['joints'].items()};prev=q0.copy();q=q0.copy()
        grip=np.asarray(config['grip_axis_world']);closeaxis=np.asarray(config['closing_axis_world'])
        native_mass={p:s.get('MASS').tolist() for p,s in states.items()};native_inertia={p:s.get('INERTIA').tolist() for p,s in states.items()}
        jitter=float(rng.uniform(*c['controller']['seed_multiplier_range']));max_pen=max_linear=max_angular=max_closure=max_axis=0.;finite=True
        metrics={'hold_samples':0,'bilateral_samples':0,'max_hold_displacement_m':0.,'min_gravity_height_m':1e10,'min_loaded_height_m':1e10,'max_torque_Nm':0.,'max_loading_force_N':0.,'max_force_after_loading_N':0.,'min_contact_axis_forces_N':[1e10,1e10]}
        extrema={p:[v,v] for p,v in q.items()};closures=[];release=[]
        for step in range(n):
            t=step*dt;poses=poses_now();vel={p:s.get('VELOCITY') for p,s in states.items()}
            if not all(np.isfinite(v).all() for v in list(poses.values())+list(vel.values())):
                finite=False;break
            forces={p:np.zeros(3) for p in states};torques={p:np.zeros(3) for p in states};measures={}
            for p,j in config['joints'].items():
                m=joint_measure(j,poses);delta=m['q']-prev[p]
                if j['kind']=='revolute':delta=(delta+math.pi)%(2*math.pi)-math.pi
                q[p]+=delta;prev[p]=m['q'];m['q_unwrapped']=q[p];m['qdot']=delta/dt;measures[p]=m
                max_closure=max(max_closure,m['closure_m']);max_axis=max(max_axis,m['axis_error_rad']);extrema[p]=[min(extrema[p][0],q[p]),max(extrema[p][1],q[p])]
            if t<ph['start_close']:fraction=0.
            elif t<ph['close_end']:fraction=(t-ph['start_close'])/(ph['close_end']-ph['start_close'])
            elif t<ph['start_open']:fraction=1.
            elif t<ph['open_end']:fraction=1-(t-ph['start_open'])/(ph['open_end']-ph['start_open'])
            else:fraction=0.
            if config.get('_synthetic_fixture'):
                for p,sign in [(left,1.),(right,-1.)]:
                    displacement=float(np.dot(np.asarray(poses[p][:3])-initial_poses[p][:3],grip))*sign
                    target=.0025*fraction
                    effort=float(np.clip(1200*(target-displacement)-2.0*np.dot(vel[p][:3],grip)*sign,-2,2))*jitter
                    if config.get('_synthetic_variant')=='no_motion':effort=0.
                    forces[p]+=effort*sign*grip
            else:
                act=config['joint_roles']['actuator'];j=config['joints'][act];m=measures[act]
                sign=float(config['actuator_sign']);target=q0[act]+sign*c['controller']['close_offset_rad']*fraction
                ctrl=c['controller'];torque=float(np.clip(ctrl['kp_Nm_rad']*(target-q[act])-ctrl['kd_Nms_rad']*m['qdot'],-ctrl['max_torque_Nm'],ctrl['max_torque_Nm']))*jitter
                # Jitter is applied to the command, then clipped to the frozen cap.
                torque=float(np.clip(torque,-ctrl['max_torque_Nm'],ctrl['max_torque_Nm']));metrics['max_torque_Nm']=max(metrics['max_torque_Nm'],abs(torque))
                torque_axis=np.asarray(m['axis_world'])
                for body,f in [(j['body1'],1),(j['body0'],-1)]:
                    if body in torques:torques[body]+=f*torque*torque_axis
                m.update(target=target,torque_Nm=torque)
            pp=states[payload].get('POSE');support=np.zeros(3)
            if t<ph['remove_loading_support']:
                fix=c['loading_fixture'];support=fix['kp_N_m']*(payload_initial[:3]-pp[:3])-fix['kd_Ns_m']*vel[payload][:3]+np.array([0,0,c['payload']['mass_kg']*9.81])
                norm=np.linalg.norm(support)
                if norm>fix['max_force_N']:support*=fix['max_force_N']/norm
                forces[payload]+=support;metrics['max_loading_force_N']=max(metrics['max_loading_force_N'],float(np.linalg.norm(support)))
            else:metrics['max_force_after_loading_N']=max(metrics['max_force_after_loading_N'],float(np.linalg.norm(support)))
            if ph['load_start']<=t<ph['start_open']:forces[payload][2]-=c['load_downward_N']
            for p,s in states.items():s.apply(forces[p],torques[p])
            sdk.step_sync(dt,t);poses=poses_now();pp=states[payload].get('POSE');cf.fill(0);contact.read_force_matrix(cf)
            # ovphysx0.4.13 step_sync returns impulses here despite its Python
            # docstring. Three analytic m*g+1N probes at two dt and two masses
            # verify this conversion; see case04_contact_units_evidence.json.
            cf/=dt
            report=sdk.get_contact_report();seps=[]
            for i in range(report['num_points']):
                pt=report['points'][i];seps.append(float(pt.separation));finite &= bool(np.isfinite(list(pt.position)+list(pt.normal)+list(pt.impulse)+[pt.separation]).all())
            current_vel={p:s.get('VELOCITY') for p,s in states.items()}
            finite &= all(np.isfinite(v).all() for v in list(current_vel.values())+[pp,cf]) and all(np.isfinite(v).all() for v in poses.values())
            # Keep invalid numbers out of JSON without disguising physical
            # instability as an evaluator/serialization failure.
            if not finite:break
            max_pen=max([max_pen]+[-x for x in seps]);axisforces=[float(np.dot(cf[0,k],grip)) for k in idx]
            bilateral=abs(axisforces[0])>=c['limits']['min_bilateral_normal_force_N'] and abs(axisforces[1])>=c['limits']['min_bilateral_normal_force_N'] and axisforces[0]*axisforces[1]<0
            hold=(ph['gravity_observe_start']<=t<ph['load_start']) or (ph['load_observe_start']<=t<ph['start_open'])
            if hold:
                metrics['hold_samples']+=1;metrics['bilateral_samples']+=int(bilateral)
                metrics['max_hold_displacement_m']=max(metrics['max_hold_displacement_m'],float(np.linalg.norm(pp[:3]-payload_initial[:3])))
                for k in [0,1]:metrics['min_contact_axis_forces_N'][k]=min(metrics['min_contact_axis_forces_N'][k],abs(axisforces[k]))
                key='min_loaded_height_m' if t>=ph['load_observe_start'] else 'min_gravity_height_m';metrics[key]=min(metrics[key],float(pp[2]))
                closures.append([float(np.dot(np.asarray(poses[p][:3])-initial_poses[p][:3],grip))*sign for p,sign in [(left,1),(right,-1)]])
            if t>=ph['open_end']:release.append(float(payload_initial[2]-pp[2]))
            max_linear=max([max_linear]+[float(np.linalg.norm(v[:3])) for v in current_vel.values()]);max_angular=max([max_angular]+[float(np.linalg.norm(v[3:])) for v in current_vel.values()])
            if witness_at is None and (step+1)*dt>=.1:witness_at={'t':(step+1)*dt,'pose':witness.get('POSE').tolist(),'velocity':witness.get('VELOCITY').tolist()}
            if step%6==0 or step==n-1:trace.append({'t':(step+1)*dt,'poses':dict(poses,**{payload:pp.tolist()}),'joints':measures,'payload_contact_axis_forces_N':axisforces,'bilateral':bool(bilateral),'payload_support_force_N':support.tolist(),'payload_external_force_N':forces[payload].tolist(),'minimum_contact_separation_m':min(seps) if seps else None})
            if not finite or max_linear>100 or max_angular>1000:break
        metrics['bilateral_fraction']=metrics['bilateral_samples']/max(1,metrics['hold_samples']);metrics['minimum_each_finger_closure_m']=np.min(closures,axis=0).tolist() if closures else [0.,0.];metrics['release_drop_m']=max(release) if release else 0.
        result={'seed':seed,'completed_steps':step+1,'expected_steps':n,'backend':'ovphysx native USD CPU PhysX','contact_units':'Native0.4.13 step_sync matrix empirically calibrated as Ns; divided bydt to obtain N','only_dynamic_writes':['RIGID_BODY_WRENCH'],'contact_sensor_paths':contact.sensor_paths,'contact_filter_paths':contact.filter_paths,'native_mass':native_mass,'native_inertia':native_inertia,'gravity_witness':{'initial_pose':witness_initial.tolist(),'at_point_one_s':witness_at},'max_contact_penetration_m':max_pen,'max_linear_speed_m_s':max_linear,'max_angular_speed_rad_s':max_angular,'max_joint_closure_m':max_closure,'max_joint_axis_error_rad':max_axis,'finite_states_and_contacts':bool(finite),'joint_extrema':extrema,'metrics':metrics,'trace':trace}
        dump(output,result)
    finally:
        for x in reversed(extra):
            try:x.destroy() if hasattr(x,'destroy') else x.close()
            except Exception:pass
        for s in states.values():s.close()
        sdk.release()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
    try:run(json.loads(a.config.read_text()),a.seed,a.output,a.device)
    except Exception as exc:
        dump(a.output,{'seed':a.seed,'infrastructure_error':str(exc),'traceback':traceback.format_exc()});raise
