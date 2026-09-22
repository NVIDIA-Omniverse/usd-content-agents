"""Isolated native solver. No pxr imports, no submitted code, and no pose writes."""
import hashlib, json, math, pathlib, sys, traceback
import numpy as np
from ovphysx import PhysX, TensorType, SceneQueryGeometryType, SceneQueryMode

def solve(req):
    s=req['spec'];out=pathlib.Path(req['output']);out.mkdir(exist_ok=True)
    result={'status':'INCONCLUSIVE','pass':False,'seed':req['seed'],'backend':'ovphysx native USD / CPU PhysX','errors':[]};p=None;bindings=[]
    try:
        if hashlib.sha256(pathlib.Path(req['scene_usd']).read_bytes()).hexdigest()!=req['scene_sha256']:raise RuntimeError('Evaluation scene changed after request authoring')
        result['scene_sha256']=req['scene_sha256']
        p=PhysX(device='cpu');handle,op=p.add_usd(req['scene_usd']);p.wait_op(op)
        def bind(path,typ):
            b=p.create_tensor_binding(pattern=path,tensor_type=typ);bindings.append(b)
            if not b.shape or b.shape[0]!=1:raise RuntimeError('Expected exactly one body binding for '+path)
            return b,np.zeros(b.shape,np.float32)
        dp,dpos=bind(req['drawer_body'],TensorType.RIGID_BODY_POSE);dv,dvel=bind(req['drawer_body'],TensorType.RIGID_BODY_VELOCITY);df,force=bind(req['drawer_body'],TensorType.RIGID_BODY_FORCE)
        pp,ppos=bind(req['payload_body'],TensorType.RIGID_BODY_POSE);pv,pvel=bind(req['payload_body'],TensorType.RIGID_BODY_VELOCITY)
        mb,mass=bind(req['drawer_body'],TensorType.RIGID_BODY_MASS);mb.read(mass)
        ib,inertia=bind(req['drawer_body'],TensorType.RIGID_BODY_INERTIA);ib.read(inertia)
        if not np.isfinite(mass).all() or mass.min()<=0 or not np.isfinite(inertia).all():raise RuntimeError('Native mass/inertia invalid')
        cb=p.create_contact_binding(sensor_patterns=[req['payload_body']],filter_patterns=[req['drawer_body']],filters_per_sensor=1,max_contact_data_count=128);bindings.append(cb)
        cf=np.zeros((cb.sensor_count,cb.filter_count,3),np.float32)
        if cb.sensor_count!=1 or cb.filter_count!=1:raise RuntimeError('Payload contact binding unresolved')
        result['resolved_contact_sensor_paths']=cb.sensor_paths;result['resolved_contact_filter_paths']=cb.filter_paths
        interior=[];iq=s['interior_queries']
        for x in iq['x_m']:
            for z in iq['z_m']:
                hits=p.overlap(SceneQueryGeometryType.SPHERE,mode=SceneQueryMode.ALL,radius=iq['sphere_radius_m'],position=[x,iq['y_m'],z])
                interior.append({'position':[x,iq['y_m'],z],'overlap_count':len(hits),'collision_ids':[int(h['collision']) for h in hits]})
        result['interior_queries']=interior
        if any(x['overlap_count'] for x in interior):result['errors'].append('Collider fills a declared drawer/cabinet interior point')
        dp.read(dpos);pp.read(ppos);initial=dpos[0].astype(float).copy();authored=np.array(req['initial_drawer_pose']);initialerror=float(np.linalg.norm(initial[:3]-authored[:3]));result['import_pose_error_m']=initialerror
        if initialerror>1e-3:raise RuntimeError('Native initial body pose differs from authored pose')
        durations=s['phase_durations_s'];settle=durations['settle'];open_end=settle+durations['open'];hold_end=open_end+durations['hold_open'];close_end=hold_end+durations['close'];total=close_end+durations['hold_closed'];dt=s['dt_s'];n=int(round(total/dt));rng=np.random.default_rng(req['seed']);lim=s['limits'];c=s['controller']
        metrics={'max_q_m':-1e10,'max_penetration_m':0.0,'max_off_axis_m':0.0,'max_rotation_deg':0.0,'max_linear_speed_m_s':0.0,'max_angular_speed_rad_s':0.0,'payload_contact_samples':0,'payload_retained':True,'max_force_n':0.0,'initial_drift_m':0.0};open_hold=[];closed_hold=[];bad=False
        with (out/'trace.jsonl').open('w') as trace:
            for i in range(n):
                t=i*dt;dp.read(dpos);dv.read(dvel)
                q=float(dpos[0,2]-initial[2]);qv=float(dvel[0,2])
                target=target_v=0.0;phase='settle'
                if settle<=t<open_end:
                    u=(t-settle)/durations['open'];target=s['opening_command_m']*(.5-.5*math.cos(math.pi*u));target_v=s['opening_command_m']*.5*math.pi/durations['open']*math.sin(math.pi*u);phase='open'
                elif open_end<=t<hold_end:target=s['opening_command_m'];phase='hold_open'
                elif hold_end<=t<close_end:
                    u=(t-hold_end)/durations['close'];target=s['opening_command_m']*(.5+.5*math.cos(math.pi*u));target_v=-s['opening_command_m']*.5*math.pi/durations['close']*math.sin(math.pi*u);phase='close'
                elif t>=close_end:phase='hold_closed'
                fz=0.0 if t<settle else float(np.clip(c['kp_n_m']*(target-q)+c['kd_ns_m']*(target_v-qv)+rng.uniform(-c['force_jitter_n'],c['force_jitter_n']),-c['force_limit_n'],c['force_limit_n']))
                force[:]=0;force[0,2]=fz;df.write(force);p.step_sync(dt,t)
                dp.read(dpos);dv.read(dvel);pp.read(ppos);pv.read(pvel);cb.read_force_matrix(cf)
                report=p.get_contact_report();contacts=[]
                for j in range(report['num_points']):
                    contact=report['points'][j];contacts.append({'p':list(contact.position),'normal':list(contact.normal),'impulse':list(contact.impulse),'separation':float(contact.separation)})
                finite=all(np.isfinite(x).all() for x in [dpos,dvel,ppos,pvel,cf]) and all(math.isfinite(x['separation']) for x in contacts)
                if not finite:result['errors'].append('Nonfinite solver state');bad=True;break
                q=float(dpos[0,2]-initial[2]);off=float(np.linalg.norm(dpos[0,:2]-initial[:2]));dot=abs(float(np.dot(dpos[0,3:7],initial[3:7])));rot=math.degrees(2*math.acos(np.clip(dot,0,1)))
                speed=max(np.linalg.norm(dvel[0,:3]),np.linalg.norm(pvel[0,:3]));spin=max(np.linalg.norm(dvel[0,3:]),np.linalg.norm(pvel[0,3:]));penetration=max([0.0]+[-x['separation'] for x in contacts]);contact_force=float(np.linalg.norm(cf))
                metrics['max_q_m']=max(metrics['max_q_m'],q);metrics['max_penetration_m']=max(metrics['max_penetration_m'],penetration);metrics['max_off_axis_m']=max(metrics['max_off_axis_m'],off);metrics['max_rotation_deg']=max(metrics['max_rotation_deg'],rot);metrics['max_linear_speed_m_s']=max(metrics['max_linear_speed_m_s'],float(speed));metrics['max_angular_speed_rad_s']=max(metrics['max_angular_speed_rad_s'],float(spin));metrics['max_force_n']=max(metrics['max_force_n'],abs(fz))
                if t<settle:metrics['initial_drift_m']=max(metrics['initial_drift_m'],abs(q))
                relative=ppos[0,:3].astype(float)-(dpos[0,:3]-initial[:3]);payload=s['payload']
                retained=bool((relative>=payload['retained_center_min_relative_to_drawer_translation_m']).all() and (relative<=payload['retained_center_max_relative_to_drawer_translation_m']).all())
                if t>=settle:
                    metrics['payload_retained'] &= retained
                    if contact_force>0.01:metrics['payload_contact_samples']+=1
                if hold_end-.5<=t<hold_end:open_hold.append(q)
                if total-.5<=t:closed_hold.append((abs(q),float(np.linalg.norm(dvel[0,:3]))))
                trace.write(json.dumps({'step':i,'time_s':(i+1)*dt,'phase':phase,'target_q_m':target,'applied_force_world_n':force[0].tolist(),'q_m':q,'drawer_pose':dpos[0].tolist(),'drawer_velocity':dvel[0].tolist(),'payload_pose':ppos[0].tolist(),'payload_velocity':pvel[0].tolist(),'payload_drawer_contact_force_n':cf.tolist(),'contacts':contacts},allow_nan=False)+'\n')
                if q<lim['min_drawer_position_m'] or q>lim['max_drawer_position_m'] or speed>lim['max_linear_speed_m_s'] or spin>lim['max_angular_speed_rad_s']:
                    result['errors'].append('Explosive/excessive motion or joint travel escape');bad=True;break
        metrics['min_open_hold_q_m']=min(open_hold) if open_hold else None;metrics['max_closed_hold_error_m']=max(x[0] for x in closed_hold) if closed_hold else None;metrics['max_closed_hold_speed_m_s']=max(x[1] for x in closed_hold) if closed_hold else None
        conditions={'opening':bool(open_hold and min(open_hold)>=s['opening_required_m']),'closing':bool(closed_hold and max(x[0] for x in closed_hold)<=s['closure_tolerance_m']),'closed_settled':bool(closed_hold and max(x[1] for x in closed_hold)<=lim['max_final_speed_m_s']),'payload_retained':metrics['payload_retained'],'payload_contact':metrics['payload_contact_samples']>=lim['min_payload_contact_samples'],'penetration':metrics['max_penetration_m']<=lim['max_contact_penetration_m'],'off_axis':metrics['max_off_axis_m']<=lim['max_drawer_off_axis_m'],'rotation':metrics['max_rotation_deg']<=lim['max_drawer_rotation_deg'],'initial_drift':metrics['initial_drift_m']<=lim['max_initial_drawer_drift_m'],'force_bounded':metrics['max_force_n']<=c['force_limit_n']+1e-6,'finite_and_stable':not bad,'interior_clear':not any(x['overlap_count'] for x in interior)}
        result.update(metrics=metrics,checks=conditions,pass_rule='all checks',status='PASS' if all(conditions.values()) else 'FAIL');result['pass']=result['status']=='PASS'
    except Exception as e:
        result['status']='INCONCLUSIVE';result['errors'].append(type(e).__name__+': '+str(e));result['traceback']=traceback.format_exc()
    finally:
        for b in reversed(bindings):
            try:b.destroy()
            except Exception:pass
        if p:
            try:p.release()
            except Exception:pass
    (out/'trial_report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result
if __name__=='__main__':
    req=json.loads(pathlib.Path(sys.argv[1]).read_text());r=solve(req);print(json.dumps({'status':r['status'],'errors':r['errors']}));sys.exit(0 if r['pass'] else (2 if r['status']=='INCONCLUSIVE' else 1))
