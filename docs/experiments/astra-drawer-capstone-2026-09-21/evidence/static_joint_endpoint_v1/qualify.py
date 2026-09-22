"""Synthetic static endpoint/collision-filter study. Never imports a source asset."""
import argparse,datetime,hashlib,json
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['build','run']);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();out=args.output
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()

def build():
    from pxr import Usd,UsdGeom,UsdPhysics,Sdf,Gf
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'prepared.json').exists()
    points=[(-.1,-.1,-.1),(.1,-.1,-.1),(.1,.1,-.1),(-.1,.1,-.1),(-.1,-.1,.1),(.1,-.1,.1),(.1,.1,.1),(-.1,.1,.1)]
    faces=[(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
    indices=[i for f in faces for tri in ((f[0],f[1],f[2]),(f[0],f[2],f[3])) for i in tri]
    records=[]
    for endpoint in ['static_xform','static_mesh']:
        for enabled in [False,True]:
            label=endpoint+('_collide' if enabled else '_disabled');path=out/(label+'.usda');assert not path.exists()
            stage=Usd.Stage.CreateNew(str(path));world=UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(world.GetPrim());UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Y')
            scene=UsdPhysics.Scene.Define(stage,'/World/PhysicsScene');scene.CreateGravityMagnitudeAttr(0.)
            UsdGeom.Xform.Define(stage,'/World/Static')
            moving=UsdGeom.Xform.Define(stage,'/World/Moving');moving.AddTranslateOp().Set(Gf.Vec3d(0,0,.15));body=UsdPhysics.RigidBodyAPI.Apply(moving.GetPrim());body.CreateKinematicEnabledAttr(False)
            mass=UsdPhysics.MassAPI.Apply(moving.GetPrim());mass.CreateMassAttr(1.);mass.CreateCenterOfMassAttr(Gf.Vec3f(0));mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1/150))
            for parent,approx in [('/World/Static','none'),('/World/Moving','convexHull')]:
                mesh=UsdGeom.Mesh.Define(stage,parent+'/Mesh');mesh.CreatePointsAttr(points);mesh.CreateFaceVertexCountsAttr([3]*12);mesh.CreateFaceVertexIndicesAttr(indices);mesh.CreateSubdivisionSchemeAttr('none');UsdPhysics.CollisionAPI.Apply(mesh.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(approx)
            moving.GetPrim().AddAppliedSchema('PhysxContactReportAPI');moving.GetPrim().CreateAttribute('physxContactReport:threshold',Sdf.ValueTypeNames.Float).Set(0.)
            joint=UsdPhysics.PrismaticJoint.Define(stage,'/World/Joint');target='/World/Static' if endpoint=='static_xform' else '/World/Static/Mesh';joint.CreateBody0Rel().SetTargets([target]);joint.CreateBody1Rel().SetTargets(['/World/Moving']);joint.CreateLocalPos0Attr(Gf.Vec3f(0,0,.15));joint.CreateLocalPos1Attr(Gf.Vec3f(0));joint.CreateAxisAttr('Z');joint.CreateLowerLimitAttr(-.25);joint.CreateUpperLimitAttr(.25);joint.CreateCollisionEnabledAttr(enabled)
            stage.GetRootLayer().Save()
            records.append({'label':label,'asset':path.name,'sha256':sha(path),'body0_target':target,'joint_collision_enabled':enabled,'source_geometry_sha256':hashlib.sha256(repr((points,indices)).encode()).hexdigest(),'initial_body_translation':[0,0,.15],'joint_world_frame0':[0,0,.15],'joint_world_frame1':[0,0,.15]})
    (out/'prepared.json').write_text(json.dumps({'schema_version':1,'created_utc':now(),'scope':'Four synthetic cubes only; all shapes/poses/mass/joint frames identical except explicit body0 target and joint collisionEnabled boolean.','script_sha256':sha(Path(__file__)),'cases':records},indent=2)+'\n')

def run():
    import numpy as np
    from ovphysx import PhysX,PhysXType,TensorType,SceneQueryGeometryType,SceneQueryMode
    prepared=json.loads((out/'prepared.json').read_text());assert not (out/'result.json').exists();results=[];dt=1/240
    for case in prepared['cases']:
        path=out/case['asset'];assert sha(path)==case['sha256'];p=PhysX(device='cpu');bindings=[];row={'label':case['label'],'asset_sha256':case['sha256'],'samples':[]}
        try:
            handle,op=p.add_usd(str(path));p.wait_op(op)
            def binding(kind):
                b=p.create_tensor_binding(pattern='/World/Moving',tensor_type=kind);bindings.append(b);return b,np.zeros(b.shape,np.float32)
            pb,pose=binding(TensorType.RIGID_BODY_POSE);vb,vel=binding(TensorType.RIGID_BODY_VELOCITY);fb,force=binding(TensorType.RIGID_BODY_FORCE);pb.read(pose)
            row['initial_pose']=pose.tolist();row['native_actor_presence']={target:bool(p.get_physx_ptr(target,PhysXType.ACTOR)) for target in ['/World/Static','/World/Static/Mesh','/World/Moving','/World/Moving/Mesh']};row['native_joint_present']=bool(p.get_physx_ptr('/World/Joint',PhysXType.JOINT))
            row['initial_overlap_query']=p.overlap(SceneQueryGeometryType.BOX,mode=SceneQueryMode.ALL,half_extent=[.01,.01,.01],position=[0,0,.1])
            cb=p.create_contact_binding(sensor_patterns=['/World/Moving'],filter_patterns=['/World/Static/Mesh'],filters_per_sensor=1,max_contact_data_count=64);bindings.append(cb);imp=np.zeros((cb.sensor_count,cb.filter_count,3),np.float32);row['contact_sensor_paths']=cb.sensor_paths;row['contact_filter_paths']=cb.filter_paths
            for i in range(8):
                force[:]=0;force[0,0]=10.;fb.write(force);p.step_sync(dt,i*dt);pb.read(pose);vb.read(vel);cb.read_force_matrix(imp)
                report=p.get_contact_report();contacts=[{'separation_m':float(report['points'][j].separation),'impulse_ns':list(report['points'][j].impulse),'normal':list(report['points'][j].normal)} for j in range(report['num_points'])]
                row['samples'].append({'step':i+1,'time_s':(i+1)*dt,'pose':pose[0].tolist(),'velocity':vel[0].tolist(),'world_force_n':force[0].tolist(),'pair_contact_impulse_ns':imp.tolist(),'contacts':contacts})
            row['finite']=all(np.isfinite(x['pose']).all() and np.isfinite(x['velocity']).all() for x in row['samples']);row['max_off_axis_m']=max(float(np.linalg.norm(x['pose'][:2])) for x in row['samples']);row['max_abs_z_displacement_m']=max(abs(x['pose'][2]-.15) for x in row['samples']);row['contact_point_samples']=sum(len(x['contacts']) for x in row['samples']);row['max_pair_contact_impulse_ns']=max(float(np.linalg.norm(x['pair_contact_impulse_ns'])) for x in row['samples']);row['asset_unchanged']=sha(path)==case['sha256']
        except Exception as e:
            row['error']=type(e).__name__+': '+str(e)
        finally:
            for b in reversed(bindings):b.destroy()
            p.release()
        results.append(row)
        (out/'partial.json').write_text(json.dumps(results,indent=2)+'\n')
    by={r['label']:r for r in results}
    positive=all(by[k].get('contact_point_samples',0)>0 for k in ['static_xform_collide','static_mesh_collide'])
    mesh=by['static_mesh_disabled'];xform=by['static_xform_disabled']
    supported=positive and mesh.get('contact_point_samples',-1)==0 and mesh.get('max_pair_contact_impulse_ns',-1)==0 and mesh.get('native_joint_present',False) and mesh.get('max_off_axis_m',1)<1e-4
    result={'schema_version':1,'completed_utc':now(),'script_sha256':sha(Path(__file__)),'prepared_sha256':sha(out/'prepared.json'),'backend':'ovphysx0.4.13 CPU','native_steps':sum(len(x['samples']) for x in results),'native_processes':1,'source_asset_or_current_run_modified':False,'results':results,'positive_collision_controls_pass':positive,'static_mesh_endpoint_pair_suppression_supported_for_fixture':supported,'static_xform_disabled_still_reports_contacts':xform.get('contact_point_samples',0)>0,'scope':'Synthetic endpoint semantics only. Actor presence + actual joint/contacts/locked-axis response are observed; native joint actor pointer pair is not directly introspected. No source repair or task-success claim.'}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='results'}))

if __name__=='__main__':globals()[args.phase]()
