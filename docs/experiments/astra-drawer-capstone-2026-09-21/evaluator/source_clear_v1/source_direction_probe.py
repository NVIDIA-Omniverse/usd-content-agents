from drawer_evaluate import gltf_reference, HERE, write_json
import numpy as np
tris=gltf_reference()['drawer_cabinet']
def ray(o,d):
    e1=tris[:,1]-tris[:,0];e2=tris[:,2]-tris[:,0];h=np.cross(d,e2);a=np.sum(e1*h,1);f=np.zeros_like(a);valid=np.abs(a)>1e-9;f[valid]=1/a[valid];s=o-tris[:,0];u=f*np.sum(s*h,1);q=np.cross(s,e1);v=f*np.sum(d*q,1);t=f*np.sum(e2*q,1);ok=valid&(u>=0)&(v>=0)&(u+v<=1)&(t>1e-6)
    return sorted(t[ok].tolist())
probes=[]
for x in [-0.32,0.0,0.32]:
    for y in [1.01,1.03,1.05]:
        p=np.array([x,y,0.0]);probes.append({'origin':p.tolist(),'positive_z_hits_m':ray(p,np.array([0,0,1])),'negative_z_hits_m':ray(p,np.array([0,0,-1]))})
r={'method':'Moller-Trumbore ray tests on untouched original cabinet triangle mesh only, no colliders or converted submission','probes':probes,'positive_Z_is_open':all(not x['positive_z_hits_m'] and x['negative_z_hits_m'] for x in probes)}
write_json(HERE/'source_direction_evidence.json',r);print(r)
