"""Freeze an original-source bed patch; never inspect an author submission."""
import argparse,json
from pathlib import Path
import numpy as np
from common import dump,sha
HERE=Path(__file__).resolve().parent

def on_patch(points,triangles):
    """Projected triangle union membership for explicit source-surface probes."""
    a=triangles[:,0,:2];b=triangles[:,1,:2];c=triangles[:,2,:2]
    v0=b-a;v1=c-a;det=v0[:,0]*v1[:,1]-v0[:,1]*v1[:,0];ok=np.abs(det)>1e-16
    answers=[]
    for p in points:
        v=p[:2]-a;u=np.zeros(len(a));w=u.copy();u[ok]=(v[ok,0]*v1[ok,1]-v[ok,1]*v1[ok,0])/det[ok];w[ok]=(v0[ok,0]*v[ok,1]-v0[ok,1]*v[ok,0])/det[ok]
        answers.append(bool(np.any(ok&(u>=-1e-8)&(w>=-1e-8)&(u+w<=1+1e-8))))
    return answers

def triangle_box_intersection(tris,center,half):
    """Separating-axis test; all13 standard triangle/AABB candidate axes."""
    t=tris-center;valid=np.all(np.min(t,axis=1)<=half+1e-10,axis=1)&np.all(np.max(t,axis=1)>=-half-1e-10,axis=1);t=t[valid]
    if not len(t):return False
    edges=np.roll(t,-1,axis=1)-t;normal=np.cross(edges[:,0],edges[:,1]);axes=[normal]
    for i in range(3):
        for axis in np.eye(3):axes.append(np.cross(edges[:,i],axis))
    active=np.ones(len(t),bool)
    for axis in axes:
        projected=np.einsum('nkj,nj->nk',t,axis);radius=np.sum(np.abs(axis)*half,axis=1);active &= (projected.min(axis=1)<=radius+1e-12)&(projected.max(axis=1)>=-radius-1e-12)
        if not active.any():return False
    return bool(active.any())

def contains_point(tris,p):
    direction=np.array([1.,0,0]);e1=tris[:,1]-tris[:,0];e2=tris[:,2]-tris[:,0];h=np.cross(direction,e2);a=np.sum(e1*h,axis=1);valid=np.abs(a)>1e-14;f=np.zeros_like(a);f[valid]=1/a[valid];s=p-tris[:,0];u=f*np.sum(s*h,axis=1);q=np.cross(s,e1);v=f*np.sum(direction*q,axis=1);distance=f*np.sum(e2*q,axis=1);hits=np.sort(distance[valid&(u>=-1e-9)&(v>=-1e-9)&(u+v<=1+1e-9)&(distance>1e-8)])
    unique=hits[np.r_[True,np.diff(hits)>1e-7]] if len(hits) else []
    return len(unique)%2==1

def find_placement(inventory_path,bed_ids,output):
    inventory_path=Path(inventory_path);inv=json.loads(inventory_path.read_text());root=inventory_path.parent;c=json.loads((HERE/'case09_payload_contract.json').read_text());parts={p['source_id']:p for p in inv['parts']};cache={}
    if not inv.get('source_coverage_complete'):raise RuntimeError('Original source coverage incomplete; cannot certify independent placement')
    def load(ident):
        if ident not in cache:
            p=parts[ident]
            if p['frame']!='source_assembly_world':raise RuntimeError('Placement requires original complete source-assembly coordinates')
            f=root/p['geometry_file']
            if sha(f)!=p['geometry_sha256']:raise RuntimeError('Source reference geometry hash changed')
            d=np.load(f);cache[ident]=d['vertices'][d['faces']]
        return cache[ident]
    bed=np.concatenate([load(ident) for ident in bed_ids]);edges=bed[:,1:]-bed[:,0,None];cross=np.cross(edges[:,0],edges[:,1]);norm=np.linalg.norm(cross,axis=1);horizontal=(np.abs(cross[:,2])>=.999999*norm)&(norm>1e-14)&(np.ptp(bed[:,:,2],axis=1)<=c['source_planarity_tolerance_m']);flat=bed[horizontal];areas=norm[horizontal]/2
    if not len(flat):raise RuntimeError('Trusted source bed has no horizontal patch in sourceZ-up frame')
    tolerance=c['source_planarity_tolerance_m'];bins=np.round(flat[:,:,2].mean(axis=1)/tolerance).astype(np.int64);totals={int(k):float(areas[bins==k].sum()) for k in np.unique(bins)};eligible=[k for k,total in totals.items() if total>=(c['size_m'][0]+2*c['source_patch_edge_margin_m'])**2];eligible.sort(reverse=True)
    rejected=[];chosen=None
    for plane in eligible:
        tris=flat[bins==plane];z=float(tris[:,:,2].mean());lo=tris[:,:,:2].min(axis=(0,1));hi=tris[:,:,:2].max(axis=(0,1));middle=.5*(lo+hi)
        xs=np.linspace(lo[0]+.02,hi[0]-.02,7) if hi[0]-lo[0]>.04 else [middle[0]];ys=np.linspace(lo[1]+.02,hi[1]-.02,7) if hi[1]-lo[1]>.04 else [middle[1]]
        candidates=[middle]+[np.array([x,y]) for x in xs for y in ys];candidates.sort(key=lambda p:float(np.linalg.norm(p-middle)))
        for xy in candidates:
            half=np.array(c['size_m'])/2;radius=half[:2]+c['source_patch_edge_margin_m'];points=[np.array([xy[0]+dx,xy[1]+dy,z]) for dx in np.linspace(-radius[0],radius[0],c['source_patch_grid_points_per_axis']) for dy in np.linspace(-radius[1],radius[1],c['source_patch_grid_points_per_axis'])]
            support=on_patch(points,tris)
            if not all(support):continue
            center=np.array([*xy,z+half[2]+c['initial_surface_clearance_m']]);obstacles=[]
            for ident,part in parts.items():
                b=np.array(part['bounds_m'])
                if np.any(b[1]<center-half) or np.any(b[0]>center+half):continue
                shape=load(ident)
                if triangle_box_intersection(shape,center,half) or (np.all(center>=b[0]) and np.all(center<=b[1]) and contains_point(shape,center)):
                    obstacles.append(ident);break
            if obstacles:
                if len(rejected)<10:rejected.append({'source_center_m':center.tolist(),'obstacle_source_ids':obstacles})
                continue
            chosen={'source_center_m':center.tolist(),'source_top_normal':[0,0,1],'source_surface_height_m':z,'source_support_probes_m':[p.tolist() for p in points],'support_probe_pass':support,'source_plane_area_m2':totals[plane],'source_plane_xy_bounds_m':[lo.tolist(),hi.tolist()],'payload_source_geometry_clear':True};break
        if chosen:break
    if not chosen:raise RuntimeError('No independently clear original bed patch found; do not use a submitted placement hint')
    result={'case_id':inv['case_id'],'source_inventory_sha256':sha(inventory_path),'bed_source_ids':bed_ids,'payload_spec':c,'placement':chosen,'rejected_candidates':rejected,'method':'Original source only: horizontal source-plane area,25 footprint support probes, triangle/AABB separating-axis clearance and closed-mesh center containment against every overlapping original part','source_geometry_hashes_consulted':{ident:parts[ident]['geometry_sha256'] for ident in cache},'benchmark_arm_outputs_read':False};dump(output,result);print(json.dumps({'placement':chosen,'bed_source_ids':bed_ids,'consulted_parts':len(cache)},indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);p.add_argument('--bed-source-id',action='append',required=True);p.add_argument('--output',type=Path,default=HERE/'case09_placement.json');a=p.parse_args();find_placement(a.inventory,a.bed_source_id,a.output)
