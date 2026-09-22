"""Independent read-only comparison of original drawer GLTF arrays to native Geometry output."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom
root=Path('/opt/astra-content-value-20260921')
source=root/'assets/01_drawer/source/drawer_cabinet_1k.gltf'
parser=argparse.ArgumentParser();parser.add_argument('--run',default='drawer_geometry_02');parser.add_argument('--candidate',type=Path);args=parser.parse_args()
run=root/'capstone/runs'/args.run
doc=json.loads(source.read_text());buffers=[(source.parent/x['uri']).read_bytes() for x in doc['buffers']]
def accessor(i):
 a=doc['accessors'][i];v=doc['bufferViews'][a['bufferView']];dt=np.dtype({5123:'<u2',5125:'<u4',5126:'<f4'}[a['componentType']]);w={'SCALAR':1,'VEC3':3}[a['type']]
 return np.ndarray((a['count'],w),dt,buffer=buffers[v['buffer']],offset=v.get('byteOffset',0)+a.get('byteOffset',0),strides=(v.get('byteStride',dt.itemsize*w),dt.itemsize)).copy()
candidate=args.candidate or run/'source_preserved.geometry.usdc'
stage=Usd.Stage.Open(str(candidate));cache=UsdGeom.XformCache();rows=[]
for prim in stage.Traverse():
 if not prim.IsA(UsdGeom.Mesh):continue
 nodeid=prim.GetCustomDataByKey('sourceGltfNode');node=doc['nodes'][nodeid]
 assert not any(k in node for k in ['matrix','translation','rotation','scale'])
 src=doc['meshes'][node['mesh']]['primitives'][0];v=accessor(src['attributes']['POSITION']);f=accessor(src['indices']).reshape(-1)
 mesh=UsdGeom.Mesh(prim);points=np.asarray(mesh.GetPointsAttr().Get());indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get());counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get())
 checks={'exact_points':np.array_equal(v,points),'exact_original_indices':np.array_equal(f,indices),'all_triangles':bool(np.all(counts==3) and len(counts)*3==len(f)),'original_world_transform':np.array_equal(np.asarray(cache.GetLocalToWorldTransform(prim)),np.eye(4))}
 rows.append({'source_node':node['name'],'usd_path':str(prim.GetPath()),'points':len(v),'triangles':len(f)//3,'checks':checks})
result={'scope':'Original source and new native capstone Geometry output only; no baseline/author artifacts used. No physical or visual-render acceptance inferred.', 'candidate':str(candidate),'candidate_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),'source_gltf_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_bin_sha256':hashlib.sha256(buffers[0]).hexdigest(),'source_mesh_count':len(doc['meshes']),'output_mesh_count':len(rows),'meters_per_unit':UsdGeom.GetStageMetersPerUnit(stage),'up_axis':str(UsdGeom.GetStageUpAxis(stage)),'parts':rows,'passed':len(rows)==len(doc['meshes'])==5 and UsdGeom.GetStageMetersPerUnit(stage)==1 and str(UsdGeom.GetStageUpAxis(stage))=='Y' and all(all(p['checks'].values()) for p in rows)}
(root/'capstone/evidence'/('original_source_'+args.run+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
