"""Independent source geometry inventory and surface preservation checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from common import check, dump, rigid_matrix, sha, transform

def mesh_record(vertices, faces):
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    edges = np.sort(np.linalg.norm(mesh.triangles - np.roll(mesh.triangles, 1, axis=1), axis=2), axis=1)
    rows = np.rint(edges / 1e-7).astype('<i8')
    rows = rows[np.lexsort(rows.T[::-1])]
    return {'vertices': int(len(mesh.vertices)), 'triangles': int(len(mesh.faces)),
            'bounds_m': mesh.bounds.tolist(), 'area_m2': float(mesh.area),
            'abs_volume_m3': abs(float(mesh.volume)), 'watertight': bool(mesh.is_watertight),
            'triangle_edge_multiset_sha256_0p1um': hashlib.sha256(rows.tobytes()).hexdigest()}

def occ_mesh(shape, unit):
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.TopLoc import TopLoc_Location
    # Absolute 0.025 mm chord tolerance, independent of either authoring arm.
    BRepMesh_IncrementalMesh(shape, 0.025 / (unit / 0.001), False, 0.1, True)
    vertices, faces = [], []
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        face = TopoDS.Face_s(ex.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            offset = len(vertices)
            tr = loc.Transformation()
            for i in range(1, tri.NbNodes()+1):
                p = tri.Node(i).Transformed(tr)
                vertices.append([p.X()*unit, p.Y()*unit, p.Z()*unit])
            for i in range(1, tri.NbTriangles()+1):
                a, b, c = tri.Triangle(i).Get()
                if face.Orientation() == TopAbs_REVERSED:
                    b, c = c, b
                faces.append([offset+a-1, offset+b-1, offset+c-1])
        ex.Next()
    if not faces:
        raise ValueError('source shape has no tessellatable faces')
    return np.asarray(vertices), np.asarray(faces)

def load_step(path, unit):
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TDocStd import TDocStd_Document
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    from OCP.TDF import TDF_LabelSequence
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.TDataStd import TDataStd_Name
    doc = TDocStd_Document(TCollection_ExtendedString('evaluator_source'))
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    previous = Path.cwd()
    try:
        os.chdir(path.parent)
        if reader.ReadFile(str(path)) != IFSelect_RetDone or not reader.Transfer(doc):
            raise ValueError('independent STEPCAF import failed')
    finally:
        os.chdir(previous)
    tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    labels = TDF_LabelSequence()
    tool.GetFreeShapes(labels)
    i = 0
    for k in range(1, labels.Length()+1):
        root = tool.GetShape_s(labels.Value(k))
        exp = TopExp_Explorer(root, TopAbs_SOLID)
        while exp.More():
            vertices, faces = occ_mesh(exp.Current(), unit)
            label=tool.FindShape(exp.Current(),False);name=TDataStd_Name();label_text='unnamed'
            if not label.IsNull() and label.FindAttribute(TDataStd_Name.GetID_s(),name):label_text=name.Get().ToExtString()
            yield f'solid:{i:04d}:{label_text}', vertices, faces, 'source_assembly_world'
            i += 1
            exp.Next()
    if i == 0:
        raise ValueError('STEP imported zero solids (external dependencies may be unresolved)')

def load_fcstd(path, unit):
    """Read saved assembly shape instances, never recompute or execute source code.

    Covers Assembly4 external links and cached A2plus shapes. Original placement
    math is not certified: saved component shapes are compared by rigid alignment.
    """
    from OCP.BRep import BRep_Builder
    from OCP.BRepTools import BRepTools
    from OCP.TopoDS import TopoDS_Shape
    import io
    cache = {}
    def document(docpath):
        if docpath not in cache:
            with zipfile.ZipFile(docpath) as z:
                x=ET.fromstring(z.read('Document.xml'))
                types={o.get('name'):o.get('type') for o in x.find('Objects') if o.get('type')}
                props={o.get('name'):{p.get('name'):p for p in o.find('Properties')} for o in x.find('ObjectData')}
                gui=ET.fromstring(z.read('GuiDocument.xml'))
                visible={o.get('name') for o in gui.find('ViewProviderData') if o.find("Properties/Property[@name='Visibility']/Bool") is not None and o.find("Properties/Property[@name='Visibility']/Bool").get('value')=='true'}
                cache[docpath]=(types,props,visible)
        return cache[docpath]
    def walk(docpath,name,prefix,ancestors):
        key=(str(docpath),name)
        if key in ancestors:raise ValueError('cyclic FCStd reference')
        types,props,visible=document(docpath);p=props[name];typ=types[name];prefix=prefix+'/'+name
        if typ=='App::Link':
            link=p['LinkedObject'].find('XLink')
            target=(docpath.parent/link.get('file')).resolve() if link.get('file') else docpath
            if not target.is_file() and target.parent.is_dir():
                matches=[f for f in target.parent.iterdir() if f.name.lower()==target.name.lower()]
                if len(matches)==1:target=matches[0]
            if not target.is_file():raise ValueError('Missing original FCStd dependency: '+str(target))
            yield from walk(target,link.get('name'),prefix,ancestors+[key])
        elif typ=='App::Part':
            children=p['Group'].findall('.//Link')
            array_templates={props[c.get('value')]['Base'].find('Link').get('value') for c in children if 'Base' in props[c.get('value')] and 'Count' in props[c.get('value')]}
            for child in children:
                if child.get('value') in array_templates:continue
                yield from walk(docpath,child.get('value'),prefix,ancestors+[key])
        elif typ in ('PartDesign::Body','Part::Feature','Part::FeaturePython','Part::Cylinder','Part::Torus','Part::Compound') or ('Shape' in p and p['Shape'].find('Part') is not None):
            if 'Shape' not in p:raise ValueError('Assembly geometry object has no cached Shape: '+name)
            part_element=p['Shape'].find('Part')
            if part_element is None and 'Base' in p and 'Count' in p:
                base=p['Base'].find('Link').get('value');count=int(p['Count'].find('Integer').get('value'))
                if not 0<count<=1000:raise ValueError('Unsupported array size')
                for i in range(count):yield from walk(docpath,base,prefix+f'[{i}]',ancestors+[key])
                return
            if part_element is None:raise ValueError('Missing saved final BREP for '+name)
            part=part_element.get('file');shape=TopoDS_Shape()
            with zipfile.ZipFile(docpath) as z:BRepTools.Read_s(shape,io.BytesIO(z.read(part)),BRep_Builder())
            vertices,faces=occ_mesh(shape,unit)
            yield prefix,vertices,faces,'part_local_only'
        elif typ not in ('App::DocumentObjectGroup','App::FeaturePython','PartDesign::CoordinateSystem','PartDesign::Line','PartDesign::Plane','App::Origin','Sketcher::SketchObject','App::Line','App::Plane'):
            raise ValueError('Unsupported terminal assembly object type '+typ)
    types,props,visible=document(path)
    if 'Assembly' in props: roots=['Assembly']
    elif 'Model' in props and types['Model']=='App::Part': roots=['Model']
    elif not any(t=='App::Part' for t in types.values()):
        roots=[n for n,t in types.items() if t in ('Part::Feature','Part::FeaturePython') and n in visible and 'Shape' in props[n]]
    else:raise ValueError('No independently identifiable FCStd assembly root')
    if not roots:raise ValueError('No visible saved assembly shape instances')
    for root in roots:yield from walk(path,root,'',[])

def sources(path, unit):
    suffix = path.suffix.lower()
    if suffix in ('.step', '.stp'):
        yield from load_step(path, unit)
    elif suffix == '.fcstd':
        yield from load_fcstd(path, unit)
    elif suffix in ('.glb', '.gltf'):
        scene = trimesh.load(path, force='scene', process=False)
        for node in sorted(scene.graph.nodes_geometry):
            matrix, geom = scene.graph[node]
            mesh = scene.geometry[geom]
            yield 'node:' + node, transform(mesh.vertices, matrix) * unit, mesh.faces, 'source_assembly_world'
    elif suffix == '.stl':
        mesh = trimesh.load(path, force='mesh', process=False)
        yield 'part:' + path.stem, np.asarray(mesh.vertices) * unit, mesh.faces, 'part_local_only'
    else:
        raise ValueError('unsupported independent source importer: ' + suffix)

def freeze(asset_root, source_paths, unit, output, case_id, excluded_ids=()):
    output.mkdir(parents=True, exist_ok=True)
    records, originals = [], []
    for source in source_paths:
        source = (asset_root / source).resolve()
        source.relative_to(asset_root.resolve())
        originals.append({'path': str(source.relative_to(asset_root)), 'sha256': sha(source)})
        for name, vertices, faces, frame in sources(source, unit):
            ident = str(source.relative_to(asset_root)) + '#' + name
            filename = f'geometry_{len(records):04d}.npz'
            np.savez_compressed(output / filename, vertices=vertices, faces=faces)
            records.append({'source_id': ident, 'source_file': str(source.relative_to(asset_root)),
                            'frame': frame, 'geometry_file': filename, 'geometry_sha256': sha(output / filename),
                            'required': ident not in excluded_ids, **mesh_record(vertices, faces)})
    complete = bool(records)
    inventory = {'schema_version': 1, 'case_id': case_id, 'source_root': str(asset_root),
                 'source_files': originals, 'length_scale_to_m': unit, 'parts': records,
                 'source_coverage_complete': complete,
                 'coverage_note': 'All selected source meshes/solids or saved Assembly4 terminal shape instances retained. Selection must be frozen by evaluator owner before arms. Part-local sources preserve shape and instance coverage; original assembly placements are not certified for STL/FCStd.',
                 'importer_versions': {k: __import__('importlib.metadata', fromlist=['version']).version(k) for k in ('numpy', 'trimesh')},
                 'excluded_ids': list(excluded_ids)}
    try:
        inventory['importer_versions']['cadquery-ocp'] = __import__('importlib.metadata', fromlist=['version']).version('cadquery-ocp')
    except Exception:
        pass
    dump(output / 'source_inventory.json', inventory)
    return inventory

from v2_geometry import surface_compare

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--source', type=Path, action='append', required=True)
    parser.add_argument('--unit', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--exclude-id', action='append', default=[])
    args = parser.parse_args()
    print(json.dumps(freeze(args.asset_root,args.source,args.unit,args.output,args.case,args.exclude_id),indent=2))
