import io,zipfile,xml.etree.ElementTree as ET
from pathlib import Path
from OCP.BRep import BRep_Builder
from OCP.BRepTools import BRepTools
from OCP.TopoDS import TopoDS_Shape
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID,TopAbs_SHELL,TopAbs_FACE,TopAbs_COMPSOLID,TopAbs_COMPOUND
p=next(Path('/opt/astra-content-value-20260921/assets/10_complex/source').glob('*/Parts/FreeCAD files/Assembly/dextra.fcstd'))
for path in [p,p.parent/'finger_module.fcstd']:
 with zipfile.ZipFile(path) as z:
  for obj in ET.fromstring(z.read('Document.xml')).find('ObjectData'):
   part=obj.find("Properties/Property[@name='Shape']/Part")
   if part is None:continue
   shape=TopoDS_Shape();BRepTools.Read_s(shape,io.BytesIO(z.read(part.get('file'))),BRep_Builder());counts={}
   for kind in [TopAbs_SOLID,TopAbs_SHELL,TopAbs_FACE,TopAbs_COMPOUND]:
    ex=TopExp_Explorer(shape,kind);n=0
    while ex.More():n+=1;ex.Next()
    counts[str(kind)]=n
   print(path.name,obj.get('name'),shape.ShapeType(),counts,flush=True)
