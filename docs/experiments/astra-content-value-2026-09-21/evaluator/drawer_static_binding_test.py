"""Synthetic ancestry mutations; no source-cabinet authoring or scored run."""
import copy,json
from pxr import UsdGeom,UsdPhysics,Gf
import drawer_evaluate as E
from drawer_test_fixtures import fixture
s,b=fixture('static_binding_positive')
b['source_components']={'drawer_cabinet':['/World/Cabinet/TinyAnchor'],'drawer_cabinet_drawer_01':['/World/Slider/Floor'],'drawer_cabinet_drawer_02':['/World/Cabinet/TinyAnchor'],'drawer_cabinet_drawer_03':['/World/Cabinet/TinyAnchor'],'drawer_cabinet_drawer_04':['/World/Cabinet/TinyAnchor']}
checks=[]
def record(name,passed,errors):checks.append({'test':name,'pass':passed,'errors':errors})
e=E.validate_component_bindings(s,b);record('static_source_parenting_positive',not e,e)
for name in ['drawer_cabinet','drawer_cabinet_drawer_02','drawer_cabinet_drawer_03','drawer_cabinet_drawer_04']:
    mutant=copy.deepcopy(b);mutant['source_components'][name]=['/World/Slider/Floor'];e=E.validate_component_bindings(s,mutant);record(name+'_moved_with_target_rejected',any('inherits moving' in x for x in e),e)
mutant=copy.deepcopy(b);mutant['cabinet_body']='/World/NonCabinet';e=E.validate_component_bindings(s,mutant);record('cabinet_outside_declared_static_root_rejected',any('under cabinet_body' in x for x in e),e)
container=UsdGeom.Xform.Define(s,'/World/Cabinet/MovingContainer').GetPrim();UsdPhysics.RigidBodyAPI.Apply(container).CreateRigidBodyEnabledAttr(True);UsdGeom.Cube.Define(s,'/World/Cabinet/MovingContainer/LowerVisual')
mutant=copy.deepcopy(b);mutant['source_components']['drawer_cabinet_drawer_02']=['/World/Cabinet/MovingContainer/LowerVisual'];e=E.validate_component_bindings(s,mutant);record('inherited_dynamic_lower_rejected',any('inherits moving' in x for x in e),e)
result={'scope':'Synthetic source-parenting mutation tests, not benchmark author outputs','pass':all(x['pass'] for x in checks),'tests':checks};E.write_json(E.HERE/'static_binding_test_evidence.json',result);print(json.dumps(result,indent=2));assert result['pass']
