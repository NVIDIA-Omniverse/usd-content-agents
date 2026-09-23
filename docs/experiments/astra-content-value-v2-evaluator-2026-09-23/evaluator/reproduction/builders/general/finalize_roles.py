"""Freeze source-identified principal mechanical roles before publishing a case."""
import argparse,json
from pathlib import Path
from common import dump,sha
p=argparse.ArgumentParser();p.add_argument('case',choices=['07_robot_arm','09_printer']);a=p.parse_args();root=Path(__file__).parent
if (root/'frozen'/a.case).exists():raise RuntimeError('Case already frozen')
p=root/'references'/a.case/'source_inventory.json';d=json.loads(p.read_text());roles=dict(d.get('source_body_role_requirements',{}))
if a.case=='09_printer':
 indices={**{i:'base' for i in range(15)},349:'gantry',421:'toolhead',**{i:'bed' for i in [546,547,548,549,550,618,619,622,625,626,627]}}
 for part in d['parts']:
  i=int(part['source_id'].split('#solid:')[1].split(':')[0])
  if i in indices:roles[part['source_id']]=indices[i]
 if len(roles)<len(indices):raise RuntimeError('Trusted original solid indices missing')
 d['role_grounding_note']='Principal source STEP bodies: original frameextrusions0..14, crossbeam349, XCarriageBody421, MIC6/heater/magnet/buildsurface546..550 andZbed support618/619/622/625..627. Remaining source geometry must still be preserved and attached appropriately. Cartesian actuator abstraction; no belt/thermal/extrusion model asserted.'
else:
 tokens={'/AssemblyArt1/Model/Art1Body_Art1Body/Body':'link1','/AssemblyArt2/Model/Art2BodyA_Art2BodyA/Body001':'link2','/AssemblyArt2/Model/Art2BodyB_Art2BodyB/Body':'link2','/AssemblyArt3/Model/Art3Body_Art3Body/Body002':'link3','/AssemblyArt4/Model/Art4Body_Art4Body/Body':'link4','/AssemblyArt56/Model/Art56MotorCoverRing_Art56MotorCoverRing/Body':'link5','/AssemblyArt56/Model/GripperBot_GripperBot/Body':'end_effector','/AssemblyArt56/Model/Unnamed_GripperTop/Body':'end_effector','/AssemblyArt56/Model/Art56GearPlate_Art56GearPlate/Body':'end_effector'}
 for token,role in tokens.items():
  matches=[p['source_id'] for p in d['parts'] if p['source_id'].endswith(token)]
  if len(matches)!=1:raise RuntimeError('Source role identity ambiguous or missing '+token+str(matches))
  roles[matches[0]]=role
 d['role_grounding_note']='Source XML major Thor Art1..4 bodies, Art56MotorCoverRing wrist link, Art56GearPlate+GripperBot+GripperTop terminal body. Original coordinate-system links identify ring→gearplate→gripper housing. Geometry preserved by part-local rigid alignment; exact original mate frames/differential motor transmission are not certified. Non-surface construction curves are retained in original files but excluded from mesh inventory.'
d['source_body_role_requirements']=roles;dump(p,d);print(a.case,sha(p),len(roles))
