"""Case-specific graph and digit observers, independent of submitted scripts."""
import numpy as np
from common import check

def inspect_extra(config,bindings):
    checks=[];c=config['contract'];br=config['body_roles'];jr=config['joint_roles']
    for key in ('bodies','joints'):
        roles=[x['role'] for x in bindings[key]]
        check(checks,'unique_'+key+'_roles',len(roles)==len(set(roles)),roles)
    for role,pair in c['required_joint_body_pairs'].items():
        joint=config['joints'].get(jr.get(role),{})
        expected={br.get(r) for r in pair};actual={joint.get('body0'),joint.get('body1')}
        check(checks,'source_grounded_joint_endpoints:'+role,len(expected)==2 and None not in expected and actual==expected and [joint.get('body0'),joint.get('body1')]==[br.get(r) for r in pair],{'expected_roles':pair,'expected_paths':sorted(str(x) for x in expected),'actual_paths':sorted(str(x) for x in actual)})
        lo,hi=joint.get('lower'),joint.get('upper')
        check(checks,'finite_bounded_joint_range:'+role,lo is not None and hi is not None and 0<hi-lo<=2*np.pi,{'lower':lo,'upper':hi})
    for role in c['required_body_roles']:
        if role in br:
            moving=config['bodies'][br[role]]['moving']
            check(checks,'required_body_motion_class:'+role,moving==(role!='base'),moving)
    return checks

def observe_extra(result,config):
    if 'trace' not in result:return []
    checks=[];c=config['contract'];prefix='seed'+str(result['seed'])+':'
    for role in c.get('terminal_body_roles',[]):
        path=config['body_roles'][role];points=np.asarray([r['poses'][path][:3] for r in result['trace'] if r['t']<7])
        travel=float(np.max(np.linalg.norm(points-points[0],axis=1)))
        check(checks,prefix+'terminal_body_motion:'+role,travel>=c['minimum_terminal_motion_m'],{'travel_m':travel,'minimum_m':c['minimum_terminal_motion_m']})
    for role in c['required_joint_roles']:
        p=config['joint_roles'][role];rows=[r['joints'][p] for r in result['trace'] if 5.7<=r['t']<6.5]
        efforts=[abs(r['external_effort']) for r in rows]
        check(checks,prefix+'nonzero_external_hold_load:'+role,bool(efforts) and min(efforts)>0,{'minimum_abs_torque_Nm':min(efforts) if efforts else None})
    return checks
