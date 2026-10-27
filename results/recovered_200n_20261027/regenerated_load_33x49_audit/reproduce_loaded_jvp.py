from pathlib import Path
import json
import numpy as np
from ehl_solver.benchmark_200n import Circular200N
from ehl_solver.property_coupled import PropertyCoupledSolution
from ehl_solver.property_load_newton import PropertyLoadSystem,verify_seed
from examples.run_scientific_validation import provenance,save_json
c=Circular200N();fluid=c.fluid()
s=json.load(open('results/regenerated_load_33x49/summary.json'))
e=[e for e in s['grids']['refined']['attempts'] if e['accepted']][-1]
rows=[]
for path,offset in [(Path('results/regenerated_load_33x49')/e['state_file'],e['offset_m']),(Path('results/regenerated_refinement_factor2/solution.npz'),json.load(open('results/regenerated_refinement_factor2/summary.json'))['grids']['refined']['attempts'][0]['offset_m'])]:
 with np.load(path) as a:
  p,t,h,w=[a[k] for k in ('pressure_pa','theta','film_m','displacement_m')]
 problem=c.problem(.5e-6,ny=p.shape[0],nx=p.shape[1]);system=PropertyLoadSystem(problem,fluid,2.,200.)
 seed=PropertyCoupledSolution(p,t,h,w,problem.load_n(p),{},(),1.)
 z=verify_seed(system,seed,offset)
 mask=p[1:-1,1:-1]>.1*p.max()
 field=mask*1e6/problem.pressure_scale_pa*(-1.)**np.arange(p.shape[1]-2)[None,:]
 v=np.r_[field.ravel(),np.zeros(system.n+1)]
 analytic=system.jacobian_vector(z,v)
 errors=[]
 for eps in (.01,.001,.0001):
  # Unpack verifies exact nonlocal film feasibility at both perturbations.
  for sign in (-1,1):
   system.sync(z+sign*eps*v);system.base.unpack((z+sign*eps*v)[:-1])
  fd=(system.residual(z+eps*v)-system.residual(z-eps*v))/(2*eps)
  errors.append({'amplitude':eps,'relative_complete_jvp_error':float(np.linalg.norm(fd-analytic)/np.linalg.norm(analytic))})
 if min(e['relative_complete_jvp_error'] for e in errors)>1e-6: raise RuntimeError('loaded complete JVP audit failed')
 rows.append({'shape_ny_nx':list(p.shape),'source':str(path),'perturbation':'alternating x, 1 MPa amplitude, supported where p>.1*pmax; no offset/content perturbation','errors':errors})
report={'runtime':provenance(),'scope':'complete discrete Jacobian consistency at saved roots; no conditioning/stability conclusion','states':rows}
save_json(Path('results/regenerated_load_33x49_audit/loaded_jvp_audit.json'),report)
print(json.dumps(rows,indent=2))
