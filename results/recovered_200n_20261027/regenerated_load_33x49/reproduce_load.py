import json
from pathlib import Path
import numpy as np
from ehl_solver.benchmark_200n import Circular200N
from ehl_solver.property_coupled import PropertyCoupledSolution
from ehl_solver.property_load_newton import continue_property_load, PropertyLoadPathError
from examples.run_property_load_verification import save_state
from examples.run_scientific_validation import provenance, save_json
from examples.run_load_directed_200n import finite_json
c=Circular200N(); f=c.fluid(); p=c.problem(.5e-6,ny=33,nx=49)
source=Path('results/regenerated_seed_33x49/solution.npz')
with np.load(source) as a:
 seed=PropertyCoupledSolution(a['pressure_pa'],a['theta'],a['film_m'],a['displacement_m'],p.load_n(a['pressure_pa']),{},(),1.)
offset=.5e-6
d=Path('results/regenerated_load_33x49'); d.mkdir(exist_ok=False)
grid={'shape_ny_nx':[33,49],'attempts':[],'closed_loads':[],'force_200n_closed':False}
report={'status':'incomplete','runtime':provenance(),'case':c.parameters(),'preconditioner':'local','grids':{'refined':grid},'force_target_n':200.,'mesh_verified':False,'physical_domain_verified':False,'matched_loads':[],'source_npz':str(source),'maximum_attempts_per_target':12}
def checkpoint(): save_json(d/'summary.json',finite_json(report))
checkpoint()
def callback(record,result):
 path=d/f'attempt_{len(grid["attempts"]):03d}';path.mkdir()
 entry=dict(record)
 if record['accepted']:
  s=result.solution
  save_state(path/'solution.npz',p,s,f)
  save_json(path/'history.json',finite_json(s.history))
  entry.update(state_file=str((path/'solution.npz').relative_to(d)),minimum_film_m=float(s.film_thickness_m.min()),central_film_m=float(s.film_thickness_m[16,24]),peak_pressure_pa=float(s.pressure_pa.max()))
 else:
  save_json(path/'failure.json',finite_json({'history':result.history,'diagnostics':result.diagnostics}))
  if result.last_root is not None:
   save_state(path/'failed_partial.npz',p,result.last_root.solution,f)
 grid['attempts'].append(entry);checkpoint()
 print(len(grid['attempts']),record,flush=True)
for target in (2.,4.,8.,10.,20.,40.,80.,120.,160.,200.):
 try:
  result=continue_property_load(p,f,2.,target,seed=seed,seed_offset_m=offset,initial_increment_n=min(2.,max(.25,seed.load_n*.2)),maximum_attempts=12,preconditioner='local',callback=callback)
 except PropertyLoadPathError as e:
  grid['stall']={'requested_load_n':target,'reason':str(e),'last_load_n':e.last_root.solution.load_n,'last_offset_m':e.last_root.central_film_offset_m}
  break
 seed=result.root.solution;offset=result.root.central_film_offset_m
 grid['closed_loads'].append({'target_n':target,'calculated_n':seed.load_n,'offset_m':offset})
 grid['force_200n_closed']=target==200.;checkpoint()
report['status']='bounded regenerated load assessment complete; mesh/physical accuracy open';checkpoint()
print('Completed',grid['closed_loads'],grid.get('stall'),flush=True)
