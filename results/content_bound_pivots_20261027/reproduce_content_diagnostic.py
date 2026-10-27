from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
from ehl_solver.benchmark_200n import Circular200N
from ehl_solver.property_load_newton import PropertyLoadSystem
from ehl_solver.cavitating_newton import solve_cavitating_correction
from examples.run_scientific_validation import provenance,save_json
c=Circular200N();problem=c.problem(.5e-6,ny=97,nx=145);fluid=c.fluid()
source=Path('results/best_candidate_refinement_factor3');meta=json.load(open(source/'summary.json'))
with np.load(source/'failed_partial.npz') as a:
 seed=SimpleNamespace(pressure_pa=a['pressure_pa'],fractional_film_content=a['theta'])
system=PropertyLoadSystem(problem,fluid,2.,200.,preconditioner='local');z=system.pack(seed,meta['failed_offset_m']);r=system.residual(z);merit=float(r@r/(2*system.size))
full=system.full(z).copy();index=int(np.argmax(z[system.n:2*system.n]));j,i=divmod(index,143)
print('Blocking cell',j+1,i+1,'P,D',z[index],z[system.n+index],'full',bool(full[index]),flush=True)
directory=Path('results/content_blocker_diagnostic');directory.mkdir(exist_ok=False)
rows=[]
for mode in ('original','single-upper-content-pivot'):
 selected=full.copy()
 if mode!='original':selected[index]=True
 view=system if mode=='original' else system.correction_view(z,selected)
 linear=solve_cavitating_correction(view,z,maximum_krylov_cycles=10,gmres_restart=60,schur_tolerance=1e-10)
 d=linear.direction;slope=system.merit_slope(z,r,d)
 margins=np.r_[z[:2*system.n]+1e-10,1+1e-10-z[system.n:2*system.n]]
 velocity=np.r_[-d[:2*system.n],d[system.n:2*system.n]]
 ratios=np.full_like(margins,np.inf);falling=velocity>margins;np.divide(margins,velocity,out=ratios,where=falling)
 bound=min(1.,.99*float(ratios.min()))
 item={'mode':mode,'true_linear_error':linear.true_linear_residual,'slope':slope,'affine_bound':bound,'blocking_component':int(ratios.argmin()),'direction_at_content_blocker':float(d[system.n+index]),'trials':[]}
 for alpha in (1.,bound,bound/2):
  trial=z+alpha*d
  try:
   if alpha<=0 or min(trial[:2*system.n]) < -1e-10 or max(trial[system.n:2*system.n])>1+1e-10:raise ValueError('pressure/content infeasible')
   tr=system.residual(trial);value=float(tr@tr/(2*system.size));item['trials'].append({'alpha':alpha,'merit':value,'strict_armijo':value<merit and slope<0 and value<=merit+1e-4*alpha*slope})
  except ValueError as e:item['trials'].append({'alpha':alpha,'reason':str(e)})
 np.savez_compressed(directory/f'{mode}.npz',state=z,direction=d,selected_full=selected)
 rows.append(item);save_json(directory/'summary.json',{'runtime':provenance(),'scope':'reconstructed rejected partial; diagnostic only, not an accepted warm root','blocking_cell_yx':[j+1,i+1],'rows':rows})
 print(item,flush=True)
