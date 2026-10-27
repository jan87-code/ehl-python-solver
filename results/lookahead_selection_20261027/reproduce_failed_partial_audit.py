import json
from pathlib import Path
import numpy as np
from scipy.integrate import trapezoid
from ehl_solver.benchmark_200n import Circular200N
from ehl_solver.property_reynolds import mass_audit, admissible
from ehl_solver.elasticity_audit import direct_midpoint_displacement
from examples.run_scientific_validation import provenance, save_json
root=Path('results/best_candidate_refinement_factor3')
report=json.load(open(root/'summary.json'));history=json.load(open(root/'failure.json'))['history']
c=Circular200N();f=c.fluid();problem=c.problem(.5e-6,ny=97,nx=145)
with np.load(root/'failed_partial.npz') as a:
 p,t,h,w=[a[k] for k in ('pressure_pa','theta','film_m','displacement_m')]
 np.testing.assert_array_equal(a['x_m'],problem.x_m);np.testing.assert_array_equal(a['y_m'],problem.y_m)
with np.load(root/'unaccepted_guess.npz') as a,np.load('results/regenerated_refinement_factor3/unaccepted_guess.npz') as b:
 for key in a.files:np.testing.assert_array_equal(a[key],b[key])
audit=mass_audit(problem.x_m,problem.y_m,h,p,t,f,2.,length_scale_m=problem.length_scale_m,film_scale_m=problem.film_scale_m,pressure_scale_pa=problem.pressure_scale_pa)
exact,ew=problem.fields(p,central_film_offset_m=report['failed_offset_m']);force=float(trapezoid(trapezoid(p,problem.x_m,axis=1),problem.y_m))
models=[m for r in history for m in r['models']];accepted=[]
for i,row in enumerate(history):
 chosen=[(m,trial) for m in row['models'] for trial in m['trials'] if trial['accepted']]
 if not chosen:
  assert i==len(history)-1
  continue
 assert len(chosen)==1
 m,trial=chosen[0]
 assert m['slope']<0 and trial['merit']<row['merit']
 assert trial['merit']<=row['merit']+1e-4*trial['alpha']*m['slope']
 eligible=[trial2['merit'] for m2 in row['models'] for trial2 in m2['trials']
           if trial2['merit'] is not None and m2['slope']<0 and trial2['merit']<row['merit']
           and trial2['merit']<=row['merit']+1e-4*trial2['alpha']*m2['slope']]
 assert trial['merit']==min(eligible)
 accepted.append(trial['alpha'])
peak=np.unravel_index(np.argmax(p),p.shape)
direct=direct_midpoint_displacement(problem.x_m,problem.y_m,p,c.contact_modulus_pa,[[0.,0.],[problem.x_m[peak[1]],problem.y_m[peak[0]]]])
result={'status':'failed partial independently assessed; no target root accepted','runtime':provenance(),'input_arrays_identical_to_legacy':True,'mass_audit':audit,'mass_and_complementarity_admissible':admissible(audit,p,t,problem.pressure_scale_pa),'force_n':force,'relative_force_error':abs(force/200-1),'maximum_film_difference_m':float(abs(h-exact).max()),'maximum_displacement_difference_m':float(abs(w-ew).max()),'direct_source_error_m':float(abs(direct-[w[48,72],w[peak]]).max()),'minimum_film_m':float(h.min()),'maximum_pressure_pa':float(p.max()),'recorded_correction_attempts':len(history),'accepted_corrections':len(accepted),'linear_models':len(models),'maximum_true_linear_error':max(m['linear_true_residual'] for m in models),'accepted_step_lengths':accepted,'best_candidate_history_verified':True,'superseded_candidates':sum('superseded' in trial['reason'] for m in models for trial in m['trials']),'final_merit':history[-1]['merit'],'final_rejection_reasons':sorted({trial['reason'] for m in history[-1]['models'] for trial in m['trials']})}
assert not result['mass_and_complementarity_admissible']
assert result['maximum_film_difference_m']==0
assert result['maximum_true_linear_error']<=5e-9
save_json(root/'failed_partial_audit.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='runtime'},indent=2))
