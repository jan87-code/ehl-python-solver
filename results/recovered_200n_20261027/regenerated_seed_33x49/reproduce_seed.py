from pathlib import Path
from ehl_solver.benchmark_200n import Circular200N
from ehl_solver.property_coupled import solve_property_coupling
from examples.run_200n_readiness import retain_failure
from examples.run_property_load_verification import save_state
from examples.run_scientific_validation import provenance, save_json
c=Circular200N(); p=c.problem(.5e-6,ny=33,nx=49); f=c.fluid()
d=Path('results/regenerated_seed_33x49');d.mkdir(exist_ok=False)
save_json(d/'runtime.json',provenance())
print('Starting cold 33x49 seed at h0=0.5 um',flush=True)
try:
 s=solve_property_coupling(p,f,c.entrainment_m_s)
except Exception as e:
 retain_failure(d,e,p)
 raise
save_state(d/'solution.npz',p,s,f)
save_json(d/'history.json',s.history)
save_json(d/'summary.json',{'load_n':s.load_n,'audit':s.audit,'offset_m':.5e-6})
print('Accepted cold seed',s.load_n,s.audit,flush=True)
