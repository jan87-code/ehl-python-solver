from pathlib import Path
import json,hashlib
import numpy as np
from scipy.integrate import trapezoid
from examples.run_scientific_validation import provenance,save_json
from examples.scientific_figures import figure,save
source=Path('results/regenerated_load_33x49');old=json.load(open(source/'summary.json'));entry=[r for r in old['grids']['refined']['attempts'] if r['accepted']][-1]
paths=[source/entry['state_file'],Path('results/content_pivot_refinement_factor2/solution.npz'),Path('results/content_pivot_refinement_factor3/solution.npz')]
with np.load('results/content_pivot_refinement_factor3/unaccepted_guess.npz') as a,np.load('results/best_candidate_refinement_factor3/unaccepted_guess.npz') as b:
 for k in a.files:np.testing.assert_array_equal(a[k],b[k])
rows=[];directory=Path('results/content_pivot_three_grid_comparison');directory.mkdir(exist_ok=False)
fig,axes=figure('Three accepted 200 N roots; raw centerlines, unresolved mesh accuracy','Identical finite +/-3a reservoir and fluid laws; +x entrainment, arrays (ny,nx)\nNo smoothing; minimum/central film and pressure stability remain separate',rows=2)
for path in paths:
 with np.load(path) as a:
  p,h,x,y=[a[k] for k in ('pressure_pa','film_m','x_m','y_m')]
 force=float(trapezoid(trapezoid(p,x,axis=1),y));assert abs(force/200-1)<=1e-8
 idx=np.unravel_index(np.argmin(h),h.shape)
 rows.append({'shape_ny_nx':list(p.shape),'force_n':force,'minimum_film_m':float(h.min()),'central_film_m':float(h[len(y)//2,len(x)//2]),'peak_pressure_pa':float(p.max()),'minimum_film_location_xy_m':[float(x[idx[1]]),float(y[idx[0]])],'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 axes[0].plot(x*1e3,p[len(y)//2]/1e6,'o-',markersize=3,label=str(p.shape))
 axes[1].plot(x*1e3,h[len(y)//2]*1e6,'o-',markersize=3,label=str(p.shape))
for ax in axes:
 ax.set_xlim(-.35,.35);ax.set_xlabel('x along entrainment [mm]');ax.legend()
axes[0].set_ylabel('Pressure [MPa]');axes[1].set_ylabel('Total film [um]');axes[1].set_ylim(0,.8)
save(fig,directory,'three_grid_centerlines')
changes={k:rows[2][k]/rows[1][k]-1 for k in ('minimum_film_m','central_film_m','peak_pressure_pa')}
report={'status':'accepted matched-force scalar comparison; mesh/physical accuracy not established','runtime':provenance(),'rows':rows,'relative_change_65x97_to_97x145':changes,'one_percent_scalar_mesh_gate':all(abs(v)<=.01 for v in changes.values()),'scope':'Sampled extrema and center values; no asymptotic order or full-field convergence inferred. The accepted grids have refinement ratio 1.5; extrema may move.','factor3_initial_arrays_identical_to_failed_baseline':True,'figure_manifest':['three_grid_centerlines.png']}
save_json(directory/'summary.json',report);print(json.dumps({'rows':rows,'changes':changes},indent=2))
