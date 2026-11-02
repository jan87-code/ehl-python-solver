# Mesh Study: 50 N Film Thickness vs Hamrock-Dowson

## Overview

This document summarizes the mesh convergence study for 50 N circular contact cases, comparing simulated minimum and central film thickness values with Hamrock-Dowson (HD) correlation predictions.

**Reference**: Hamrock-Dowson minimum/central film: **284.150 nm / 497.969 nm**

---

## Existing Results from ROADMAP

### Mesh Convergence Data

| Grid | Physical Nodes | Interior Nodes | Minimum Film (nm) | Central Film (nm) | Min vs HD | Central vs HD | Notes |
|------|----------------|----------------|-------------------|------------------|-----------|--------------|-------|
| HD Reference | N/A | N/A | 284.150 | 497.969 | 0.00% | 0.00% | Correlation baseline |
| **33×33** | 1,089 | 841 | 159.402 | 296.593 | **-43.90%** | **-40.44%** | Coarse, large deviation |
| **65×65** | 4,225 | 3,969 | 257.963 | 428.588 | **-9.22%** | **-13.93%** | Significant improvement |
| **129×129** | 16,641 | 15,877 | 281.506 | 462.814 | **-0.93%** | **-7.06%** | Close to HD minimum |
| **141×141** | 19,881 | 19,045 | 283.271 | 464.741 | **-0.309%** | **-6.673%** | Slightly better |
| **173×173** | 29,929 | 29,121 | 286.605 | 468.093 | **+0.864%** | **-6.000%** | Minimum now above HD |
| **199×199** | 39,601 | 38,809 | 286.690 | 469.711 | **+0.894%** | **-5.675%** | Small improvement |
| **223×223** | 49,729 | 48,841 | **TBD** | **TBD** | **TBD** | **TBD** | **Target of this study** |

### Film Convergence Trend

**Minimum Film**:
- 33×33 → 65×65: +61.5% (159.4 → 257.96)
- 65×65 → 129×129: +9.9% (257.96 → 281.51)
- 129×129 → 141×141: +0.627% (281.51 → 283.27)
- 141×141 → 173×173: +1.177% (283.27 → 286.61)
- 173×173 → 199×199: +0.029% (286.61 → 286.69)

**Central Film**:
- 33×33 → 65×65: +44.5% (296.59 → 428.59)
- 65×65 → 129×129: +7.986% (428.59 → 462.81)
- 129×129 → 141×141: +0.416% (462.81 → 464.74)
- 141×141 → 173×173: +0.721% (464.74 → 468.09)
- 173×173 → 199×199: +0.346% (468.09 → 469.71)

### Deviation vs HD Correlation

| Grid | Min Deviation (%) | Central Deviation (%) | Min Error (nm) | Central Error (nm) |
|------|-------------------|---------------------|---------------|------------------|
| 33×33 | -43.90 | -40.44 | -124.748 | -201.376 |
| 65×65 | -9.22 | -13.93 | -26.187 | -69.381 |
| 129×129 | -0.93 | -7.06 | -2.644 | -35.155 |
| 141×141 | -0.309 | -6.673 | -0.879 | -33.228 |
| 173×173 | **+0.864** | -6.000 | +2.455 | -29.876 |
| 199×199 | **+0.894** | -5.675 | +2.540 | -28.258 |

**Observation**: Minimum film converges to HD within ~1% starting from 129×129, while central film remains ~6% below HD even at 199×199.

---

## Pressure Differences

From ROADMAP (non-nested comparison, bilinear sampling):

| Comparison | Pressure L2 | Pressure Max | Minimum Film Change |
|------------|-------------|--------------|---------------------|
| 129 → 173 | 2.173% | 13.395% | +1.811% |
| 141 → 173 | 2.307% | 11.619% | +1.177% |

**Observation**: Pressure convergence is slower than film convergence, with L2 differences ~2-2.3% and max differences ~11-13% between grids.

---

## Current Status

| Grid | Status | Film Data | Preflight | Output Directory |
|------|--------|-----------|-----------|------------------|
| 33×33 | ✅ Complete | Yes | N/A | `results/limited_secant_50n_129_20261031/` |
| 65×65 | ✅ Complete | Yes | N/A | (referenced in 129 comparison) |
| 129×129 | ✅ Complete | Yes | N/A | `results/limited_secant_50n_129_20261031/` |
| 141×141 | ✅ Complete | Yes | N/A | `results/limited_mesh_50n_141_20261031/` |
| 173×173 | ✅ Complete | Yes | N/A | `results/limited_mesh_50n_173_retry_20261031/` |
| 199×199 | ✅ Complete | Yes | ✅ Passed | `results/limited_mesh_50n_199_20261031/` |
| **223×223** | ⏳ **Pending** | **TBD** | ✅ Passed (2 GB cap) | `results/limited_mesh_50n_223_20261102/` |

---

## Setup for 223×223 Run

### Preflight Configuration
- **Preflight cap**: 2 GiB (2,147,483,648 bytes)
- **Commit**: `1f3f1a1` - Raise preflight cap to 2 GiB for 50k mesh study
- **Case**: `cases/circular_50n_223.json`
- **Parameters**:
  - Load: 50 N
  - Radius: 0.015 m
  - Contact modulus: 113 GPa
  - Viscosity: 0.04 Pa s
  - Density: 850 kg/m³
  - Entrainment: 2.0 m/s
  - α: 2.21e-8 Pa⁻¹
  - Roelands pressure: 198 MPa

### Run Command
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src .venv/Scripts/python \
  examples/run_limited_transport_path.py \
  --case cases/circular_50n_223.json \
  --output results/limited_mesh_50n_223_20261102 \
  --max-attempts 24
```

### Expected Results
- **Minimum film**: ~286.7-287.0 nm (extrapolated from trend)
- **Central film**: ~470.0-471.0 nm (extrapolated)
- **Min vs HD**: ~+0.9-1.0%
- **Central vs HD**: ~-5.4-5.5%
- **Runtime**: Several hours (49,729 nodes)
- **Memory**: ~1.2-1.8 GB (from historical scaling)

---

## Mesh Convergence Analysis

### Film Thickness Convergence

**Minimum Film**:
- Converging to HD from below
- Crossed HD at 173×173 (+0.864%)
- Trend: Slight increase with refinement
- **Convergence rate**: ~1% change from 141→173→199

**Central Film**:
- Consistently below HD
- Slow convergence: -7.06% → -6.673% → -6.000% → -5.675%
- **Convergence rate**: ~0.3-0.4% improvement per refinement
- **Remaining error**: ~5.7% at 199×199

### Pressure Convergence
- L2 norm: ~2.2% between grids
- Max norm: ~12% between grids
- **Not yet converged** - needs finer grids or domain adjustment

---

## Interpretation

### Minimum Film
- **Good agreement**: Within ~1% of HD starting from 129×129
- **Convergence**: Appears to be converging, possibly to value slightly above HD
- **Physical meaning**: HD is a correlation, not exact physics; small deviations expected

### Central Film
- **Persistent offset**: ~6% below HD across all refined meshes
- **Not converging to HD**: May indicate:
  - Domain size effects (HD assumes infinite domain)
  - Boundary condition effects
  - Model limitations (e.g., cavitation treatment, rheology)
  - HD correlation limitations for this specific case

### Pressure
- **Slow convergence**: L2 ~2%, Max ~12% between grids
- **Minimum location shifts**: Position moves with refinement
- **Conclusion**: Pressure field not yet mesh-converged

---

## Recommendations

### For 223×223 Run
1. **Run with 2 GiB cap** - Should provide adequate headroom
2. **Monitor memory** - Track actual RSS vs preflight estimate
3. **Document film values** - Compare with HD and trend from smaller grids
4. **Check convergence** - Compare with 199×199 results

### For Mesh Convergence
1. **Continue refinement** - Add 257×257, 289×289 if resources allow
2. **Check domain independence** - Verify results with larger domain
3. **Compare with HD parameters** - Ensure same E', material properties
4. **Investigate central film offset** - Understand why central film remains ~6% below HD

### For Physical Validation
1. **Don't claim convergence yet** - Pressure and central film not converged
2. **HD correlation is approximate** - 1-2% film agreement is reasonable
3. **Focus on trends** - Film values approaching HD with refinement
4. **Document limitations** - Domain effects, boundary conditions, model assumptions

---

## Data Sources

All data extracted from `docs/LIMITED_ENDPOINT.md`:
- 33×33, 65×65, 129×129: Section "Third mesh: 50 N 129x129"
- 141×141: Section "Fourth mesh: 50 N 141x141"
- 173×173: Section "Reproduce the refinement" and "The retry passed preflight"
- 199×199: Section "40,000-node 50 N mesh: 199x199"

---

## Next Steps

1. ✅ **Raise preflight cap to 2 GiB** - Done (commit `1f3f1a1`)
2. ⏳ **Run 223×223 case** - Pending
3. ⏳ **Extract film values** - After run completes
4. ⏳ **Update mesh study table** - After run completes
5. ⏳ **Analyze convergence** - After run completes

**Estimated time for step 2**: 2-6 hours depending on system performance
