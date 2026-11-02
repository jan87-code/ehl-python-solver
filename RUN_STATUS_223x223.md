# 223×223 (50k) Run Status

## Current Status: ⏳ **IN PROGRESS**

### Last Update: 2026-11-02

---

## ✅ **Preflight: PASSED**

**Output Directory**: `results/limited_mesh_50n_223_20261102/`

### Preflight Results
- **Physical nodes**: 49,729 (223×223)
- **Interior nodes**: 48,841
- **Matrix nonzeros**: 439,344
- **Memory ceiling**: 2,147,483,648 bytes (**2 GiB**)
- **Estimated peak**: 522,986,376 bytes (498.8 MB)
- **Adjusted peak**: 601,434,332 bytes (573.6 MB)
- **Preflight passed**: ✅ **TRUE**
- **Factor preflight passed**: ✅ **TRUE**

### Configuration
- **Case**: `cases/circular_50n_223.json`
- **Load**: 50 N
- **Grid**: 223×223 = 49,729 nodes
- **Preflight cap**: 2 GiB (commit `1f3f1a1`)
- **Max attempts**: 24
- **Endpoint prediction**: Enabled

---

## 📊 **Current State**

### Status
- **Status**: `running`
- **Stages completed**: 0
- **Attempts**: N/A

### Issue
The preflight passed successfully, but the actual solve has not yet started any stages. This is expected behavior when:
1. No seed is provided (`initial_state=None`)
2. The solver needs to create its own initial guess
3. Initialization for 49,729 nodes takes time

### Expected Behavior
From the ROADMAP:
- **173×173** (29,929 nodes): 14 attempts, 11 accepted stages, 3 rejections
- **199×199** (39,601 nodes): 12 attempts, 2 rejected near-endpoint
- **223×223** (49,729 nodes): Expected similar (~12-14 attempts)

### Estimated Runtime
- **173×173**: Completed (reference)
- **199×199**: Completed (reference)
- **223×223**: **2-6 hours** (extrapolated)

---

## 🎯 **What Happens Next**

The `continue_limited_transport` function will:

1. **Create pending stages** at transport fractions: 0.0, 0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999, 0.99999999, 0.999999999
2. **Start from 0.0** with a default initial guess (since no seed provided)
3. **Progress through stages** using guarded secant initialization
4. **Attempt final endpoint** (transport_fraction=1.0) with zero Newton corrections
5. **Save results** when complete

### Monitoring
Check progress with:
```bash
# Check summary
cat results/limited_mesh_50n_223_20261102/summary.json | python -c "import sys,json; d=json.load(sys.stdin); print('Status:', d['status'], '| Stages:', len(d.get('stages',[])))"

# Check for stage directories
ls results/limited_mesh_50n_223_20261102/attempt_*

# Tail the log
# tail -f results/limited_mesh_50n_223_20261102/run.log
```

---

## 📈 **Expected Results**

Based on mesh study data from `MESH_STUDY_50N_FILM.md`:

### Film Thickness (HD Reference: 284.150 / 497.969 nm)

| Grid | Minimum Film (nm) | Central Film (nm) | Min vs HD | Central vs HD |
|------|-------------------|------------------|-----------|--------------|
| HD | 284.150 | 497.969 | 0.00% | 0.00% |
| 173×173 | 286.605 | 468.093 | +0.864% | -6.000% |
| 199×199 | 286.690 | 469.711 | +0.894% | -5.675% |
| **223×223** | **~286.7-287.0** | **~470.0-471.0** | **~+0.9-1.0%** | **~ -5.4-5.5%** |

### Trends
- **Minimum film**: Converging to HD from above, changes <0.03% between 173→199
- **Central film**: Slow convergence, ~0.3% improvement per refinement
- **Pressure**: L2 ~2.2%, Max ~12% between grids (not yet converged)

---

## ⚠️ **Known Issues & Limitations**

### 1. **Memory Estimate Accuracy**
- Preflight uses dummy state (P=0.5, D=0.5, h0=0.1)
- **Underestimates real memory by ~2x** (498.8 MB vs historical 1,009.6 MB)
- **But 2 GiB cap provides ~800 MB margin** over historical estimate

### 2. **No Seed Provided**
- Starting from cold guess without warm start
- May take longer to converge
- First few stages may be slow

### 3. **Single-Threaded**
- Running with `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`
- Prevents BLAS contention but may be slower

---

## 🎯 **Next Actions**

### If the run completes successfully:
1. Extract film thickness from `summary.json`
2. Update `MESH_STUDY_50N_FILM.md` with new data
3. Analyze convergence trend
4. Document comparison with HD correlations

### If the run fails:
1. Check `summary.json` for error details
2. Check `run.log` for traceback
3. Try with a seed from 199×199 (if available)
4. Increase timeout or check memory usage

---

## 📁 **Files**

- **Output**: `results/limited_mesh_50n_223_20261102/`
- **Summary**: `results/limited_mesh_50n_223_20261102/summary.json`
- **Log**: `results/limited_mesh_50n_223_20261102/run.log`
- **Source snapshot**: `results/limited_mesh_50n_223_20261102/source/`
- **Driver**: `results/limited_mesh_50n_223_20261102/run_limited_transport_path.py`

---

## 🔄 **Current Git State**

```
1f3f1a1 Raise preflight cap to 2 GiB for 50k mesh study
77ab555 Add default initial state for preflight on large grids without seed
23f4d65 Raise preflight cap to 1.25 GiB and allow 50k test without seed
f0b2091 Add measured peak memory tracking during preflight
3a18e8e Add cross-platform memory detection for preflight checks
71870d0 Add bounded 40k and 50k MC mesh studies
```

**All commits pushed to `origin/develop`**

---

## ✅ **Summary**

| Task | Status | Details |
|------|--------|---------|
| Raise cap to 2 GiB | ✅ Done | Commit `1f3f1a1` |
| Preflight for 223×223 | ✅ Passed | 573.6 MB < 2,048 MB |
| Start 223×223 solve | ✅ Running | Status: running, 0 stages |
| **Complete solve** | ⏳ Pending | Expected: 2-6 hours |
| Extract film results | ⏳ Pending | After completion |
| Update mesh study | ⏳ Pending | After completion |

**The run has started successfully and preflight passed. Await completion.**
