# EHL solver development

Read and follow the repository-root `AGENTS.md` as the authoritative persistent
project instructions. Before numerical changes, read `docs/PHYSICS.md`,
`docs/ROADMAP.md` and `docs/MATLAB_LESSONS.md`; determine implemented capabilities
from source and independently run tests.

Develop successive small, reproducible CPU milestones on `develop`. Preserve
unrelated work and published history. Keep resource use bounded, document
acceptance evidence and unresolved limitations in the roadmap, and follow the
verification and commit/push requirements in `AGENTS.md`.

The signed full-film nonlinear solver and dense pseudo-arclength continuation
are mathematical verification tools. They do not establish physical EHL load
closure, coupled cavitation, or strong-compliance robustness. Continue from the
roadmap and measured diagnostics rather than assuming those goals are complete.
