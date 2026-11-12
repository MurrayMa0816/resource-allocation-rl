External baseline cores for the current chapter 2/3 tasks
=======================================================

Source identity and scope
-------------------------

migration_sources.json records the original shared implementation, immutable
upstream commits, source hashes and the AST hashes of five algorithm update
methods. Those update methods are unchanged. DIME is extracted from the old
combined module; unused BRO/PASPO environment factories are excluded. TD7 and
DIME MIT notices are retained, RPO and DIME upstream MIT license files are
included, and the SimBaV2 Apache-2.0 license is included. PASPO is the existing
independent paper implementation with task-specific capped-simplex intervals.

The runtime imports no old experiments, methods, physical simulator or
FoundationAdapter. Source paths in the manifest are provenance records only.
Checkpoints deliberately do not load old baseline weights: public features,
network inputs, action dimensions and task identities have changed.

Public neural information
-------------------------

The adapter observation schema contains exactly local, global, caps and budget.
local/global contain the current simplified public input. caps/budget carry
FP64 feasible geometry and are not additional neural state variables.

TD7, SimBaV2 and DIME receive only the reduced local/global vector, including
only its coordinates in their replay. PASPO retains the full vector for its
feasible Beta intervals and masks caps/budget to zero before Actor and Critic
encoding. RPO retains precise geometry in replay for equality completion and
GRG correction, while masking those fields before running normalization and
again at its neural output boundary. Geometric constraints can still affect
actual sampled actions through the original feasible decoder; they cannot
influence the neural state features. neural_input.py implements this boundary.

Chapter 2 uses N allocation coordinates with its fixed executable quota.
Chapter 3 PASPO/RPO use the true physical equality polytope with the UNUSED
coordinate first, followed by N allocations. PASPO's dependent coordinate and
RPO's equality-completed coordinate are therefore physical objects. Generic
TD7/SimBaV2/DIME instead retain one amount control and N spatial controls in
their own replay/Q input. Their fixed decoder uses a public remaining-resource
per-remaining-round scale and a feasible spatial projection. Both families
cover the full physical action set. No new network or A/B learned head is added.

Declared neutral action calibration
----------------------------------

The initial chapter-3 PASPO Beta MLE uses reward-blind amount/spatial samples on
the first public training polytope. Its first unused-resource conditional then
has a fixed state-only log-shape offset: log BetaPrior(r_t)-log initial shapes,
where r_t=min(R/(H-t),U)/U. Sampling and known density use the identical offset.
The spatial prior remains an exchangeable geometry prior, not an exact uniform
volume claim. RPO initial output biases are fitted to neutral feasible shares;
its state-only pre-tanh mean offset is used inside both sampling and log density.
The RPO constraint unit is converted back to the physical unit before computing
r_t. Chapter-3 RPO warmup draws from that calibrated actor; chapter 2 retains its
original uniform warmup. Context and initial anchors are checkpointed.

DIME prior_std is explicitly calibrated from the author's 2.5 to 0.5 for these
bounded-control tasks, identically across chapters, environments and seeds.
This changes the declared initial exploration scale, not its update equations:
prior samples, OU drift and DIS density all use that same parameter. No reward
or test path is used to select the value. The initial temperature remains 0.01.

Execution and short checks
--------------------------

The common factory in ../agent.py accepts paspo, rpo, td7, simba_v2 and dime,
with smoke or formal profiles. Smoke is for engineering checks only. It reduces
network dimensions and uses a two-episode warmup. Formal retains the previously
registered algorithm widths and uses an eight-episode warmup. Training calls
must span complete episodes; PASPO additionally requires a whole PPO rollout
(two smoke or eight formal episodes). All runs count the warmup in the budget.

PASPO retains conditional-Beta-mode deployment; TD7 retains its training-chosen
checkpoint actor; RPO/SimBaV2 retain deterministic actor execution. DIME retains
sampled diffusion deployment. DIME rejects deterministic=True because zero
noise is not its marginal policy mean. These are declared algorithm semantics,
not alternate execution curves for A/B.

The finite terminal objective, no reward shaping, and original native reward
must be supplied by the adapter. Internal SAC entropy, SimBaV2 normalization,
DIME diffusion and TD7 checkpoint schedules remain algorithm components. The
registered formal DIME BRN warmup and TD7 checkpoint phase thresholds can exceed
a short research budget; no reward-based retuning or paper performance claim
is made by this migration. The categorical value supports and temperatures
remain registered task settings requiring separate research-budget evaluation.

Run the focused synthetic check with the current package on PYTHONPATH::

    python -m allocation23.baselines.cores.test_migration <verification.json>

It checks both action dimensions, actual parameter updates, geometry isolation,
exact resumed continuation and unchanged update-method AST hashes. It is not
an experiment on either research environment and produces no research curve.
