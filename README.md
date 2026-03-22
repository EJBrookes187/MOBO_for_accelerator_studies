MOBO Optimiser README
=====================

This repository provides a multi-objective Bayesian optimisation (MOBO) workflow for accelerator tuning, beamline optimisation, and other expensive optimisation problems where each evaluation is costly and several competing objectives must be balanced.

What the current code does
--------------------------
The current implementation supports:
- Multi-objective Bayesian optimisation using one independent Gaussian Process (GP) per objective
- Feasibility-first Pareto-front handling for constrained problems
- Batch Upper Confidence Bound (BATCH_UCB) candidate selection with a greedy diversity penalty
- Manual operator-driven evaluation or automated Python goal-function evaluation
- Optional per-objective measurement-error handling
- Optional soft penalisation of predicted constraint violation during acquisition
- Optional objective-region penalties that add a smooth penalty to all objectives before acquisition and reporting
- Saving repositories, metrics, metadata, GP models, and copied goal-function source files under the working directory
- Checkpointing for run resumption

Main entry points
-----------------
- run_mobo(...)
- run_mobo_from_config(config_path, n_runs=1)

Current optimisation behaviour
-----------------------------
- All objectives are handled internally as minimisation targets. Some objectives are best constructed as a minimisation problem from their value to a specific defined objective value (eg. if an objective should be minimised but could be positive or negative, it should be defined as a minimisation of its distance to 0).
- Objectives declared with `direction="max"` are negated internally.
- The final Pareto front is always built with a feasibility-first rule:
  - if feasible points exist, only feasible points are considered;
  - otherwise the least-infeasible point or points are returned.
- Multiple constraints are combined into a scalar violation value using the maximum per-constraint violation.
- The active acquisition path is `BATCH_UCB`.
- Candidate points are generated continuously within the input bounds; the `step` field is stored and documented, but it is not enforced by the active acquisition path.

Constraint handling
-------------------
Each constraint is defined by:
- `name`
- `sense` (`<=` or `>=`)
- `threshold`
- optional `scale`

For each constraint, the code computes a non-negative violation:
- `<=` constraint: `max(0, value - threshold)`
- `>=` constraint: `max(0, threshold - value)`

The combined scalar constraint violation is:

    CV = max(per_constraint_violations)

Important implementation note:
The code only normalises a constraint by `scale` if the constraint dictionary contains `normalise=True`. The current public config-file parser and normal examples do not add `normalise=True`, so in normal usage the `scale` field is stored but not currently applied during CV calculation.

Acquisition and penalties
--------------------------
For each candidate point, each objective GP predicts a mean and standard deviation. The code then forms the raw minimisation-style UCB score:

    acq_raw(x) = sum_j [ mu_j(x) - beta * sigma_j(x) ]

If `weighting=True`, the per-objective terms are multiplied by the user-supplied `weight` vector before summation.

If constraints exist and `constraint_penalty_alpha > 0`, the code fits an additional GP to the scalar constraint-violation repository and applies:

    penalty_factor(x) = exp(alpha * max(0, cv_mu(x) - feasible_tol))
    acq_penalised(x) = acq_raw(x) * penalty_factor(x)

This discourages candidates predicted to be infeasible while leaving clearly feasible candidates unchanged.

A separate penalty mechanism can also be applied through the `penalties` specification. Each penalty defines an allowed interval `[lower, upper]` and uses an exponential region penalty outside that interval. The total penalty is added to every objective value for acquisition and stored diagnostics.

Initial data sources
------------------
The optimiser needs initial data before the Bayesian loop starts. The code uses these sources in this order:

1. `training_csv_path`, if the file exists
2. `initial_samples`, if provided
3. online random initial sampling within the input bounds

CSV initialisation
------------------
When `training_csv_path` points to an existing CSV, the code loads the initial dataset directly from it. The expected column order is:

    [inputs..., objectives..., errors..., constraints..., penalties...]

Details:
- input columns must appear first, in the same order as `inputs`
- objective columns must appear next, in the same order as `objectives`
- error columns come next, one per objective
- constraint columns come next, one per configured constraint
- penalty raw-value columns come last, one per configured penalty

If the CSV has no usable error columns, the code fills errors with `default_objective_error`.

Direct initial samples
-----------------------
initial_samples is a list of dictionaries. Each entry may contain:
- `X`: input vector
- `Y`: objective vector
- `error`: optional error vector, length = number of objectives
- `C`: optional raw constraint values, length = number of constraints
- P: optional raw penalty values, length = number of penalties

If an optional field is omitted, the code fills it with defaults.

Online initial sampling
-----------------------
If no CSV and no `initial_samples` are supplied, the code generates `no_of_meas` random points uniformly inside the input bounds, evaluates them, and uses them as the initial repository.

Evaluation modes
----------------
Supported evaluation methods in the current code are:
- `MANUAL`
- `GOAL_FUNCTION` (also accepted: `GOAL`, `FUNC`, `FUNCTION`)

`MANUAL`
- prints the proposed settings
- waits for operator confirmation
- asks for each objective value
- asks for each objective error
- asks for each constraint value if constraints are defined
- returns no penalty values

`GOAL_FUNCTION`
- imports a Python function from `goal_function_path`
- calls `goal_function_name(x.reshape(1, -1), **goal_function_kwargs)`
- supports serial or multiprocessing batch evaluation

Accepted goal-function return formats
-------------------------------------
The goal function may return a dictionary containing:
- `objectives` or `objs`: required unless `raw` is provided
- `constraints` or `cons`: optional; defaults to zeros if constraints are configured
- `errors`: optional; defaults to `default_objective_error`
- `penalties`: optional; defaults to zeros if penalties are configured
- `raw`: optional objective vector; if present, it is used instead of `objectives`

`raw` is expected to have length equal to the number of objectives.

The accepted dictionary shape is therefore typically:

    {
        "objectives": [...],
        "errors": [...],
        "constraints": [...],
        "penalties": [...]
    }

Public Python entry point
-----------------
run_mobo(...) accepts the arguments:
- `inputs`
- `objectives`
- `constraints=None`
- `penalties=None`
- `initial_samples=None`
- `evaluation_method="Manual"`
- `weighting=False`
- `weight=None`
- `iterations=100`
- `no_of_meas=20`
- `batch_size=1`
- `save_name="mobo_run"`
- `resume=False`
- `restart_from_iteration=None`
- `multiprocess_bool=False`
- `use_real_error=False`
- `default_objective_error=1e-10`
- `goal_function_path=""`
- `goal_function_name="goal_function"`
- `goal_function_kwargs=None`
- `model_path=""`
- `training_csv_path=""`
- `feasible_tol=0.0`
- `constraint_penalty_alpha=0.0`
- `working_dir="."`
- `outputs_dirname="Outputs"`
- `benchmarks_dirname="Benchmarks"`
- `n_runs=1`

Config-file entry point
-----------------------
`run_mobo_from_config(config_path, n_runs=1)` reads a text config file with these sections:
- `[INPUTS]` required
- `[OBJECTIVES]` required
- `[CONSTRAINTS]` optional
- `[PENALTIES]` optional
- `[INITIAL_SAMPLES]` optional
- `[GENERAL]` optional but usually needed

Important parser notes:
- The current parser supports `MANUAL` and `GOAL_FUNCTION`
- `goal_function_kwargs` in a config file must be valid JSON, for example:

      goal_function_kwargs = {"alpha": 0.5, "beta": 2}

- Penalty bounds are later converted using Python float(...) In config files, use numeric values that `float` can parse directly. For infinity, use `inf`, not np.inf.

Key user-facing config options
------------------------------
`inputs`
- List of dictionaries with `name`, `min`, `max`, `step`
- `step` is not normally enforced by the active acquisition code

`objectives`
- List of dictionaries with `name`, `direction`, `ref_value`
- `direction` must be `min` or `max`
- `ref_value` should be a conservative worst-case value, larger than expected penalised objective values after internal minimisation conversion

`constraints`
- List of dictionaries with `name`, `sense`, `threshold`, optional `scale`
- `sense` must be `<=` or `>=`

`penalties`
- List of dictionaries with `name`, `lower`, `upper`, `scale`, `rate`
- Used for soft objective penalisation outside an allowed interval

`weighting`
- Enables weighted acquisition aggregation
- `False` for balanced search, `True` if some objectives should dominate acquisition decisions

`weight`
- Relative objective weights used only when `weighting=True`
- Length should equal the number of objectives
- Reasonable values: all ones for balanced search; larger values for objectives you want to prioritise more strongly

`iterations`
- Number of Bayesian optimisation iterations after initialisation
- Reasonable starting point: at least several times the number of inputs; many practical runs use 50 to 100+

`no_of_meas`
- Number of initial evaluations when initial data is generated online
- Reasonable starting point: 5 * n_inputs when budget allows

`batch_size`
- Number of candidates proposed per iteration
- Use `1` for simplest and safest machine operation; use larger values only when parallel evaluation is genuinely available

`save_name`
- Prefix used for saved output folders and checkpoint files
- Use a short descriptive run name

`working_dir`
- Root directory where `Outputs/` and `Benchmarks/` are created
- Should usually be a dedicated run directory

`resume`
- If `True`, the optimiser attempts to load an existing checkpoint and continue
- `False` for fresh runs, `True` only when you expect matching saved state to exist

`restart_from_iteration`
- Optional truncation/restart index used with resumed state
- Intended as a 0-based iteration index

`multiprocess_bool`
- Only affects `GOAL_FUNCTION` evaluation
- If `True`, batch points are evaluated with a multiprocessing pool using all CPU cores
- Use `True` only if the goal function is multiprocessing-safe and importable in worker processes

`use_real_error`
- If `True`, GP observation noise uses the supplied per-objective errors
- If `False`, all objective errors are replaced by `default_objective_error`

`default_objective_error`
- Fallback per-objective standard deviation used when real errors are absent or disabled
- Reasonable values depend on your measurement noise floor; use a small positive number, not zero

`goal_function_path`
- Path to a `.py` file or importable module containing the goal function

`goal_function_name`
- Name of the callable inside the module, usually `goal_function`

`goal_function_kwargs`
- Optional dictionary passed through to the goal function on every call
- Use for fixed evaluation settings that should also be recorded in metrics and metadata

`training_csv_path`
- Path to the initial-data CSV
- If this file exists, it takes precedence over generated initial samples

`feasible_tol`
- Soft feasibility margin used only in the acquisition penalty and some plotting/feasibility summaries
- Reasonable values: `0.0` for strict feasibility, small positive values when constraint signals are noisy

`constraint_penalty_alpha`
- Strength of the exponential acquisition penalty for predicted infeasibility
- `0.0` disables the penalty GP path
- Reasonable values: start around `0.1` to `2.0`, then increase if the optimiser keeps proposing infeasible points

`outputs_dirname`, `benchmarks_dirname`
- Folder names under `working_dir`
- Defaults are usually sensible

Internal config fields present but not user-exposed
---------------------------------------------------
The `MOBOConfig` dataclass also contains fields such as:
- `aq`
- `penalise`
- `batch`
- `plot_acquisition`
- `auto_length_scales`
- `gp_n_restarts_optimizer`
- `gp_length_scale_lower_factor`
- `gp_length_scale_upper_factor`
- `gp_length_scale_init_factor`

These exist in the config object, and some are used internally, but the current public `run_mobo(...)` and `load_txt_config(...)` paths do not provide them as user-configurable settings. In current public use:
- acquisition stays on `BATCH_UCB`
- GP length-scale controls are fixed to internal defaults
- acquisition plotting is effectively an internal toggle, not part of the public API

Outputs
---------
Results are written under:

    working_dir/Outputs/<save_name>/

The code saves, among other things:
- `_raw_repository.csv`
- `_raw_repository_pareto.csv`
- `_metrics_repository.csv`
- `_metadata.json`
- `_run_config.json`
- `_gp_models.pkl`
- `_goal_function_source.py`

The raw repository includes iteration index, inputs, objectives, objective errors, raw constraint values, processed constraint values, the scalar constraint value used, raw penalties, processed penalties, and total penalty.

Checkpoints are stored under:

    working_dir/Benchmarks/

Minimal Python example
----------------------
```python
from MOBO import run_mobo

repo, pf_repo, metrics_repo = run_mobo(
    inputs=inputs,
    objectives=objectives,
    constraints=constraints,
    penalties=penalties,
    training_csv_path="./Example_initial_samples.csv",
    evaluation_method="GOAL_FUNCTION",
    iterations=50,
    no_of_meas=10,
    batch_size=1,
    save_name="injector_scan",
    working_dir="./mobo_runs",
    use_real_error=True,
    default_objective_error=1e-6,
    goal_function_path="./Example_goal_function.py",
    goal_function_name="goal_function",
    goal_function_kwargs={"alpha": 0.5, "beta": 2},
    feasible_tol=0.0,
    constraint_penalty_alpha=1.0,
)
```

Practical recommendations
-------------------------
- Start with `batch_size=1` unless true parallel evaluation is available.
- Supply realistic per-objective errors whenever you can.
- Keep input bounds conservative for early machine studies.
- Use explicit constraints for protection or minimum acceptable operation.
- Use `constraint_penalty_alpha > 0` when infeasible evaluations are costly.
- Make `ref_value` conservative enough that obviously bad or invalid outputs can be clamped safely.

Current limitations and caveats
-------------------------------
- Only `MANUAL` and `GOAL_FUNCTION` are implemented in the current evaluator factory.
- The active candidate search is continuous; `step` is not enforced.
- The `scale` value in constraints is not normally applied unless `normalise=True` is present in the in-memory constraint dictionaries.
- GP hyperparameter controls exist in the config object but are not exposed by the public entry points

For full usage details, see `USER_MANUAL.txt`.
