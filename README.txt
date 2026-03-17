MOBO Optimiser README
=====================

This repository provides a multi-objective Bayesian optimisation (MOBO) workflow for accelerator tuning, beamline optimisation, and other expensive physics optimisation problems.

The code supports:
- Multi-objective optimisation with Gaussian Process surrogate models
- Feasibility-first handling of constraints
- Batch Upper Confidence Bound (BATCH_UCB) candidate selection
- Manual or automated evaluation modes
- Per-objective RMS/noise handling
- Checkpointing and reproducible run outputs

Main entry points
-----------------
- run_mobo(...)
- run_mobo_from_config(...)

Core behaviour
--------------
- All objectives are treated internally as minimisation targets.
- Maximisation objectives are negated internally.
- If feasible points exist, the Pareto front is built only from feasible points.
- If no feasible points exist, the least-infeasible points are used.
- The active acquisition path is BATCH_UCB with Sobol candidate generation.
- Constraints are combined using the maximum per-constraint violation.

Constraint penalty
------------------
If constraints are defined and constraint_penalty_alpha > 0, the optimiser fits an extra GP to the scalar constraint violation CV and applies:

    penalty_factor(x) = exp(alpha * max(0, cv_mu(x) - feasible_tol))

The penalised acquisition is:

    acq_penalised(x) = acq_raw(x) * penalty_factor(x)

This discourages predicted infeasible points while leaving predicted feasible points unchanged.

Initial sampling options
------------------------
The optimiser requires an initial dataset before Bayesian optimisation begins. This data is used to train the first Gaussian Process models.

Two modes are supported:

1. CSV-based initial data (recommended)
   Provide a CSV file via:

       training_csv_path = "path/to/file.csv"

   The file must contain:
   - Input columns (in the same order as defined in `inputs`)
   - Output columns ordered as:
         [objectives..., constraints...]

   The number of rows defines the number of initial samples.

2. Online initial sampling
   If no CSV is provided, the optimiser will generate and evaluate initial points:

       no_of_meas = N

   - N initial points are generated using Sobol sampling within the input bounds.
   - These points are evaluated using the selected evaluation method (MANUAL or GOAL_FUNCTION).
   - The resulting data is used to initialise the GP models.

Notes:
- The number of initial points should typically be at least:
  
      max(5 * number_of_inputs, 10)

- Poor initial coverage can significantly degrade optimisation performance.
- When using constraints, ensure that initial samples include at least some feasible points if possible.
- If `resume=True`, initial sampling is skipped and data is loaded from previous runs.

Recommended files to include with the code
------------------------------------------
- README.txt
- USER_MANUAL.txt
- Example_config.in
- Example_goal_function.py
- Example_MOBO_use.py

Typical workflow
----------------
1. Define inputs (name, min, max, step)
2. Define objectives (name, direction, ref_value)
3. Optionally define constraints (name, sense, threshold, scale)
4. Choose MANUAL or GOAL_FUNCTION mode
5. Set initial samples with no_of_meas
6. Run optimisation for the requested iterations

Minimal example
---------------
from MOBO import run_mobo

run_mobo(
    inputs=inputs,
    objectives=objectives,
    constraints=constraints,
    training_csv_path='/.../Example_initial_samples.csv',
    evaluation_method="GOAL_FUNCTION",
    iterations=20,
    no_of_meas=5,
    batch_size=1,
    save_name="injector_scan",
    working_dir="./mobo_runs",
    goal_function_path='./mobo_runs',
    goal_function_name='goal_function'
)

Notes
-----
- The input 'step' field is currently not strictly enforced in the active Sobol-based acquisition path.
- The active production acquisition path is BATCH_UCB.
- Objective GPs are independent (no joint multi-output GP).
- Use realistic RMS values whenever possible for robust machine operation.

For full details, see USER_MANUAL.txt
