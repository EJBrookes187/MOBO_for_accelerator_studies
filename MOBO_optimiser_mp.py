from abc import ABC, abstractmethod
import os
import numpy as np
from pymoo.indicators.hv import HV
from scipy.spatial.distance import cdist
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence, Tuple, Optional, List
import numpy as np
import pandas as pd
import logging
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.stats.qmc import Sobol
from scipy.stats import norm, multivariate_normal
import pickle
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import qmc
import time
import pandas as pd
import subprocess
import json

def _get_working_dir(config):
    wd = getattr(config, 'working_dir', None) or '.'
    return Path(wd)

def _get_outputs_dir(config):
    return _get_working_dir(config) / str(getattr(config, 'outputs_dirname', 'Outputs'))

def _get_benchmarks_dir(config):
    return _get_working_dir(config) / str(getattr(config, 'benchmarks_dirname', 'Benchmarks'))

def compute_pareto_front_constrained(Y, CV):
    """
    Feasibility-based Pareto front.
    Charge is negative: CV = max(0, Q - Q_threshold)
    """
    Y = np.asarray(Y)
    CV = np.asarray(CV)

    feasible = CV <= 1e-12
    infeasible = ~feasible

    # Case 1: feasible solutions exist
    if np.any(feasible):
        Y_feas = Y[feasible]
        idx_feas = np.where(feasible)[0]

        nd_local = NonDominatedSorting().do(
            Y_feas, only_non_dominated_front=True
        )

        pf_idx = idx_feas[nd_local]
        return Y[pf_idx], pf_idx

    # Case 2: all infeasible → minimum charge loss
    best = np.min(CV)
    pf_idx = np.where(CV == best)[0]
    return Y[pf_idx], pf_idx

        
def replace_inf_with_reference(Y: np.ndarray, reference_point: np.ndarray) -> np.ndarray:
    """
    Replace inf, -inf, or NaN values in Y with the reference point.
    """
    Y_clean = np.array(Y, copy=True, dtype=float)
    ref = np.asarray(reference_point).reshape(1, -1)

    # Replace inf/-inf/NaN
    bad_mask = ~np.isfinite(Y_clean)
    Y_clean[bad_mask] = np.take(ref.flatten(), np.where(bad_mask)[1])

    # Clamp values above reference
    too_high_mask = Y_clean > ref
    Y_clean[too_high_mask] = np.take(ref.flatten(), np.where(too_high_mask)[1])

    return Y_clean

def filter_previously_sampled(
    X_candidates: np.ndarray,
    X_existing: np.ndarray,
    tol: float = 1e-6
            ) -> np.ndarray:
    if X_existing is None or len(X_existing) == 0:
        return X_candidates

    dists = cdist(X_candidates, X_existing)
    mask = np.all(dists > tol, axis=1)
    return X_candidates[mask]

def extract_Y_CV(raw_outputs, objectives_spec, constraints_spec, method):
    method = str(method).upper()
    raw = np.asarray(raw_outputs, dtype=float)
    raw = np.atleast_2d(raw)

    n_obj = len(objectives_spec)
    n_con = len(constraints_spec)

    # strict shape validation to ensure constraints match config
    if method in ("MANUAL", "GOAL_FUNCTION", "GOAL", "FUNC", "FUNCTION"):
        expected = int(n_obj + n_con)
        if raw.shape[1] < expected:
            raise ValueError(
                f"Evaluator returned {raw.shape[1]} values per point, but config expects {expected} "
                f"(n_obj={n_obj}, n_con={n_con}). Ensure the goal function returns objectives + ALL constraints."
            )
    else:
        req = [int(o["index"]) for o in objectives_spec]
        if n_con:
            req += [int(c["index"]) for c in constraints_spec]
        if req:
            max_idx = int(np.max(req))
            if raw.shape[1] <= max_idx:
                raise ValueError(
                    f"Evaluator returned raw vector length {raw.shape[1]}, but config requests index {max_idx}. "
                    f"Ensure the evaluator returns all outputs required for objectives/constraints."
                )

    if method in ("MANUAL", "GOAL_FUNCTION"):
        # Manual returns [obj..., con...] in entered order
        Y_phys = raw[:, :n_obj]
        C = raw[:, n_obj:n_obj+n_con] if n_con else np.empty((raw.shape[0], 0))
    else:
        C = np.empty((raw.shape[0], 0))

    # convert max objectives to minimisation
    Y = Y_phys.copy()
    for j, o in enumerate(objectives_spec):
        if o["direction"].lower() == "max":
            Y[:, j] = -Y[:, j]

    # constraints to scalar CV
    if not constraints_spec:
        CV = np.zeros(raw.shape[0], dtype=float)
    else:
        CV = np.zeros(raw.shape[0], dtype=float)
        for j, c in enumerate(constraints_spec):
            thr = float(c["threshold"])
            sense = c["sense"].strip()
            v = C[:, j]
            if sense == "<=":
                viol = np.maximum(0.0, v - thr)
            elif sense == ">=":
                viol = np.maximum(0.0, thr - v)
            else:
                raise ValueError(f"Invalid constraint sense: {sense}")

            if bool(c.get("normalise", False)):
                scale = float(c.get("scale", 1.0) or 1.0)
                if scale <= 0.0:
                    scale = 1.0
                viol = viol / scale

            CV = np.maximum(CV, viol)

    return Y, CV



def map_train_to_pool_indices(X_train, X_pool, rtol=1e-10, atol=1e-12):
    """
    Returns pool_idx such that X_pool[pool_idx[i]] == X_train[i] (within tolerance).
    """
    X_train = np.asarray(X_train, float)
    X_pool  = np.asarray(X_pool, float)

    pool_idx = np.empty(X_train.shape[0], dtype=int)

    for i, x in enumerate(X_train):
        # find rows in pool close to this training point
        matches = np.where(np.all(np.isclose(X_pool, x, rtol=rtol, atol=atol), axis=1))[0]
        if len(matches) == 0:
            raise ValueError(f"Training point {i} not found in input_pool (try loosening atol/rtol). x={x}")
        pool_idx[i] = int(matches[0])  # if duplicates, take first

    return pool_idx

def map_points_to_grid_index(X_points, X_grid):
    X_points = np.asarray(X_points, float)
    X_grid = np.asarray(X_grid, float)
    idx = np.empty(X_points.shape[0], dtype=int)
    for i, x in enumerate(X_points):
        idx[i] = int(np.argmin(np.sum((X_grid - x) ** 2, axis=1)))
    return idx

from itertools import product

def build_discrete_grid(input_specs, order="input0_slowest"):
    levels = []

    # Build discrete values per dimension
    for s in input_specs:
        vmin = float(s["min"])
        vmax = float(s["max"])
        step = float(s["step"])

        # Number of steps
        n = int(np.floor((vmax - vmin) / step + 1e-12)) + 1
        arr = vmin + step * np.arange(n)

        # Ensure we don't overshoot due to floating point
        arr = arr[arr <= vmax + 1e-9]

        levels.append(arr)

    # Generate combinations in desired order
    if order == "input0_slowest":
        combos = list(product(*levels))
    elif order == "input0_fastest":
        combos = list(product(*levels[::-1]))
        combos = [c[::-1] for c in combos]
    else:
        raise ValueError("order must be 'input0_slowest' or 'input0_fastest'")

    X_grid = np.asarray(combos, dtype=float)

    return X_grid, levels

def region_exponential_penalty(values, lower, upper, scale=1.0, rate=5.0):
    values = np.asarray(values, dtype=float)

    d = np.zeros_like(values)

    below = values < lower
    above = values > upper

    d[below] = lower - values[below]
    d[above] = values[above] - upper

    penalty = np.zeros_like(values)
    mask = d > 0

    penalty[mask] = scale * (np.exp(rate * d[mask]) - 1.0)

    return penalty


def extract_Y_CV_details(raw_outputs, con_outputs, objectives_spec, constraints_spec, method, penalty_outputs=None, penalty_specs=None):
    """Like extract_Y_CV, but also returns per-constraint values and per-constraint violations.
    """
    method = str(method).upper()
    raw = np.asarray(raw_outputs, dtype=float)
    con = np.asarray(con_outputs, dtype=float)
    raw = np.atleast_2d(raw)
    con = np.atleast_2d(con)

    n_obj = len(objectives_spec)
    n_con = len(constraints_spec)
    penalty_specs = penalty_specs or []
    n_samples = raw.shape[0]

    total_penalty = np.zeros(n_samples)

    # strict shape validation to ensure constraints returned match config
    if method in ("MANUAL", "GOAL_FUNCTION", "GOAL", "FUNC", "FUNCTION"):
        expected = int(n_obj)
        if raw.shape[1] < expected:
            raise ValueError(
                f"Evaluator returned {raw.shape[1]} values per point, but config expects {expected} "
                f"(n_obj={n_obj}, n_con={n_con}). Ensure the goal function returns objectives + all constraints."
            )
    else:
        req = [int(o["index"]) for o in objectives_spec]
        # if n_con:
        #     req += [int(c["index"]) for c in constraints_spec]
        if req:
            max_idx = int(np.max(req))
            if raw.shape[1] <= max_idx:
                raise ValueError(
                    f"Evaluator returned raw vector length {raw.shape[1]}, but config requests index {max_idx}. "
                    f"Ensure the evaluator returns all outputs required for objectives/constraints."
                )

    # Manual returns [obj..., con...] in entered order
    Y_phys = raw # [:, :n_obj]
    C = con #raw[:, n_obj:n_obj+n_con] if n_con else np.empty((raw.shape[0], 0))


    # convert max objectives to minimisation
    Y = Y_phys.copy()
    for j, o in enumerate(objectives_spec):
        if o["direction"].lower() == "max":
            Y[:, j] = -Y[:, j]

    # constraints to per-constraint violation matrix + scalar CV
    if not constraints_spec:
        V = np.zeros((raw.shape[0], 0), dtype=float)
        CV = np.zeros(raw.shape[0], dtype=float)
    else:
        V = np.zeros((raw.shape[0], n_con), dtype=float)
        for j, c in enumerate(constraints_spec):
            thr = float(c["threshold"])
            sense = c["sense"].strip()
            v = C[:, j]

            if sense == "<=":
                viol = np.maximum(0.0, v - thr)
            elif sense == ">=":
                viol = np.maximum(0.0, thr - v)
            else:
                raise ValueError(f"Invalid constraint sense: {sense}")

            if bool(c.get("normalise", False)):
                scale = float(c.get("scale", 1.0) or 1.0)
                if scale <= 0.0:
                    scale = 1.0
                viol = viol / scale

            V[:, j] = viol
        CV = np.max(V, axis=1)

    penalty_specs = penalty_specs or []
    penalty_outputs = np.asarray(penalty_outputs, dtype=float) if penalty_outputs is not None else np.empty((raw.shape[0], 0))

    if penalty_specs:
        if penalty_outputs.shape[1] != len(penalty_specs):
            raise ValueError(
                f"Penalty output width {penalty_outputs.shape[1]} does not match number of penalties {len(penalty_specs)}."
            )
        P = np.zeros((raw.shape[0], len(penalty_specs)), dtype=float)
        for j, p in enumerate(penalty_specs):
            P[:, j] = region_exponential_penalty(
                penalty_outputs[:, j],
                lower=float(p["lower"]),
                upper=float(p["upper"]),
                scale=float(p.get("scale", 1.0)),
                rate=float(p.get("rate", 5.0))
            )
        total_penalty = np.sum(P, axis=1)
    else:
        P = np.empty((raw.shape[0], 0), dtype=float)
        total_penalty = np.zeros(raw.shape[0], dtype=float)

    # penalised objectives
    Y_pen = Y + total_penalty[:, None]

    return Y_pen, CV, C, V, Y_phys, penalty_outputs, P, total_penalty



@dataclass
class InitialSetupResult:
    input_repository: np.ndarray      
    output_repository: np.ndarray       
    input_pool: np.ndarray               
    output_pool: np.ndarray              
    best_pareto_front: np.ndarray         
    best_pareto_inputs: np.ndarray
    init_pareto_front: np.ndarray
    init_pareto_inputs: np.ndarray
    gp_models: List[GaussianProcessRegressor]
    input_scaler: Optional[StandardScaler]
    output_scaler: Optional[StandardScaler]

class InitialSetup():
    def __init__(self,
                 experiment,
                 csv_path: str,
                 input_columns: Sequence[str],
                 all_labels: Sequence[str],
                 objective_names: Sequence[int],
                 evaluator,
                 input_bounds: Optional[Sequence[Tuple[float, float]]] = None,
                 reference_point: Optional[np.ndarray] = None,
                 scale_data: bool = True,
                 random_seed: int = 0,
                 gp_alpha: float = 1e-4,
                 gp_n_restarts: int = 20,
                 grid_candidate_size: int = 2000,
                 nu=2.5,
                 Eve=2.5,
                 iterations=50):
        self.csv_path = Path(csv_path)
        self.input_columns = list(input_columns)
        self.all_labels = list(all_labels)
        self.objective_names = list(objective_names)
        self.evaluator = evaluator     
        self.input_bounds = input_bounds
        self.reference_point = np.asarray(reference_point) if reference_point is not None else None
        self.scale_data = bool(scale_data)
        self.rng = np.random.default_rng(random_seed)
        self.gp_alpha = gp_alpha
        self.gp_n_restarts = gp_n_restarts
        self.grid_candidate_size = int(grid_candidate_size)
        self.nu = nu
        self.Eve = Eve
        self.experiment = experiment
        self.plotter = MOBOPlotter()

        # GP length-scale history (saved each iteration for post-analysis)
        self.gp_length_scales_history = []  # list of (n_obj, n_dim) arrays
        self.cv_gp_length_scales_history = []  # list of (n_dim,) arrays (or nan)
        # Provide plotter with output root (working_dir/Outputs)
        try:
            self.plotter.base_output_dir = _get_outputs_dir(self.config)
            self.plotter.config = self.config
        except Exception:
            pass

        self.iterations = iterations

        self.input_pool = None
        self.output_pool = None
        self.constraint_pool = None
        self.constraint_value_pool = None
        self.constraint_violation_pool = None
        self.penalty_raw_pool = None
        self.input_scaler = None
        self.output_scaler = None

    def _validate(self):
        if self.input_bounds is not None and len(self.input_bounds) != len(self.input_columns):
            raise ValueError("input_bounds length must match number of input columns")

    def load_and_clean(self) -> None:

        logger = logging.getLogger("InitialSetup")
        logging.basicConfig(level=logging.INFO)
        p = Path(self.csv_path) if self.csv_path else None

        if p and p.exists() and p.is_file():
            df = pd.read_csv(self.csv_path)

            # Get problem dimensions
            input_cols = self.experiment.input_columns
            n_inputs = len(input_cols)

            n_obj = len(self.experiment.objectives_spec)
            n_con = len(self.experiment.constraints_spec)
            n_pen = len(self.experiment.penalties)

            if n_inputs == 0:
                raise ValueError(
                    "No input columns defined. Check [INPUTS]"
                )

            # Validate CSV shape
            expected_min_cols = n_inputs + n_obj + n_pen
            if df.shape[1] < expected_min_cols:
                raise ValueError(
                    f"CSV has too few columns: {df.shape[1]} < {expected_min_cols}"
                )

            # Extract data based on known layout
            # Expected:
            # Input1..N | obj1..M | error1..M | con1..K

            X = df.iloc[:, 0:n_inputs].values
            Y_phys = df.iloc[:, n_inputs:n_inputs + n_obj].values

            # error
            error_start = n_inputs + n_obj
            error_end = error_start + n_obj

            if df.shape[1] >= error_end:
                error = df.iloc[:, error_start:error_end].values
            else:
                error = np.full(
                    (len(df), n_obj),
                    self.experiment.config.default_objective_error,
                    dtype=float
                )

            # Constraints
            con_start = error_end
            con_end = con_start + n_con 
            pen_end = con_end+n_pen

            if n_pen > 0:
                pen_raw = df.iloc[:, con_end:pen_end].values
            else:
                pen_raw = np.empty((len(df), 0), dtype=float)
            if n_con > 0:
                C = df.iloc[:, con_start:con_end].values
            else:
                C = np.empty((len(df), 0), dtype=float)

            Y_pen, CV, C, V, Y_phys, penalty_raw, penalty_value, penalty_total = extract_Y_CV_details(
                raw_outputs=Y_phys,
                con_outputs=C,
                objectives_spec=self.experiment.objectives_spec,
                constraints_spec=self.experiment.constraints_spec,
                method=self.experiment.config.evaluation_method,
                penalty_outputs=pen_raw,
                penalty_specs=self.experiment.config.penalties
            )

            # Convert objectives (max → min)
            Y = Y_phys.copy()
            for j, o in enumerate(self.experiment.objectives_spec):
                if o["direction"] == "max":
                    Y[:, j] = -Y[:, j]

            # Final validation
            if X.shape[1] == 0:
                raise ValueError(
                    "CSV loading failed: 0 input features detected."
                )

            # Store
            self.input_pool = np.asarray(X, dtype=float)
            self.output_pool = np.asarray(Y, dtype=float)
            self.constraint_pool = np.asarray(CV, dtype=float)
            self.constraint_value_pool = np.asarray(C, dtype=float)
            self.constraint_violation_pool = np.asarray(V, dtype=float)
            self.error_pool = np.asarray(error, dtype=float)
            self.penalty_raw_pool = np.asarray(penalty_raw, dtype=float)
            self.penalty_value_pool = np.asarray(penalty_value, dtype=float)
            self.penalty_total_pool = np.asarray(penalty_total, dtype=float)

            logger.info(
                f"Loaded CSV: {len(self.input_pool)} rows, "
                f"{self.input_pool.shape[1]} inputs, "
                f"{self.output_pool.shape[1]} objectives, "
                f"{self.error_pool.shape[1]} stds, "
                f"{np.shape(self.penalty_raw_pool)[0]} penalties"
            )
            

            return

        if self.input_bounds is None:
            raise ValueError("input_bounds must be provided when not using a CSV file")

        input_cols = self.experiment.input_columns
        n_inputs = len(input_cols)
        bounds = np.array(self.input_bounds, dtype=float)

        N_POOL = self.experiment.config.no_of_meas
        self.rng = np.random.default_rng()

        X_pool = self.rng.uniform(bounds[:, 0], bounds[:, 1], size=(N_POOL, n_inputs))

        # Evaluate pool
        if hasattr(self.evaluator, "evaluate_batch"):
            raw_outputs, error_outputs, con_outputs, pen_raw_outputs = self.experiment.evaluator.evaluate_batch(X_pool)
        else:
            results = [self.evaluator.evaluate(x.reshape(1, -1)) for x in X_pool]
            raw_outputs = np.vstack([r[0].squeeze() for r in results])
            rcon_outputs = np.vstack([r[2].squeeze() for r in results])
            aux = None
            if len(results) > 0 and isinstance(results[0], (tuple, list)) and len(results[0]) > 1:
                aux = np.vstack([np.atleast_1d(r[1]).squeeze() for r in results])


        Y_pen_pool, CV_pool, C_pool, V_pool, Y_phys_pool, penalty_raw_pool, penalty_value_pool, penalty_total_pool = extract_Y_CV_details(
            raw_outputs,
            con_outputs,
            objectives_spec=self.experiment.objectives_spec,
            constraints_spec=self.experiment.constraints_spec,
            method=self.experiment.config.evaluation_method,
            penalty_outputs=pen_raw_outputs,
            penalty_specs=self.experiment.config.penalties)

        self.raw_pool = np.asarray(raw_outputs, dtype=float)
        self.experiment.constraint_pool = np.asarray(CV_pool, dtype=float)
        self.experiment.constraint_values_pool = np.asarray(C_pool, dtype=float)
        self.experiment.constraint_violations_pool = np.asarray(V_pool, dtype=float)
        self.penalty_raw_pool = np.asarray(penalty_raw, dtype=float)
        self.penalty_value_pool = np.asarray(penalty_value, dtype=float)
        self.penalty_total_pool = np.asarray(penalty_total, dtype=float)

        # error
        default_error = getattr(self.experiment.config, "default_objective_error", 1e-3)
        use_real_error = getattr(self.experiment.config, "use_real_error", True)

        n_pool = X_pool.shape[0]
        n_obj = Y_phys_pool.shape[1]

        if use_real_error==False:
            error_pool = np.full((n_pool, n_obj), default_error, dtype=float)
        else:
            error_pool = self.error_pool

        if str(self.experiment.config.evaluation_method).upper() == "MANUAL" and use_real_error and aux is not None:
            aux_arr = np.asarray(aux, dtype=float)
            aux_arr = np.atleast_2d(aux_arr)

            if aux_arr.shape == (n_pool, n_obj):
                error_pool = aux_arr
            elif aux_arr.shape == (n_obj,):
                error_pool[0, :] = aux_arr

        # Handle inf
        if self.reference_point is not None:
            Y_pen_pool = replace_inf_with_reference(Y_pen_pool, self.reference_point)

        # Store
        self.input_pool = X_pool
        self.output_pool = np.atleast_2d(Y_pen_pool)
        self.constraint_pool = np.asarray(CV_pool, dtype=float)
        self.constraint_value_pool = np.asarray(C_pool, dtype=float)
        self.constraint_violation_pool = np.asarray(V_pool, dtype=float)
        self.error_pool = np.asarray(error_pool, dtype=float)

        logger.info(
            f"Generated pool: {len(self.input_pool)} rows, "
            f"{self.input_pool.shape[1]} inputs, "
            f"{self.output_pool.shape[1]} objectives, "
            f"{self.error_pool.shape[1]} errors, "
            f"{self.constraint_pool.shape[1]} constraints, "
            f"{np.shape(self.penalty_raw_pool)[0]} penalties"
        )


    def choose_initial_samples(self, no_of_init_samples: int, ensure_min_distance: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        print('Reference points: ', self.reference_point)
        logger = logging.getLogger("InitialSetup")
    
        n_pool = len(self.input_pool)
    
        if no_of_init_samples > n_pool:
            raise ValueError("no_of_init_samples > number of available pool points")
    
        # choose indices
        chosen_idx = self.rng.choice(n_pool, size=no_of_init_samples, replace=False)
    
        if ensure_min_distance is not None and no_of_init_samples > 1:
            kept = []
            for idx in chosen_idx:
                x = self.input_pool[idx]
                if not kept:
                    kept.append(idx)
                else:
                    dists = np.linalg.norm(self.input_pool[kept] - x, axis=1)
                    if np.all(dists >= ensure_min_distance):
                        kept.append(idx)
                if len(kept) == no_of_init_samples:
                    break
    
            if len(kept) < no_of_init_samples:
                logger.warning("Could not enforce min distance for all initial samples - returning subset.")
            chosen_idx = np.array(kept)
    
        X0 = self.input_pool[chosen_idx, :]

        Y0 = self.output_pool[chosen_idx, :]
        CV0 = self.constraint_pool[chosen_idx]
        C0 = self.constraint_value_pool[chosen_idx]
        V0 = self.constraint_violation_pool[chosen_idx]
        PEN0 = self.penalty_raw_pool[chosen_idx]
        error0=self.error_pool[chosen_idx]


        # Store per-constraint values/violations + raw outputs for the chosen initial samples
        if hasattr(self, "constraint_values_pool"):
            self.constraint_values_init = np.asarray(self.experiment.constraint_values_pool)[chosen_idx, :]
        else:
            self.constraint_values_init = np.empty((len(chosen_idx), 0), dtype=float)

        if hasattr(self, "constraint_violations_pool"):
            self.constraint_violations_init = np.asarray(self.experiment.constraint_violations_pool)[chosen_idx, :]
        else:
            self.constraint_violations_init = np.empty((len(chosen_idx), 0), dtype=float)

        if hasattr(self, "penalty_raw_pool"):
            self.penalty_init = np.asarray(self.experiment.penalty_raw_pool)[chosen_idx, :]
        else:
            self.penalty_init = np.empty((len(chosen_idx), 0), dtype=float)

        if hasattr(self, "raw_pool"):
            self.raw_init = np.asarray(self.raw_pool)[chosen_idx, :]
        else:
            self.raw_init = np.empty((len(chosen_idx), 0), dtype=float)
        if self.reference_point is not None:
            Y0 = replace_inf_with_reference(Y0, self.reference_point)

        #print('c0: ',C0)

        return X0, Y0, CV0, error0, C0, V0, PEN0


    def clamp_outputs_to_reference(self, Y: np.ndarray, reference: np.ndarray) -> np.ndarray:
        ref = np.asarray(reference).reshape(1, -1)
        mask = np.any(Y > ref, axis=1)
        Y_clamped = Y.copy()
        Y_clamped[mask, :] = ref
        return Y_clamped

    def build_gp_models(self, input_dim: int, n_objectives: int) -> List[GaussianProcessRegressor]:
        if self.input_bounds is not None:
            ranges = np.array([b[1] - b[0] for b in self.input_bounds])
            init_ls = (ranges / 4.0).tolist()
            bounds = [(r / 1e3, r * 1e3) for r in ranges]
        else:
            init_ls = [1.0] * input_dim
            bounds = [(1e-5, 1e5)] * input_dim
        models = []
        for _ in range(n_objectives):
            kernel = Matern(length_scale=init_ls, length_scale_bounds=bounds, nu=self.nu)
            gpr = GaussianProcessRegressor(kernel=kernel, alpha=self.gp_alpha,
                                           n_restarts_optimizer=self.gp_n_restarts,
                                           normalize_y=True)
            models.append(gpr)
        return models
    
    def _use_config_initial_samples(self):
        samples = self.experiment.config.initial_samples

        n_obj = len(self.experiment.objectives_spec)
        n_con = len(self.experiment.constraints_spec)
        n_pen = len(self.experiment.penalties)

        X = np.asarray([s["X"] for s in samples], dtype=float)
        Y_phys = np.asarray([s["Y"] for s in samples], dtype=float)

        C = (
            np.asarray(
                [s["C"] if s.get("C") is not None else [np.nan] * n_con for s in samples],
                dtype=float
            )
            if n_con > 0 else np.empty((len(samples), 0), dtype=float)
        )

        error = (
            np.asarray(
                [s["error"] if s.get("error") is not None else [self.experiment.config.default_objective_error] * n_obj for s in samples],
                dtype=float
            )
            if n_obj > 0 else np.empty((len(samples), 0), dtype=float)
        )

        pen_raw = (
            np.asarray(
                [s["P"] if s.get("P") is not None else [0.0] * n_pen for s in samples],
                dtype=float
            )
            if n_pen > 0 else np.empty((len(samples), 0), dtype=float)
        )

        Y_pen, CV, C, V, Y_phys, penalty_raw, penalty_value, penalty_total = extract_Y_CV_details(
            raw_outputs=Y_phys,
            con_outputs=C,
            objectives_spec=self.experiment.objectives_spec,
            constraints_spec=self.experiment.constraints_spec,
            method=self.experiment.config.evaluation_method,
            penalty_outputs=pen_raw,
            penalty_specs=self.experiment.config.penalties
        )

        Y = Y_phys.copy()
        for j, o in enumerate(self.experiment.objectives_spec):
            if o["direction"] == "max":
                Y[:, j] = -Y[:, j]

        self.input_pool = X
        self.output_pool = Y
        self.constraint_pool = CV
        self.constraint_value_pool = C
        self.constraint_violation_pool = V
        self.error_pool = error
        self.penalty_raw_pool = penalty_raw
        self.penalty_value_pool = penalty_value
        self.penalty_total_pool = penalty_total

        return X, Y, CV, error, C, V, penalty_raw

    def run(self, no_of_init_samples: int = 10, ensure_min_distance: Optional[float] = None) -> InitialSetupResult:
        logger = logging.getLogger("InitialSetup")
        self.load_and_clean()

        if self.csv_path and Path(self.csv_path).exists():
            self.load_and_clean()
            X_init = self.input_pool
            Y_init = self.output_pool
            expected_n_obj = len(self.experiment.objectives_spec)

            if Y_init.shape[1] != expected_n_obj:
                raise ValueError(
                    f"Mismatch: Y_init has {Y_init.shape[1]} objectives, "
                    f"but config defines {expected_n_obj}"
                )
            CV_init = self.constraint_pool
            C_init = self.constraint_value_pool
            V_init = self.constraint_violation_pool
            Pen_init = self.penalty_raw_pool
            error_init = self.error_pool
            no_of_init_samples = len(Y_init)

        elif self.experiment.config.initial_samples:
            X_init, Y_init, CV_init, error_init, C_init, V_init, Pen_init = self._use_config_initial_samples()
            expected_n_obj = len(self.experiment.objectives_spec)

            if Y_init.shape[1] != expected_n_obj:
                raise ValueError(
                    f"Mismatch: Y_init has {Y_init.shape[1]} objectives, "
                    f"but config defines {expected_n_obj}"
                )
            no_of_init_samples = len(Y_init)

        else:
            self.load_and_clean()
            X_init, Y_init, CV_init, error_init, C_init, V_init, Pen_init = self.choose_initial_samples(no_of_init_samples, ensure_min_distance=ensure_min_distance)

            expected_n_obj = len(self.experiment.objectives_spec)

            if Y_init.shape[1] != expected_n_obj:
                raise ValueError(
                    f"Mismatch: Y_init has {Y_init.shape[1]} objectives, "
                    f"but config defines {expected_n_obj}"
                )
        #print('c_init3: ',C_init)
        
        self.experiment.Y_init=Y_init

        pf, pf_idx = compute_pareto_front_constrained(Y_init, CV_init)

        labels = [o["name"] for o in self.experiment.objectives_spec]
        directions = [o["direction"] for o in self.experiment.objectives_spec]

        self.plotter.plot_pareto_front_colourmap(
            Y_init=Y_init,
            Y=Y_init,
            pf_idx=pf_idx,
            objective_labels=labels,
            objective_directions=directions,
            save_name=self.experiment.config.save_name,
            CV=CV_init
        )
        if Y_init.shape[1] == 3:
            self.plotter.plot_3obj_pareto_physical_axes(
                Y_init, pf_idx, self.objective_names,
                title=str(self.experiment.config.working_dir) + "/Outputs/_3D_initial_PF"
            )

        # Reference point
        if self.reference_point is None:
            self.reference_point = np.max(self.output_pool, axis=0) + 1e6
            logger.info(f"No reference_point supplied; using default {self.reference_point}")
        Y_init = self.clamp_outputs_to_reference(Y_init, self.reference_point)
        expected_n_obj = len(self.experiment.objectives_spec)

        if Y_init.shape[1] != expected_n_obj:
            raise ValueError(
                f"Mismatch: Y_init has {Y_init.shape[1]} objectives, "
                f"but config defines {expected_n_obj}"
            )

        # Scaling
        if self.scale_data:
            self.input_scaler = StandardScaler().fit(self.input_pool)
            self.output_scaler = StandardScaler().fit(self.output_pool)
            # X_init_scaled = self.input_scaler.transform(X_init)
            # Y_init_scaled = self.output_scaler.transform(Y_init)
            logger.info("Input & output scalers fitted from pool (for GPs).")
        else:
            self.input_scaler = None
            self.output_scaler = None
            # X_init_scaled, Y_init_scaled = X_init, Y_init

        # Build GP objects
        # print(Y_init.shape[1])
        gp_models = self.build_gp_models(input_dim=X_init.shape[1], n_objectives=Y_init.shape[1])

        #- error handling (real vs default)-
        default_error = getattr(self.experiment.config, "default_objective_error", 1e-3)
        use_real_error = getattr(self.experiment.config, "use_real_error", True)

        # Guarantee shape (n_init, n_obj)
        n_init = X_init.shape[0]
        self.experiment.iteration_repository = np.zeros(n_init, dtype=int)
        for i in range(len(self.experiment.iteration_repository)):
            self.experiment.iteration_repository[i]=-1
        n_obj = Y_init.shape[1]
        # print(n_obj)
        if (not use_real_error) or (error_init is None):
            error_init = np.full((n_init, n_obj), default_error, dtype=float)
        else:
            error_init = np.asarray(error_init, dtype=float)
            if error_init.shape != (n_init, n_obj):
                error_init = np.full((n_init, n_obj), default_error, dtype=float)
        # Store repositories on experiment (so optimiser can fit GPs with alpha=error^2)
        self.experiment.input_repository = X_init
        self.experiment.output_repository = Y_init
        # self.experiment.constraint_repository = np.asarray(CV_init, dtype=float)
        self.experiment.error_repository = error_init

        # Store full raw outputs + per-constraint values/violations for post-analysis
        self.experiment.raw_repository = getattr(self, "raw_init", np.empty((len(X_init), 0), dtype=float))
        self.experiment.constraint_repository = np.asarray(CV_init).reshape(-1)
        self.experiment.constraint_values_repository = np.asarray(C_init)
        self.experiment.constraint_violations_repository = np.asarray(V_init)
        self.experiment.penalty_raw_repository = np.asarray(Pen_init, dtype=float)
        self.experiment.penalty_value_repository = np.asarray(self.penalty_value_pool[:len(X_init)], dtype=float)
        self.experiment.penalty_total_repository = np.asarray(self.penalty_total_pool[:len(X_init)], dtype=float)
        

        best_pf, best_idx = compute_pareto_front_constrained(self.output_pool, self.constraint_pool)
        init_pf, init_idx = compute_pareto_front_constrained(Y_init, CV_init)

        return InitialSetupResult(
            input_repository=X_init,
            output_repository=Y_init,
            input_pool=self.input_pool,
            output_pool=self.output_pool,
            best_pareto_front=best_pf,
            best_pareto_inputs=self.input_pool[best_idx],
            init_pareto_front=init_pf,
            init_pareto_inputs=X_init[init_idx],
            gp_models=gp_models,
            input_scaler=self.input_scaler,
            output_scaler=self.output_scaler
        )

class Optimiser(ABC):
    def __init__(self, experiment):
        self.experiment = experiment
        self.adjustable_beta = experiment.config.adjustable_beta
        self.iterations = experiment.config.iterations
        self.eve = experiment.config.eve
        self.aq = experiment.config.aq
        self.weight = experiment.config.weight
        self.reference_point = experiment.reference_point
        self.input_repository = experiment.input_repository
        self.output_repository = experiment.output_repository
        self.error_repository = experiment.error_repository
        self.penalty_raw_repository = np.empty(0)
        self.input_bounds = experiment.config.input_bounds
        self.nu = experiment.config.nu
        self.gp_models = self._build_gp_models()
        self.beta = 2.5
        self.constraint_repository = self.experiment.constraint_repository
        self.constraint_values_repository = self.experiment.constraint_values_repository
        self.constraint_violations_repository = self.experiment.constraint_violations_repository
        self.config = experiment.config
        self.Y_init=self.experiment.Y_init
        


class BatchOptimiser(Optimiser):
    def __init__(self, experiment):
        self.experiment = experiment
        self.evaluation_method = experiment.config.evaluation_method
        self.iterations = experiment.config.iterations
        self.aq = experiment.config.aq
        self.weight = experiment.config.weight
        self.reference_point = experiment.reference_point
        self.input_repository = experiment.input_repository
        self.output_repository = experiment.output_repository
        self.constraint_repository = experiment.constraint_repository
        self.constraint_values_repository = experiment.constraint_values_repository
        self.constraint_violations_repository = experiment.constraint_violations_repository
        self.error_repository = experiment.error_repository
        self.penalty_value_repository = experiment.penalty_value_repository
        self.penalty_total_repository = experiment.penalty_total_repository
        self.multiprocess_bool=experiment.multiprocess_bool

        # Extra repositories for full post-analysis (per-constraint values/violations + raw outputs)
        self.constraint_values_repository = getattr(experiment, "constraint_values_repository", None)
        self.constraint_violations_repository = getattr(experiment, "constraint_violations_repository", None)
        self.raw_repository = getattr(experiment, "raw_repository", None)
        self.input_bounds = experiment.config.input_bounds
        self.batch_size = experiment.config.batch_size
        self.nu = experiment.config.nu
        self.config = experiment.config
        self.gp_models = self._build_gp_models()
        self.beta = 2.5
        self.logger = logging.getLogger("MOBO_outputs")
        self.plotter = MOBOPlotter()
        n_pen = len(self.experiment.penalties or [])
        self.penalty_raw_repository = np.empty((0, n_pen), dtype=float)
        self.penalty_value_repository = np.empty((0, n_pen), dtype=float)
        self.penalty_total_repository = np.empty((0,), dtype=float)

        # GP length-scale history (saved each iteration for post-analysis)
        self.gp_length_scales_history = []  # list of (n_obj, n_dim) arrays
        self.cv_gp_length_scales_history = []  # list of (n_dim,) arrays (or nan)

    def _build_gp_models(self):
        """
        Build one Gaussian Process model per objective.
        """
        n_outputs = self.output_repository.shape[1]
        gp_models = []

        # Automatic GP length-scale handling
        # Hyperparameters (including length_scales) are re-optimised on every gp.fit() call via sklearn's optimizer.
        X0 = np.asarray(self.input_repository, dtype=float)
        d = X0.shape[1]
        ranges = np.ptp(X0, axis=0)
        ranges = np.where(ranges <= 0.0, 1.0, ranges)

        # Config toggles (defaults to automatic behaviour)
        auto_ls = bool(getattr(self.experiment.config, "auto_length_scales", True))
        n_restarts = int(getattr(self.experiment.config, "gp_n_restarts_optimizer", 5))
        ls_lower = float(getattr(self.experiment.config, "gp_length_scale_lower_factor", 1e-6))
        ls_upper = float(getattr(self.experiment.config, "gp_length_scale_upper_factor", 10.0))
        ls_init_factor = float(getattr(self.experiment.config, "gp_length_scale_init_factor", 0.2))

        if auto_ls:
            initial_length_scales = (ls_init_factor * ranges).tolist()
            length_scale_bounds_revisited = np.array([(ls_lower * r, ls_upper * r) for r in ranges], dtype=float)
        else:
            # Backwards compatible hard-coded defaults (will be truncated/extended to match dimensionality)
            initial_length_scales = [0.01, 0.01, 0.1, 0.1, 0.1, 0.01]
            if len(initial_length_scales) != d:
                if len(initial_length_scales) > d:
                    initial_length_scales = initial_length_scales[:d]
                else:
                    initial_length_scales = (initial_length_scales + [initial_length_scales[-1]] * (d - len(initial_length_scales)))
            length_scale_bounds_revisited = np.array([(1e-7, 1.0)] * d, dtype=float)

        for i in range(n_outputs):
            X = self.input_repository
            y = self.output_repository[:, i]
            kernel = Matern(length_scale=initial_length_scales, length_scale_bounds=length_scale_bounds_revisited, nu=self.nu)
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts, normalize_y=True)
            gp.fit(X, y)
            gp_models.append(gp)
        return gp_models

    @staticmethod
    def _extract_length_scales_from_kernel(kernel, n_dim: int):
        """Best-effort extraction of per-dimension length scales from a fitted sklearn kernel
        Returns a 1D array, or None if not found.
        """
        try:
            # Direct attribute on kernels like Matern/RBF
            if hasattr(kernel, "length_scale"):
                ls = np.asarray(getattr(kernel, "length_scale"), dtype=float)
                if ls.ndim == 0:
                    return np.full(n_dim, float(ls), dtype=float)
                if ls.size == n_dim:
                    return ls.reshape(-1)
                # If mismatch, try to broadcast a single value
                if ls.size == 1:
                    return np.full(n_dim, float(ls.reshape(-1)[0]), dtype=float)
                return None

            # Recurse into composed kernels
            for child_name in ("k1", "k2", "base_kernel"):
                if hasattr(kernel, child_name):
                    child = getattr(kernel, child_name)
                    out = BatchOptimiser._extract_length_scales_from_kernel(child, n_dim)
                    if out is not None:
                        return out
        except Exception:
            return None
        return None


    def save_checkpoint(self, iteration, filename):
        """Save optimiser state to a file."""
        state = {"iteration": iteration,
            "input_repository": self.input_repository,
            "output_repository": self.output_repository,
            "error_repository": self.error_repository,
            "penalty_raw_repository": self.penalty_raw_repository,
            "constraint_repository": self.constraint_repository,
            "constraint_values_repository": self.constraint_values_repository,
            "constraint_violaiton_repository": self.constraint_violations_repository,
            "hypervolume": self.HV,
            "GD": self.GD,
            "diversity": self.diversity,
            "Spacing": self.spacing,
            "count": self.count,
            "gp_models": self.gp_models, 
            "runtime": self.runtime_records,
            "gp_length_scales_history": getattr(self, "gp_length_scales_history", []),
            "cv_gp_length_scales_history": getattr(self, "cv_gp_length_scales_history", []),
            "config": self.config}
        with open(filename, "wb") as f:
            pickle.dump(state, f)
        logging.info(f"[Checkpoint] Saved at iteration {iteration} -> {filename}")

    def load_checkpoint(self, filename):
        """Load optimiser state from a file."""
        if not os.path.exists(filename):
            logging.info("[Checkpoint] No checkpoint found.")
            return 0  # start from scratch

        with open(filename, "rb") as f:
            state = pickle.load(f)
        self.input_repository = state["input_repository"]
        self.output_repository = state["output_repository"]
        self.constraint_repository = state.get("constraint_repository", None)
        self.constraint_values_repository = state.get("constraint_values_repository", None)
        self.constraint_violations_repository = state.get("constraint_violations_repository", None)
        self.error_repository = state.get("error_repository", None)
        self.penalty_raw_repository = state.get("penalty_raw_repository",None)
        if self.constraint_repository is not None:
            self.experiment.constraint_repository = self.constraint_repository
        if self.constraint_values_repository is not None:
            self.experiment.constraint_values_repository = self.constraint_values_repository
        if self.constraint_violations_repository is not None:
            self.experiment.constraint_violations_repository = self.constraint_violations_repository
        self.HyperV = state['hypervolume']
        self.GD = state['GD']
        self.Diversity = state['diversity']
        self.Spacing = state['Spacing']
        self.Count = state['count']
        self.gp_models = state["gp_models"]
        self.config = state["config"]
        self.runtime_records = state["runtime"]
        self.gp_length_scales_history = state.get("gp_length_scales_history", [])
        self.cv_gp_length_scales_history = state.get("cv_gp_length_scales_history", [])
        logging.info(f"[Checkpoint] Loaded from {filename} at iteration {state['iteration']}")
        return state["iteration"]

    def run(self, resume=True):
        start_iter = self.config.restart_from_iteration
        save_name = self.config.save_name
        if resume:
            try:
                checkpoint = self.load_checkpoint(str(_get_benchmarks_dir(self.config) / f"{save_name}.pkl"))
                # pf, pf_idx = compute_pareto_front(self.output_repository)
                self.experiment.input_repository = self.input_repository
                self.experiment.output_repository = self.output_repository
                self.experiment.constraint_repository = self.constraint_repository
                self.experiment.constraint_values_repository = self.constraint_values_repository
                self.experiment.constraint_violations_repository = self.constraint_violations_repository
                self.experiment.error_repository = self.error_repository
                self.experiment.penalty_raw_repository = self.penalty_raw_repository
                

                # Optional: truncate history and restart from a specific iteration (0-based) (use restart=True to resume from the last completed iteration).
                #ri = getattr(self.config, "restart_from_iteration", None)
                if start_iter>0:
                    print('Restarting from iteration')
                    try:
                        start_iter = int(start_iter)
                        if start_iter < 0:
                            start_iter = 0
                        n_init = int(getattr(self.config, "no_of_meas", 0) or 0)
                        bs = int(getattr(self.config, "batch_size", 1) or 1)
                        n_keep = n_init + start_iter * bs
                        n_keep = min(n_keep, self.input_repository.shape[0])

                        self.input_repository = self.input_repository[:n_keep]
                        self.output_repository = self.output_repository[:n_keep]
                        if self.constraint_repository is not None:
                            self.constraint_repository = np.asarray(self.constraint_repository)[:n_keep]#.reshape(-1, len(self.experiment.constraint_specs))
                        if self.constraint_values_repository is not None:
                            self.constraint_values_repository = np.asarray(self.constraint_values_repository)[:n_keep]#.reshape(-1, len(self.experiment.constraint_specs))
                        if self.constraint_violations_repository is not None:
                            self.constraint_violations_repository = np.asarray(self.constraint_violations_repository)[:n_keep]#.reshape(-1, len(self.experiment.constraint_specs))
                        if self.error_repository is not None:
                            self.error_repository = np.asarray(self.error_repository)[:n_keep]
                        if self.penalty_raw_repository is not None:
                            self.penatly_repository = np.asarray(self.penalty_raw_repository)[:n_keep]

                        self.experiment.input_repository = self.input_repository
                        self.experiment.output_repository = self.output_repository
                        self.experiment.constraint_repository = self.constraint_repository
                        self.experiment.constraint_values_repository = self.constraint_values_repository
                        self.experiment.constraint_violations_repository = self.constraint_violations_repository
                        self.experiment.error_repository = self.error_repository
                        self.experiment.penalty_raw_repository = np.vstack([self.penalty_raw_repository, penalty_outputs_new])
                        self.experiment.penalty_value_repository = np.vstack([self.penalty_value_repository, P_new])
                        self.experiment.penalty_total_repository = np.hstack([self.penalty_total_repository, total_penalty_new])

                        # metrics history (if present)
                        if isinstance(getattr(self, "HyperV", None), list):
                            self.HyperV = self.HyperV[:start_iter]
                        if isinstance(getattr(self, "GD", None), list):
                            self.GD = self.GD[:start_iter]
                        if isinstance(getattr(self, "Diversity", None), list):
                            self.Diversity = self.Diversity[:start_iter]
                        if isinstance(getattr(self, "Spacing", None), list):
                            self.Spacing = self.Spacing[:start_iter]
                        if isinstance(getattr(self, "Count", None), list):
                            self.Count = self.Count[:start_iter]
                        if isinstance(getattr(self, "runtime_records", None), list):
                            self.runtime_records = self.runtime_records[:start_iter]

                        start_iter = start_iter
                        logging.info(f"[Restart] Truncated state to iteration {start_iter} (rows kept: {n_keep}).")
                    except Exception as e:
                        logging.warning(f"[Restart] Could not apply restart_from_iteration={start_iter}: {e}")
                pf, pf_idx = compute_pareto_front_constrained(self.output_repository, self.experiment.constraint_repository)
                pf_input = self.input_repository[pf_idx]
                HyperV = self.HyperV
                GD = self.GD
                Diversity = self.Diversity
                Spacing = self.Spacing
                Count = self.Count
                runtime_records = self.runtime_records
            except:
                print('No checkpoint file located - starting from scratch')
                HyperV = []
                GD = []
                Diversity = []
                Spacing = []
                Count = []
                runtime_records = []
                self.HV = []
                self.GD = []
                self.Diversity = []
                self.Spacing = []
                self.Count = []
                self.runtime_records = []
        else:
            HyperV = []
            GD = []
            Diversity = []
            Spacing = []
            Count = []
            runtime_records = []
            self.HV = []
            self.GD = []
            self.Diversity = []
            self.Spacing = []
            self.Count = []
            self.runtime_records = []

        for i in range(start_iter, self.iterations):
            start = time.perf_counter()
            iter_col = np.full(self.input_repository.shape[0], i+1, dtype=int)
            self.experiment.iteration_repository = np.hstack([self.experiment.iteration_repository, i])
            
            acq = AcquisitionFactory.create(self.aq, beta=self.config.nu, weights=self.weight, reference_point=self.experiment.reference_point, n_samples=5000)

            # Fit Gaussian Processes
            logging.info(f"Iteration {i} - Fitting GP")
            default_error = getattr(self.experiment.config, "default_objective_error", 1e-3)

            print(self.output_repository)
            
            for j, gp in enumerate(self.gp_models):
                y = self.output_repository[:, j]

                if hasattr(self, "error_repository") and self.error_repository.shape[0] == len(y):
                    # print(self.error_repository)
                    error = self.error_repository[:, j]
                else:
                    error = np.full_like(y, default_error, dtype=float)

                alpha = np.clip(error**2, 1e-12, np.inf)   # variance, avoid zeros
                gp.alpha = alpha
                gp.fit(self.input_repository, y) # Fit the GP models to the available data

            cv_gp = None
            try:
                alpha = float(getattr(self.experiment.config, "constraint_penalty_alpha", 0.0) or 0.0)
                if getattr(self.experiment, "constraints_spec", None):
                    CV_train = np.asarray(self.experiment.constraint_repository, dtype=float).reshape(-1)
                    if CV_train.size == self.input_repository.shape[0]:
                        cv_gp = GaussianProcessRegressor(
                            kernel=Matern(nu=2.5),
                            alpha=1e-6,
                            normalize_y=True
                        )
                        cv_gp.fit(self.input_repository, CV_train)
            except Exception as e:
                logging.warning(f"Could not fit CV GP for penalised acquisition: {e}")


            try:
                n_dim = int(self.input_repository.shape[1])
                n_obj = int(len(self.gp_models))
                ls_mat = np.full((n_obj, n_dim), np.nan, dtype=float)
                for jj, gp in enumerate(self.gp_models):
                    kern = getattr(gp, "kernel_", None)
                    if kern is None:
                        kern = getattr(gp, "kernel", None)
                    ls = self._extract_length_scales_from_kernel(kern, n_dim)
                    if ls is not None:
                        ls_mat[jj, :] = ls
                self.gp_length_scales_history.append(ls_mat)

                if cv_gp is not None:
                    cv_kern = getattr(cv_gp, "kernel_", None)
                    if cv_kern is None:
                        cv_kern = getattr(cv_gp, "kernel", None)
                    cv_ls = self._extract_length_scales_from_kernel(cv_kern, n_dim)
                    if cv_ls is None:
                        cv_ls = np.full(n_dim, np.nan, dtype=float)
                else:
                    cv_ls = np.full(n_dim, np.nan, dtype=float)
                self.cv_gp_length_scales_history.append(np.asarray(cv_ls, dtype=float).reshape(-1))
            except Exception as e:
                logging.warning(f"Could not record GP length scales at iteration {i}: {e}")

            
            # Select batch of candidates and evaluate
            X_new = filter_previously_sampled(self.experiment.input_repository, self.input_repository, tol=1e-6)
            X_new, UCB_values = acq.select_candidates(
                gp_models=self.gp_models,
                pareto_front=self.experiment.best_pareto_front,
                input_bounds=self.input_bounds,
                X_existing=self.experiment.input_repository,
                n_candidates=self.batch_size,
                cv_gp=cv_gp,
                feasible_tol=getattr(self.experiment.config, 'feasible_tol', 0.0),
                constraint_penalty_alpha=getattr(self.experiment.config, 'constraint_penalty_alpha', 0.0),
           
                weights=(self.weight if bool(getattr(self.experiment.config, 'weighting', False)) else None),
            )
            # Plot acquisition values for this iteration
            if getattr(self.experiment.config, 'plot_acquisition', False):
                try:
                    info = UCB_values if isinstance(UCB_values, dict) else {'acq_raw': np.asarray(UCB_values)}
                    self.plotter.plot_acquisition(iteration=i, save_prefix=self.experiment.config.save_name, info=info)
                except Exception as e:
                    logging.warning(f'Acquisition plotting failed: {e}')

            raw_new, error_new, con_new, pen_raw_new = self.experiment.evaluator.evaluate_batch(X_new)
            # Y_new, CV_new, C_new, V_new, Y_phys_new, Y_pen_new = extract_Y_CV_details(raw_new, con_new, objectives_spec=self.experiment.objectives_spec, constraints_spec=self.experiment.constraints_spec, method=self.experiment.config.evaluation_method, penalty_specs=self.experiment.config.penalties)
            Y_pen_new, CV_new, C_new, V_new, Y_phys_new, penalty_outputs_new, P_new, total_penalty_new = extract_Y_CV_details(
                    raw_new, con_new,
                    objectives_spec=self.experiment.objectives_spec,
                    constraints_spec=self.experiment.constraints_spec,
                    method=self.experiment.config.evaluation_method,
                    penalty_specs=self.experiment.config.penalties,
                    penalty_outputs=pen_raw_new
                )

            labels = [o["name"] for o in self.experiment.objectives_spec]

            
            # Add to repositories
            self.experiment.input_repository = np.vstack([self.input_repository, X_new])
            Y_new = Y_phys_new.copy()
            for j, o in enumerate(self.experiment.objectives_spec):
                if o["direction"] == "max":
                    Y_new[:,j]=-Y_new[:,j]
            self.experiment.output_repository = np.vstack([self.output_repository, Y_new])
            self.experiment.constraint_repository = np.hstack([self.constraint_repository, CV_new])
            self.experiment.constraint_values_repository = np.vstack([self.constraint_values_repository, C_new])
            self.experiment.constraint_violations_repository = np.vstack([self.constraint_violations_repository, V_new])
            self.experiment.error_repository = np.vstack([self.error_repository,error_new])
            self.experiment.penalty_raw_repository = np.vstack([self.experiment.penalty_raw_repository,penalty_outputs_new])
            self.experiment.penalty_value_repository = np.vstack([self.experiment.penalty_value_repository, P_new])
            self.experiment.penalty_total_repository = np.hstack([self.experiment.penalty_total_repository, total_penalty_new])


            # Append extra repositories (raw outputs + per-constraint info)
            try:
                if getattr(self.experiment, "raw_repository", None) is None or np.size(getattr(self.experiment, "raw_repository", np.empty((0,0)))) == 0:
                    self.experiment.raw_repository = np.asarray(raw_new, dtype=float)
                else:
                    self.experiment.raw_repository = np.vstack([self.experiment.raw_repository, np.asarray(raw_new, dtype=float)])
            except Exception:
                pass


            self.input_repository = self.experiment.input_repository
            self.output_repository = self.experiment.output_repository
            self.constraint_repository = self.experiment.constraint_repository
            self.constraint_values_repository = self.experiment.constraint_values_repository
            self.constraint_violations_repository = self.experiment.constraint_violations_repository
            self.constraint_values_repository = self.experiment.constraint_values_repository
            self.constraint_violations_repository = self.experiment.constraint_violations_repository
            self.error_repository = self.experiment.error_repository

            self.constraint_values_repository = getattr(self.experiment, "constraint_values_repository", None)
            self.constraint_violations_repository = getattr(self.experiment, "constraint_violations_repository", None)
            self.raw_repository = getattr(self.experiment, "raw_repository", None)

            elapsed = time.perf_counter() - start
            

            # Update metrics 
            pf, pf_idx = compute_pareto_front_constrained(self.output_repository, self.experiment.constraint_repository)
            pf_input = self.input_repository[pf_idx]
            pf_error = self.error_repository[pf_idx]
            hv = Metrics.hypervolume(pf, self.experiment.reference_point)
            gd = Metrics.generational_distance(pf, self.experiment.best_pareto_front)
            div = Metrics.diversity(pf)
            spacing = Metrics.spacing(pf)
            count = Metrics.num_pf_points(pf)
            goal_kwargs_json = json.dumps(self.experiment.config.goal_function_kwargs, sort_keys=True, default=str)
            HyperV.append(hv)
            GD.append(gd)
            Diversity.append(div)
            Spacing.append(spacing)
            Count.append(count)
            runtime_records.append(elapsed)
            print({"iteration": i, "elapsed_s": elapsed})

            self.HV = HyperV
            self.GD = GD
            self.diversity = Diversity
            self.spacing = Spacing
            self.count = Count
            self.runtime_records = runtime_records

            labels = [o["name"] for o in self.experiment.objectives_spec]
            directions = [o["direction"] for o in self.experiment.objectives_spec]
            self.plotter.plot_pareto_front_colourmap(Y_init=self.experiment.Y_init,Y=self.output_repository,pf_idx=pf_idx,objective_labels=labels,objective_directions=directions,save_name=self.experiment.config.save_name,CV=self.experiment.constraint_repository)
            # self.plotter.plot_3obj_pareto_physical_axes(self.output_repository, pf_idx, labels, title=str(self.experiment.config.working_dir)+'/Outputs/3D_initial_PF')
            self.plotter.plot_hypervolume_evolution(self.HV, self.evaluation_method,  str(_get_outputs_dir(self.config)) + "/", i, self.config.save_name)

            self.logger.info(f"Iter {i}: HV={hv:.3f}, X_new={len(X_new)} new points")
            try:
                _get_benchmarks_dir(self.config).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            self.save_checkpoint(i + 1, str(_get_benchmarks_dir(self.config) / f"{save_name}.pkl"))

        

        labels = [o["name"] for o in self.experiment.objectives_spec]

        Samples = np.asarray(self.input_repository)
        Objectives = np.asarray(self.output_repository)
        error_values = np.asarray(self.error_repository)
        C_values = np.asarray(self.constraint_values_repository)
        V_values = np.asarray(self.constraint_violations_repository)
        CV_values = np.asarray(self.constraint_repository).reshape(-1, 1)
        it = np.asarray(self.experiment.iteration_repository).reshape(-1, 1)
        penalty_raw_values = np.asarray(self.experiment.penalty_raw_repository, dtype=float)
        penalty_processed_values = np.asarray(self.experiment.penalty_value_repository, dtype=float)
        penalty_total_values = np.asarray(self.experiment.penalty_total_repository, dtype=float).reshape(-1, 1)

        raw_repo = np.hstack([
            it,
            Samples,
            Objectives,
            error_values,
            C_values,
            V_values,
            CV_values,
            penalty_raw_values,
            penalty_processed_values,
            penalty_total_values
        ])

        metrics_repo = []
        for j in range(len(HyperV)):
            metrics_repo.append({
                "Iteration": int(it[len(it) - len(HyperV) + j].reshape(-1)[0]),
                "Hypervolume": float(HyperV[j]),
                "Generational_Distance": float(GD[j]),
                "Diversity": float(Diversity[j]),
                "Spacing": float(Spacing[j]),
                "PF Count": int(Count[j]),
                "Runtime": float(runtime_records[j]),
                "Goal_func_kwargs": goal_kwargs_json,
            })

        return {"raw_repo": raw_repo,
                "gp_models": self.gp_models,
                "cv_gp_model": getattr(self, "cv_gp_model", None),
                "metrics_repo": metrics_repo}

class Acquisition(ABC):
    @abstractmethod
    def select_candidates(
        self,
        gp_models: Sequence[GaussianProcessRegressor],
        pareto_front: np.ndarray,
        input_bounds: Sequence[Tuple[float, float]],
        n_candidates: int = 1
    ) -> np.ndarray:
        """Return X_candidates"""
        pass

class BatchGreedyUCB(Acquisition):
    """Batch selection by greedy UCB (sequential penalisation), with optional constraint penalty."""

    def __init__(self, beta=2.5, batch_size=5, n_samples=2000, random_seed=0, length_scale=0.1):
        self.beta = float(beta)
        self.batch_size = int(batch_size)
        self.n_samples = int(n_samples)
        self.rng = np.random.default_rng(random_seed)
        self.length_scale = float(length_scale)

    @staticmethod
    def _diversity_penalty(acq_vals, X_candidates, chosen_points, length_scale):
        """Increase acquisition values near already chosen points (we are minimising)"""
        if not chosen_points:
            return acq_vals
        penalties = np.zeros_like(acq_vals, dtype=float)
        chosen = np.vstack(chosen_points)
        for x in chosen:
            dists = np.linalg.norm(X_candidates - x, axis=1)
            penalties += np.exp(-0.5 * (dists / length_scale) ** 2)
        return acq_vals + penalties * float(np.max(acq_vals))

    def select_candidates(self, gp_models, pareto_front, input_bounds, X_existing, n_candidates=None, **kwargs):
        """Return (X_best, info_dict)
        """
        if n_candidates is None:
            n_candidates = self.batch_size

        n_dim = len(input_bounds)
        sampler = Sobol(d=n_dim, scramble=True, seed=int(self.rng.integers(1_000_000)))
        X_candidates = sampler.random(self.n_samples)
        bounds = np.array(input_bounds, dtype=float)
        X_candidates = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * X_candidates

        # avoid re-sampling previous points
        X_candidates = filter_previously_sampled(X_candidates, X_existing, tol=1e-6)

        # Base UCB (minimisation): sum_j (mu_j - beta*sigma_j)
        means, stds = [], []
        for gp in gp_models:
            mu, sigma = gp.predict(X_candidates, return_std=True)
            means.append(mu)
            stds.append(sigma)
        means = np.vstack(means).T
        stds = np.vstack(stds).T
        # Optional objective weighting (applied to the acquisition aggregation only).
        weights = kwargs.get("weights", None)
        if weights is None:
            acq_raw = np.sum(means - self.beta * stds, axis=1)
        else:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if w.size != means.shape[1]:
                raise ValueError(f"weights length {w.size} must match number of objectives {means.shape[1]}.")
            acq_raw = np.sum((means - self.beta * stds) * w.reshape(1, -1), axis=1)

        # exponential constraint penalty, derived from CV models (consistent across evaluation methods)
        penalty_factor = np.ones_like(acq_raw, dtype=float)
        cv_gp = kwargs.get("cv_gp", None)
        alpha = float(kwargs.get("constraint_penalty_alpha", 0.0) or 0.0)
        feasible_tol = float(kwargs.get("feasible_tol", 0.0) or 0.0)
        if (cv_gp is not None) and (alpha > 0.0):
            cv_mu = np.asarray(cv_gp.predict(X_candidates), dtype=float).reshape(-1)
            penalty_factor = np.exp(alpha * np.maximum(0.0, cv_mu - feasible_tol))

        acq_penalised = acq_raw * penalty_factor

        # Greedy selection with diversity penalisation
        length_scale = float(kwargs.get("length_scale", None) or (0.1 * np.mean(bounds[:, 1] - bounds[:, 0])))
        chosen = []
        chosen_idx = []

        acq_working = acq_penalised.copy()
        for _ in range(int(n_candidates)):
            idx = int(np.argmin(acq_working))
            chosen.append(X_candidates[idx:idx+1, :])
            chosen_idx.append(idx)
            acq_working = self._diversity_penalty(acq_penalised, X_candidates, chosen, length_scale)

        X_best = X_candidates[np.array(chosen_idx), :]

        info = {
            "X_candidates": X_candidates,
            "acq_raw": acq_raw,
            "penalty_factor": penalty_factor,
            "acq_penalised": acq_penalised,
        }
        return X_best, info

class AcquisitionFactory:
    @staticmethod
    def create(name: str, **kwargs) -> Acquisition:
        name = name.upper()
        if name == "BATCH_UCB":
            allowed = {k: kwargs[k] for k in ["beta","batch_size","n_samples","random_seed","length_scale"] if k in kwargs}
            return BatchGreedyUCB(**allowed)
        else:
            raise ValueError(f"Unknown acquisition: {name}")
        


class EvaluatorBase:
    @abstractmethod
    def __init__(self, experiment):
        self.experiment = experiment
        self.config = experiment.config
        self.logger = experiment.logger
        self.gp_models = experiment.gp_models
        self.input_repository = experiment.input_repository
        self.output_repository = experiment.output_repository

class InteractiveEvaluator(EvaluatorBase):
    def __init__(self, inputs, objectives, constraints):
        self.inputs = inputs
        self.objectives = objectives
        self.constraints = constraints

    def _ask_float(self, prompt):
        while True:
            try:
                return float(input(prompt))
            except ValueError:
                print("Please enter a valid number.")

    def evaluate(self, X):
        print("\n=== NEW EVALUATION ===")
        print("Set machine to:")
        for inp, val in zip(self.inputs, X[0]):
            print(f"  {inp['name']} = {val:.6g}")

        input("\nPress ENTER once measurement is complete...")

        Y_obj = []
        Y_error = []

        print("\nEnter objective values:")
        for obj in self.objectives:
            v = self._ask_float(f"  {obj['name']} value: ")
            r = self._ask_float(f"  {obj['name']} error  : ")
            Y_obj.append(v)
            Y_error.append(r)

        Y_con = []
        if self.constraints:
            print("\nEnter constraint values:")
            for con in self.constraints:
                v = self._ask_float(f"  {con['name']} value: ")
                Y_con.append(v)

        Y_all = np.array(Y_obj + Y_con, dtype=float)
        return Y_obj, np.array(Y_error, dtype=float).reshape(1, -1), Y_con

    def evaluate_batch(self, X):
        Ys, Yerror, Yc = [], [], []
        for x in X:
            y, r, c = self.evaluate(x.reshape(1, -1))
            Ys.append(y)
            Yerror.append(r)
            Yc.append(c)
        return np.array(Ys, dtype=float), np.array(Yerror, dtype=float), np.array(Yc)


class GoalFunctionEvaluator(EvaluatorBase):
    """Evaluator that calls a user-provided goal function and evaluates batches in parallel."""

    def __init__(
        self,
        goal_fn,
        inputs,
        objectives_spec,
        constraints_spec,
        penalties_spec=None,
        default_objective_error: float = 1e-3,
        goal_function_path=None,
        goal_function_name="goal_function",
        goal_function_kwargs=None,
        multiprocess_bool=False
    ):
        self.goal_fn = goal_fn
        self.goal_function_path = goal_function_path
        self.goal_function_name = goal_function_name
        self.goal_function_kwargs = goal_function_kwargs
        self.inputs = inputs
        self.objectives_spec = objectives_spec
        self.constraints_spec = constraints_spec or []
        self.penalties_spec = penalties_spec or []
        self.default_objective_error = float(default_objective_error)
        self.multiprocess_bool=multiprocess_bool

    def _coerce_vec(self, v, expected_len, name):
        arr = np.asarray(v, dtype=float).reshape(-1)
        if arr.size != expected_len:
            raise ValueError(
                f"Goal function returned {name} length {arr.size}, expected {expected_len}."
            )
        return arr

    def _coerce_error(self, error, n_obj, n_con, n_pen):
        if error is None:
            return np.full(n_obj, self.default_objective_error, dtype=float)

        r = np.asarray(error, dtype=float).reshape(-1)

        if r.size == n_obj:
            return np.clip(r, 0.0, None)

        if (n_obj + n_con) and r.size == (n_obj + n_con):
            return np.clip(r[:n_obj], 0.0, None)
        
        if (n_obj + n_con + n_pen) and r.size == (n_obj + n_con + n_pen):
            return np.clip(r[:n_obj+n_con], 0.0, None)

        # if n_con and r.size == n_con:
        #     return np.full(n_obj, self.default_objective_error, dtype=float)

        raise ValueError(
            f"Goal function error length {r.size} does not match "
            f"n_obj={n_obj}, n_obj+n_con={n_obj+n_con}, or n_obj+n_con+n_pen={n_obj+n_con+n_pen}."
        )

    @staticmethod
    def _evaluate_single(args):
        (   x,
            goal_function_path,
            goal_function_name,
            goal_function_kwargs,
            objectives_spec,
            constraints_spec,
            penalty,
            default_objective_error,
        ) = args

        import importlib
        import importlib.util

        # Load goal function inside worker
        if str(goal_function_path).endswith(".py") and os.path.exists(str(goal_function_path)):
            spec = importlib.util.spec_from_file_location(
                "mobo_goal_module", str(goal_function_path)
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        else:
            mod = importlib.import_module(str(goal_function_path))

        goal_fn = getattr(mod, goal_function_name)

        n_obj = len(objectives_spec)
        n_con = len(constraints_spec)
        n_pen = len(penalty)

        def _coerce_vec(v, expected_len, name):
            arr = np.asarray(v, dtype=float).reshape(-1)
            if arr.size != expected_len:
                raise ValueError(
                    f"Goal function returned {name} length {arr.size}, expected {expected_len}."
                )
            return arr

        def _coerce_error(error):
            if error is None:
                return np.full(n_obj, default_objective_error, dtype=float)

            r = np.asarray(error, dtype=float).reshape(-1)

            if r.size == n_obj:
                return np.clip(r, 0.0, None)

            if (n_obj + n_con) and r.size == (n_obj + n_con):
                return np.clip(r[:n_obj], 0.0, None)
            
            if (n_obj + n_con + n_pen) and r.size == (n_obj + n_con + n_pen):
                return np.clip(r[:n_obj], 0.0, None)

            # if n_con and r.size == n_con:
            #     return np.full(n_obj, default_objective_error, dtype=float)

            raise ValueError(
                f"Goal function error length {r.size} does not match "
                f"n_obj={n_obj}, n_con={n_con}, n_obj+n_con={n_obj+n_con}, or n_obj+n_con+n_pen={n_obj+n_con+n_pen}."
            )


        res = goal_fn(x.reshape(1,-1), **(goal_function_kwargs or {}))

        pens = None
        objs = None
        cons = None
        error = None
        raw = None

        if isinstance(res, dict):
            raw = res.get("raw", None)
            objs = res.get("objectives", res.get("objs", None))
            cons = res.get("constraints", res.get("cons", None))
            error = res.get("errors", None)
            pens = res.get("penalties", None)

        if raw is not None:
            raw_arr = np.asarray(raw, dtype=float).reshape(-1)
            if raw_arr.size != n_obj:
                raise ValueError(
                    f"Goal function returned raw length {raw_arr.size}, expected {n_obj}."
                )
        else:
            raw_arr = _coerce_vec(objs, n_obj, "objectives")

        cons_arr = (
            _coerce_vec(cons if cons is not None else np.zeros(n_con), n_con, "constraints")
            if n_con
            else np.asarray([], dtype=float)
        )

        pen_arr = (
            _coerce_vec(pens if pens is not None else np.zeros(n_pen), n_pen, "penalties")
            if n_pen
            else np.asarray([], dtype=float)
        )

        error_arr = _coerce_error(error)
        return raw_arr, error_arr, cons_arr, pen_arr

    def evaluate_batch(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        X = np.atleast_2d(X)

        n_batch = X.shape[0]

        args = [
            (
                X[i],
                self.goal_function_path,
                self.goal_function_name,
                self.goal_function_kwargs,
                self.objectives_spec,
                self.constraints_spec,
                self.penalties_spec,
                self.default_objective_error,
            )
            for i in range(n_batch)
        ]

        if self.multiprocess_bool==True:
            with Pool(processes=os.cpu_count()) as pool:
                results = pool.map(self._evaluate_single, args)
        if self.multiprocess_bool==False:
            results=[]
            for i in range(len(X)):
                results.append(self._evaluate_single(args[i]))

        raw_rows, error_rows, con_rows, pen_rows = zip(*results)

        raw_new = np.vstack(raw_rows).astype(float)
        error_new = np.vstack(error_rows).astype(float)
        con_new = np.vstack(con_rows).astype(float) if len(self.constraints_spec) else np.empty((n_batch, 0))
        pen_new = np.vstack(pen_rows).astype(float) if len(self.penalties_spec) else np.empty((n_batch, 0))

        return raw_new, error_new, con_new, pen_new

    def evaluate(self, X: np.ndarray):
        raw, error, con, pen = self.evaluate_batch(X)
        return raw, error, con, pen
    


class EvaluatorFactory:
    @staticmethod
    def create(method: str, **kwargs) -> EvaluatorBase:
        method = method.upper()
        if method == "MANUAL":
            inputs = kwargs["inputs"]
            objectives = kwargs["objectives"]
            constraints = kwargs.get("constraints", [])
            return InteractiveEvaluator(inputs, objectives, constraints)

        elif method in ("GOAL", "GOAL_FUNCTION", "FUNC", "FUNCTION"):
            # Load goal function from a module path or .py file path
            goal_path = kwargs.get("goal_function_path", "")
            goal_name = kwargs.get("goal_function_name", "goal_function")
            default_error = kwargs.get("default_objective_error", 1e-3)
            multiprocess = kwargs.get("multiprocess_bool", False)
            penalties_spec = kwargs.get("penalties", [])
            goal_function_kwargs = kwargs.get("goal_function_kwargs", None)

            if not goal_path:
                raise ValueError("GOAL evaluation requires goal_function_path in config")

            ################## Find goal function #################

            import importlib
            import importlib.util

            if str(goal_path).endswith(".py") and os.path.exists(str(goal_path)):
                spec = importlib.util.spec_from_file_location("mobo_goal_module", str(goal_path))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
            else:
                mod = importlib.import_module(str(goal_path))

            if not hasattr(mod, goal_name):
                raise ValueError(f"Goal module '{goal_path}' has no function '{goal_name}'")

            goal_fn = getattr(mod, goal_name)


            return GoalFunctionEvaluator(goal_fn=goal_fn,
                inputs=kwargs["inputs"],
                objectives_spec=kwargs.get("objectives_spec") or kwargs.get("objectives"),
                constraints_spec=kwargs.get("constraints_spec") or kwargs.get("constraints", []),
                default_objective_error=default_error,
                goal_function_path=goal_path,
                goal_function_name=goal_name,
                goal_function_kwargs= kwargs.get("goal_function_kwargs"),
                multiprocess_bool=multiprocess,
                penalties_spec=penalties_spec)
        else:
            raise ValueError(f"Unknown evaluation method: {method}")

class MOBOPlotter:
    def __init__(self):
        pass

    def _save_fig(self, filename: str, dpi=300):
        path = filename
        plt.tight_layout()
        plt.savefig(path, dpi=dpi)
        plt.close()
        return path

    def plot_acquisition(self, iteration: int, save_prefix: str, info: dict):
        """Plot acquisition values for the candidate pool (raw vs penalised)."""
        if not info:
            return
        Xc = info.get("X_candidates", None)
        acq_raw = info.get("acq_raw", None)
        acq_pen = info.get("acq_penalised", None)
        if Xc is None or acq_raw is None:
            return

        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(len(acq_raw)), acq_raw, label="acq_raw")
        if acq_pen is not None:
            plt.plot(np.arange(len(acq_pen)), acq_pen, label="acq_penalised")
        plt.xlabel("candidate index")
        plt.ylabel("acquisition")
        plt.title(f"Acquisition at iteration {iteration}")
        plt.legend()

        outdir = (getattr(self, "base_output_dir", None) or Path("Outputs")) / str(save_prefix)
        outdir.mkdir(parents=True, exist_ok=True)
        self._save_fig(str(outdir / f"acq_iter_{iteration:04d}.png"))

    def plot_pareto_front(self, Pareto_front, Reference_point, Iteration,
                        Best_pareto_front, Init_pareto_front, O_train, O_tot,
                        Output_indices, Labels, save_prefix):
        """Plot Pareto front with dominated hypervolume shading.
        For 2D only!"""
        pareto_sorted = Pareto_front[np.argsort(Pareto_front[:, 0])]
        initial_sorted = Init_pareto_front[np.argsort(Init_pareto_front[:, 0])]
        plt.figure(figsize=(8, 6))

        # Step plot
        for i in range(1, len(pareto_sorted)):
            plt.plot([pareto_sorted[i-1, 0], pareto_sorted[i, 0]],
                    [pareto_sorted[i-1, 1], pareto_sorted[i-1, 1]], 'orange')
            plt.plot([pareto_sorted[i, 0], pareto_sorted[i, 0]],
                    [pareto_sorted[i-1, 1], pareto_sorted[i, 1]], 'orange')

        # Dominated HV shading
        hv_x, hv_y = [Reference_point[0]], [Reference_point[1]]
        for x, y in pareto_sorted:
            hv_x.extend([x, x])
            hv_y.extend([hv_y[-1], y])
        hv_x.append(Reference_point[0])
        hv_y.append(hv_y[-1])
        plt.fill(hv_x, hv_y, color='b', alpha=0.3, label='Dominated HV')

        plt.plot(initial_sorted[:, 0], initial_sorted[:, 1], 'x', label='Initial PF', color='b')
        plt.plot(O_train[:, 0], O_train[:, 1], '+', c='k', label='All Samples')
        plt.plot(Best_pareto_front[:, 0], Best_pareto_front[:, 1], 'x', c='r', label='Best PF')

        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.title(f'Dominated HV at Iteration {Iteration}')
        plt.legend()
        filename = f"{save_prefix}_HV_iter_{Iteration}.png"
        return self._save_fig(filename)

    def plot_3obj_pareto_physical_axes(self, Y, pareto_idx, objective_names, title="3D Pareto Front", cmap="viridis"):
        """If D!=3, does not plot. """
        if Y.shape[1]!=3:
            return
        first_obj_label = objective_names[0]
        second_obj_label = objective_names[1]
        third_obj_label = objective_names[2]
        # ensure Y is a real 2D float array
        Y = np.asarray(Y)
    
        # If a single point shape (3,), make it (1, 3)
        if Y.ndim == 1:
            Y = np.atleast_2d(Y)
    
        # If it's a scalar/unsized, error
        if Y.ndim == 0:
            raise ValueError(f"plot_3obj_pareto_physical_axes received a scalar: {Y!r}")
    
        # If object dtype, force numeric conversion
        if Y.dtype == object:
            Y = np.asarray(Y, dtype=float)
    
        mask = np.ones(Y.shape[0], dtype=bool)
        mask[pareto_idx] = False
    
        plt.figure(figsize=(6, 5))
    
        # Background: dominated points
        sc_bg = plt.scatter(
            Y[mask, 0],
            Y[mask, 1],
            c=-Y[mask, 2],
            cmap=cmap,
            alpha=0.2,
            s=25,
            linewidths=0,
        )
    
        # Foreground: 3D Pareto front
        sc_pf = plt.scatter(
            Y[pareto_idx, 0],
            Y[pareto_idx, 1],
            c=-Y[pareto_idx, 2],
            cmap=cmap,
            edgecolors="k",
            s=60,
            label="3D Pareto front",
        )
    
        cbar = plt.colorbar(sc_pf)
        cbar.set_label(third_obj_label)
    
        plt.xlabel(first_obj_label)
        plt.ylabel(second_obj_label)
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(Path(os.getcwd()) / f"{title}.png"))
        # plt.show()

    def plot_pareto_front_colourmap(self, Y_init, Y, pf_idx, objective_labels, objective_directions, save_name, CV=None, ax=None, cmap="viridis"):
        """
        Plot ALL sampled points in PHYSICAL space. This is for better operator interpretability
          - feasible: circles coloured by objective 3 (physical)
          - infeasible: x markers (gray)
          - Pareto front (feasible PF): highlighted with black edge
        For 2 or 3 dims - if 3 dims, 3rd objective is colour. If 2 dims, colour is just second objective again. Otherwise will only see 3 dimensions of whatever inputted.
        """
    
        Y = np.asarray(Y, dtype=float)
        if CV is None:
            CV = np.zeros(len(Y), dtype=float)
        CV = np.asarray(CV, dtype=float)
    
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
    
        # Convert to PHYSICAL values for plotting only (undo sign flip for "max")
        Y_plot = Y.copy()
        Y_init_plot=Y_init.copy()
        for i, d in enumerate(objective_directions):
            if d == "max":
                Y_plot[:, i] = -Y_plot[:, i]
                Y_init_plot[:,i]=-Y_init_plot[:,i]
    
        feasible = CV <= 1e-12
        infeasible = ~feasible
        init_feasible=feasible[:len(Y_init)]

        n_obj = Y_plot.shape[1]
        if n_obj < 2:
            # nothing meaningful to scatter in 2D; just return
            return
        color_dim = 2 if n_obj >= 3 else 1  # for 2 objectives color by objective 2
        c = Y_plot[feasible, color_dim]

        

        # all feasible points
        sc = ax.scatter(
            Y_plot[feasible, 0], Y_plot[feasible, 1],
            c=c,
            cmap=cmap,
            s=45,
            alpha=0.85,
            edgecolor="none",
            label="Feasible samples",
        )
    
        # infeasible points
        if np.any(infeasible):
            ax.scatter(
                Y_plot[infeasible, 0], Y_plot[infeasible, 1],
                marker="x",
                s=60,
                linewidths=1.5,
                c="gray",
                alpha=0.9,
                label="Infeasible samples",
            )

        c = Y_plot[pf_idx, color_dim]
        # Pareto front points
        pf_idx = np.asarray(pf_idx, dtype=int)
        ax.scatter(
            Y_plot[pf_idx, 0], Y_plot[pf_idx, 1],
            c=c,
            cmap=cmap,
            s=80,
            edgecolor="k",
            linewidths=1.0,
            label="Pareto front",
        )

        # plot initial samples
        sc = ax.scatter(
            Y_init_plot[init_feasible, 0], Y_init_plot[init_feasible, 1],
            color='r',
            marker='x',
            s=45,
            alpha=0.85,
            edgecolor="none",
            label="Initial samples",
        )
    
        ax.set_xlabel('Objective 1')
        ax.set_ylabel('Objective 2')
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Objective 3')
    
        ax.set_title("Pareto front")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
    
        plt.savefig(
            str(Path(os.getcwd()) / "Outputs" / f"_{save_name}_current_sampled_points.png")
        )
        return ax


    def plot_gp_models_over_discrete_grid(
    self,
    gp_models,
    input_specs,   
    X_train,
    Y_train,
    error_train=None,
    CV_train=None,
    labels=None,
    iteration=None,
    input_scaler=None,
    output_scaler=None,
    order="input0_slowest",
    two_sigma=2.0,
    feasible_tol=0.0,
    default_objective_error=1e-3,
    next_index=None,
    save_path=None,
    show=False,
    max_candidates=None):
        print('GP plotting')
        print(error_train)
        if save_path==None:
            save_path=str(_get_outputs_dir(getattr(self, "config", type("C", (), {})())) / f"gp_models_over_inputs_iter_{iteration}.png")
            save=str(_get_outputs_dir(getattr(self, "config", type("C", (), {})())) / "gp_models_over_inputs.png")
    
        # Build ordered discrete candidate list from input_specs
        X_grid, _ = build_discrete_grid(input_specs, order=order)
        if max_candidates is not None and X_grid.shape[0] > max_candidates:
            X_grid = X_grid[:max_candidates]

        N = X_grid.shape[0]
        x_idx = np.arange(N)

        # Predict across this grid
        Xg = input_scaler.transform(X_grid) if input_scaler is not None else X_grid

        n_obj = len(gp_models)
        mu = np.zeros((n_obj, N))
        std = np.zeros((n_obj, N))
        for j, gp in enumerate(gp_models):
            m, s = gp.predict(Xg, return_std=True)
            if output_scaler is not None:
                m = m * output_scaler.scale_[j] + output_scaler.mean_[j]
                s = s * output_scaler.scale_[j]
            mu[j] = m
            std[j] = s

        #Map training points to grid indices
        train_idx = map_points_to_grid_index(X_train, X_grid)

        # error and scaling consistency
        Y_train = np.asarray(Y_train, float)
        if error_train is None:
            error_train = np.full_like(Y_train, float(default_objective_error))
        else:
            error_train = np.asarray(error_train, float)

        Y_plot = Y_train.copy()
        error_plot = error_train.copy()
        if output_scaler is not None:
            Y_plot = Y_plot * output_scaler.scale_ + output_scaler.mean_
            error_plot = error_plot * output_scaler.scale_

        #Feasibility mask
        if CV_train is None:
            feasible = np.ones(len(train_idx), dtype=bool)
        else:
            CV = np.asarray(CV_train, float)
            feasible = (CV <= feasible_tol) if CV.ndim == 1 else np.all(CV <= feasible_tol, axis=1)
        infeasible = ~feasible

        #Plot
        fig, ax = plt.subplots(figsize=(16, 9))
        title = "Gaussian Process Models"
        if iteration is not None:
            title = f"Iteration {iteration}\n" + title
        ax.set_title(title)
        ax.set_xlabel("Ordered discrete candidates (index)")
        ax.set_ylabel("Model")

        colors = ['r', 'b', 'g', 'm', 'c', 'y']

        for j in range(n_obj):
            c = colors[j % len(colors)]
            name = labels[j] if labels and j < len(labels) else f"Obj {j+1}"

            ax.plot(x_idx[::1000], mu[j][::1000], c, label=f"GP model {name}")
            ax.fill_between(x_idx[::1000], mu[j][::1000] - two_sigma * std[j][::1000], mu[j][::1000] + two_sigma * std[j][::1000], color=c, alpha=0.15)

            y = Y_plot[:, j]
            error = error_plot[:, j]
            error = np.clip(np.asarray(error, dtype=float), 0.0, None)

            if np.any(infeasible):
                ax.errorbar(
                    train_idx[infeasible], y[infeasible], yerr=error[infeasible],
                    fmt='o', markersize=4, capsize=2, elinewidth=1,
                    color='0.6', ecolor='0.6', alpha=0.8,
                    label="measured infeasible ±error" if j == 0 else None
                )
            if np.any(feasible):
                ax.errorbar(
                    train_idx[feasible], y[feasible], yerr=error[feasible],
                    fmt='o', markersize=4, capsize=2, elinewidth=1,
                    color=c, ecolor=c, alpha=0.9,
                    label="measured feasible ±error" if j == 0 else None
                )

        if next_index is not None:
            ax.axvline(int(next_index), color='k', linestyle=':', alpha=0.9, label="Next point")

        ax.legend(loc="upper left")
        fig.tight_layout()

        if save_path is not None:
            out_dir = os.path.dirname(save_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            # fig.savefig(save, dpi=150, bbox_inches="tight")
            print(f"[plot_gp_models_over_discrete_grid] saved -> {save_path}")

        # if show:
        #     plt.show()
        print(f'Saving gp models to {save_path}')
        plt.savefig(save_path, dpi=200)

        return X_grid


    def plot_everything_with_error(self,
        Iteration,
        gp_models,
        X_candidates,              
        input_scaler=None,        
        output_scaler=None,       
        Labels=None,
        objective_names=None,      
        CV_train=None,
        feasible_tol=0.0,
        beta=2.0,
        Weighting='F',
        Weights=None,
        UCB_values=None,         
        top_indices=None,        
        Penalise='F',
        UCB_penalised=None,
        X_train=None,                
        Y_train=None,                
        error_train=None,             
        default_objective_error=1e-3,
        include_meas_noise_in_band=True,  
        save_path=None,):
            """
            Plot GP predictions over a candidate list, plus UCB on twin axis. Includes measurement error.
            """
            # print("first train row:", self.experiment.input_repository[0])
            if save_path==None:
                save_path=str(_get_outputs_dir(getattr(self, "config", type("C", (), {})())) / f"gp_models_iter_{Iteration}.png")
                save=str(_get_outputs_dir(getattr(self, "config", type("C", (), {})())) / "gp_models.png")
            n_obj = len(gp_models)
            N = X_candidates.shape[0]

            if Weights is None:
                Weights = [1.0] * n_obj
            Weights = np.asarray(Weights, float)

            # beta handling
            if np.isscalar(beta):
                beta_vec = np.full(n_obj, float(beta))
            else:
                beta_vec = np.asarray(beta, float).reshape(-1)
                if beta_vec.size != n_obj:
                    raise ValueError(f"beta must be scalar or length {n_obj}")

            if Weighting == 'T':
                beta_vec = beta_vec * Weights

            # Prepare candidate inputs in the same space GP expects
            Xcand_gp = input_scaler.transform(X_candidates) if input_scaler is not None else X_candidates

            # Predict for each objective
            mu_list, std_list = [], []
            for j, gp in enumerate(gp_models):
                mu, std = gp.predict(Xcand_gp, return_std=True)
                # If output was scaled during training, invert to physical
                if output_scaler is not None:
                    mu = mu * output_scaler.scale_[j] + output_scaler.mean_[j]
                    std = std * output_scaler.scale_[j]
                mu_list.append(mu)
                std_list.append(std)

            mu = np.vstack(mu_list)
            sigma = np.vstack(std_list)

            # Measurement noise level to add to the band
            # We can use median error per objective from training data if provided
            meas_sigma = np.full(n_obj, float(default_objective_error))
            if error_train is not None:
                error_train = np.asarray(error_train, float)
                if output_scaler is not None:
                    # if error was stored in scaled space, convert to physical
                    meas_sigma = np.median(error_train, axis=0) * output_scaler.scale_[:n_obj]
                else:
                    meas_sigma = np.median(error_train, axis=0)

            # Plot
            fig, ax1 = plt.subplots(figsize=(16, 12))
            fig.suptitle(f"Iteration {Iteration}")

            x_idx = np.arange(N)
            colours = ['r', 'b', 'g', 'm', 'c', 'y']

            for j in range(n_obj):
                c = colours[j % len(colours)]
                name = None
                if objective_names is not None and j < len(objective_names):
                    name = objective_names[j]
                elif Labels is not None:
                    name = str(Labels[j])
                else:
                    name = f"Objective {j+1}"

                y_mean = mu[j] * Weights[j]

                # band std: GP std + measurement error
                if include_meas_noise_in_band:
                    std_plot = np.sqrt(sigma[j]**2 + meas_sigma[j]**2)
                else:
                    std_plot = sigma[j]

                y_lo = y_mean - beta_vec[j] * Weights[j] * std_plot
                y_hi = y_mean + beta_vec[j] * Weights[j] * std_plot

                ax1.plot(x_idx, y_mean, c, label=f"GP model {name}")
                ax1.fill_between(x_idx, y_lo, y_hi, alpha=0.2, color=c)

            ax1.set_title("Gaussian Process Models")
            ax1.set_xlabel("Input selections (candidate index)")
            ax1.set_ylabel("Models")

            # measured points on the SAME x-axis
            if X_train is not None and Y_train is not None:
                pool_idx = map_train_to_pool_indices(X_train, X_candidates)

                # feasibility mask from CV_train
                if CV_train is None:
                    feasible = np.ones(len(pool_idx), dtype=bool)
                else:
                    CV = np.asarray(CV_train, float)
                    if CV.ndim == 1:
                        feasible = CV <= feasible_tol
                    else:
                        feasible = np.all(CV <= feasible_tol, axis=1)
                infeasible = ~feasible

                for j in range(n_obj):
                    # physical units for scatter + error
                    if output_scaler is not None:
                        y_phys = Y_train[:, j] * output_scaler.scale_[j] + output_scaler.mean_[j]
                        if error_train is not None:
                            error_phys = np.asarray(error_train)[:, j] * output_scaler.scale_[j]
                        else:
                            error_phys = np.full_like(y_phys, default_objective_error, dtype=float)
                    else:
                        y_phys = Y_train[:, j]
                        error_phys = np.asarray(error_train)[:, j] if error_train is not None else np.full_like(y_phys, default_objective_error)

                    c = colours[j % len(colours)]

                    # infeasible points in grey
                    if np.any(infeasible):
                        ax1.errorbar(
                            pool_idx[infeasible],
                            (y_phys * Weights[j])[infeasible],
                            yerr=(error_phys * Weights[j])[infeasible],
                            fmt="o",
                            color="0.55",
                            ecolor="0.55",
                            capsize=3,
                            markersize=5,
                            alpha=0.85,
                            label="measured infeasible ±error" if j == 0 else None,
                        )

                    # feasible points in objective colour
                    if np.any(feasible):
                        ax1.errorbar(
                            pool_idx[feasible],
                            (y_phys * Weights[j])[feasible],
                            yerr=(error_phys * Weights[j])[feasible],
                            fmt="o",
                            color=c,
                            ecolor=c,
                            capsize=3,
                            markersize=5,
                            alpha=0.9,
                            label="measured feasible ±error" if j == 0 else None,
                        )

            ax1.legend(loc="upper left")
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150)
                plt.savefig(save, dpi=150)
                plt.close(fig)
            # else:
            #     plt.show()

            return

    def plot_pareto_front_colored(self, X, Y, pareto_idx, iteration, i, save_name, error_Y=None):
        """
        Plot evaluated points colored by third objective, and highlight Pareto front
        """
        # X=X.to_numpy(dtype=float)
        # Y=Y.to_numpy(dtype=float)
        if Y.shape[1]!=3:
            return
        if error_Y is not None:
            error_Y = np.asarray(error_Y, dtype=float)
        # Create a color array based on evaluation order
        order_colors = np.arange(len(Y))

        plt.figure(figsize=(8,6))

        if error_Y is not None and error_Y.shape == Y.shape:
            plt.errorbar(
                Y[:, 0], Y[:, 1],
                xerr=error_Y[:, 0],
                yerr=error_Y[:, 1],
                fmt="none",
                ecolor="0.6",
                elinewidth=0.8,
                alpha=0.35,
                capsize=2,
                zorder=1,
            )

        sc = plt.scatter(
            Y[:, 0], Y[:, 1],
            c=Y[:, 2],
            cmap="viridis",
            s=20,
            edgecolor="None",
            alpha=0.8,
            label="Sampled Points",
            zorder=2,
        )

        if error_Y is not None and error_Y.shape == Y.shape and len(pareto_idx) > 0:
            plt.errorbar(
                Y[pareto_idx, 0], Y[pareto_idx, 1],
                xerr=error_Y[pareto_idx, 0],
                yerr=error_Y[pareto_idx, 1],
                fmt="none",
                ecolor="k",
                elinewidth=1.0,
                alpha=0.6,
                capsize=2,
                zorder=3,
            )

        plt.scatter(
            Y[pareto_idx, 0], Y[pareto_idx, 1],
            c="None",
            s=20,
            edgecolor="k",
            label="Pareto Front",
            zorder=4,
        )

        plt.xlabel("Objective 1")
        plt.ylabel("Objective 2")
        plt.title(f"Pareto Front and Sample Order (Iteration {i} out of {iteration})")
        plt.colorbar(sc, label="Objective 3 (2 if in 2dim)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(str(_get_outputs_dir(getattr(self, "config", type("C", (), {})())) / f"_{save_name}_final_sampled_points.png"))



    def plot_gp_slices_all_dims(
    self,
    gp_models,
    input_bounds,
    X_train,
    Y_train,
    error_train=None,
    CV_train=None,
    labels=None,
    iteration=None,
    input_scaler=None,
    output_scaler=None,
    x0=None,
    anchor_x0_to_data=True,
    n_grid=300,
    two_sigma=2.0,
    slice_tol_frac=0.10,
    default_objective_error=1e-3,
    feasible_tol=0.0,
    save_path=None,
            ):
        """
        Grid of 1D GP slices for each input dimension k.
        - All training points are shown in every slice (projected onto x_k).
        - Infeasible points are greyed out.
        - Points near the slice (in other dims) get error error bars.
        """
        import math
        print('Plotting sliced GPs')
        print(error_train)
        if save_path==None:
            save_path=str(_get_outputs_dir(getattr(self, "config", type("C", (), {})())) / f"gp_models_sliced_iter_{iteration}.png")
            save=str(_get_outputs_dir(getattr(self, "config", type("C", (), {})())) / "gp_models_sliced.png")
        X_train = np.asarray(X_train, float)
        Y_train = np.asarray(Y_train, float)
        n_train, d = X_train.shape
        n_obj = len(gp_models)

        # feasibility mask from CV train
        if CV_train is None:
            feasible = np.ones(n_train, dtype=bool)
        else:
            CV = np.asarray(CV_train, float)
            if CV.ndim == 1:
                feasible = CV <= feasible_tol
            else:
                feasible = np.all(CV <= feasible_tol, axis=1)

        infeasible = ~feasible

        # error
        if error_train is None:
            error_train = np.full((n_train, n_obj), float(default_objective_error), dtype=float)
        else:
            error_train = np.asarray(error_train, float)
            if error_train.shape != (n_train, n_obj):
                raise ValueError(f"error_train must be shape {(n_train, n_obj)}, got {error_train.shape}")

        # choose x0
        if x0 is None:
            if anchor_x0_to_data:
                x_med = np.median(X_train, axis=0)
                idx0 = int(np.argmin(np.sum((X_train - x_med) ** 2, axis=1)))
                x0 = X_train[idx0].copy()
            else:
                x0 = np.array([(a + b) / 2 for a, b in input_bounds], dtype=float)
        else:
            x0 = np.asarray(x0, float).reshape(-1)
            if x0.shape[0] != d:
                raise ValueError(f"x0 must have length {d}, got {x0.shape[0]}")

        # subplot layout
        ncols = 3 if d >= 3 else d
        nrows = math.ceil(d / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.7 * ncols, 4.0 * nrows), squeeze=False)

        # for "near slice" selection
        widths = np.ptp(X_train, axis=0)
        widths[widths == 0] = 1.0
        tol = slice_tol_frac * widths

        for k in range(d):
            ax = axes[k // ncols][k % ncols]

            # slice range
            xmin, xmax = float(np.min(X_train[:, k])), float(np.max(X_train[:, k]))
            pad = 0.05 * (xmax - xmin if xmax > xmin else 1.0)
            xmin, xmax = xmin - pad, xmax + pad
            t = np.linspace(xmin, xmax, n_grid)

            Xg_raw = np.tile(x0, (n_grid, 1))
            Xg_raw[:, k] = t
            Xg = input_scaler.transform(Xg_raw) if input_scaler is not None else Xg_raw

            # near-slice mask
            other = [j for j in range(d) if j != k]
            if len(other) == 0:
                near = np.ones(n_train, dtype=bool)
            else:
                near = np.all(np.abs(X_train[:, other] - x0[other]) <= tol[other], axis=1)

            # plot ALL training points (projected onto x_k)
            # Show feasible in colour-ish, infeasible in grey
            for j in range(n_obj):
                if output_scaler is not None:
                    y = Y_train[:, j] * output_scaler.scale_[j] + output_scaler.mean_[j]
                    error = error_train[:, j] * output_scaler.scale_[j]
                else:
                    y = Y_train[:, j]
                    error = error_train[:, j]

                # infeasible (greyed out)
                if np.any(infeasible):
                    print('Plotting infeasible points')
                    ax.scatter(
                        X_train[infeasible, k], y[infeasible],
                        s=12, alpha=0.35,
                        color="0.6",  # grey
                        label="infeasible (all)" if (k == 0 and j == 0) else None,
                    )

                # feasible (normal)
                if np.any(feasible):
                    print('Plotting feasible points')
                    ax.scatter(
                        X_train[feasible, k], y[feasible],
                        s=14, alpha=0.45,
                        label="feasible (all)" if (k == 0 and j == 0) else None,
                    )

                # error bars for points near slice
                if np.any(near):
                    print('Points near slice')
                    # near & infeasible
                    m = near & infeasible
                    if np.any(m):
                        ax.errorbar(
                            X_train[m, k], y[m], yerr=error[m],
                            fmt="o", markersize=4.5, capsize=2.5, elinewidth=1.0,
                            color="0.45", ecolor="0.45", alpha=0.9,
                            label="near slice ±error (infeas)" if (k == 0 and j == 0) else None,
                        )
                    # near & feasible
                    m = near & feasible
                    if np.any(m):
                        ax.errorbar(
                            X_train[m, k], y[m], yerr=error[m],
                            fmt="o", markersize=4.5, capsize=2.5, elinewidth=1.0,
                            alpha=0.95,
                            label="near slice ±error (feas)" if (k == 0 and j == 0) else None,
                        )

            # GP mean/band for each objective
            for j, gp in enumerate(gp_models):
                mu, std = gp.predict(Xg, return_std=True)

                if output_scaler is not None:
                    mu = mu * output_scaler.scale_[j] + output_scaler.mean_[j]
                    std = std * output_scaler.scale_[j]

                name = labels[j] if labels and j < len(labels) else f"Obj {j+1}"
                ax.plot(t, mu, label=f"{name} mean" if k == 0 else None)
                ax.fill_between(t, mu - two_sigma * std, mu + two_sigma * std, alpha=0.18)

            ax.set_title(f"Slice k={k} (near={int(np.sum(near))})")
            ax.set_xlabel(f"Input {k}")
            ax.set_ylabel("Objective")

        # # turn off unused axes
        for kk in range(d, nrows * ncols):
            axes[kk // ncols][kk % ncols].axis("off")

        title = "GP slices (all inputs)"
        if iteration is not None:
            title += f" – iter {iteration}"
        fig.suptitle(title, y=1.02, fontsize=14)

        # single legend
        handles, leglabels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, leglabels, loc="upper right")

        fig.tight_layout()

        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        fig.savefig(save, dpi=150, bbox_inches="tight")

        plt.close(fig)

    def plot_hypervolume_evolution(self, HV, eval_method, save_prefix, i, save_name):
        plt.figure(figsize=(8, 6))
        HV = np.array(HV).flatten()
        plt.plot(range(len(HV)), HV, '-o', markersize=3, label="Hypervolume")
        plt.xlabel("Iteration")
        plt.ylabel("Hypervolume")
        plt.title("Hypervolume Evolution")
        plt.legend()
        # plt.ylim(bottom=0)
        filename = str(Path(os.getcwd()) / f"{save_prefix}_{save_name}_{eval_method}_HV_evolution.png")
        plt.savefig(filename)
        return 


    def plot_runtime(self, runtime, save_prefix):
        plt.figure(figsize=(8,6))
        iterations=range(len(runtime))
        plt.plot(iterations, runtime, '-o')
        plt.xlabel("Iteration")
        plt.ylabel("Runtime (s)")
        plt.title("Runtime per Iteration")
        plt.grid(True)
        filename = str(Path(os.getcwd()) / f"{save_prefix}_runtime.png")
        plt.savefig(filename, dpi=300)
        plt.close()


    def plot_pareto_comparison(self, final_pf, best_pf,
                               init_pf, output_indices, Labels,
                               eval_method, save_prefix, save_name):
        """Compare final PF vs initial PF."""

        final_pf = np.atleast_2d(final_pf)
        # best_pf = np.atleast_2d(best_pf)
        init_pf = np.atleast_2d(init_pf)

        plt.figure(figsize=(8, 6))
        plt.scatter(init_pf[:, 0], init_pf[:, 1], label="Initial PF", color="orange")
        plt.scatter(final_pf[:, 0], final_pf[:, 1], marker='x', label="Final PF", color="blue")
        # plt.scatter(best_pf[:, 0], best_pf[:, 1], label="Best PF", color="red")
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.title("PF Comparison")
        plt.legend()
        filename = str(Path(os.getcwd()) /f"{save_prefix}_{save_name}_{eval_method}_PF_comparison.png")
        return self._save_fig(filename)

    def plot_sample_evolution(self, O_train, Trunc_pf, no_of_meas, Iteration,
                              Labels, Output_indices, eval_method, save_prefix):
        """Plot sample objective values over iterations."""
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(O_train[no_of_meas:, 0])), O_train[no_of_meas:, 0], 'x', color='r',
                 label='Objective 1')
        plt.plot(range(len(O_train[no_of_meas:, 1])), O_train[no_of_meas:, 1], 'x', color='b',
                 label='Objective 2')
        plt.xlabel('Iteration')
        # plt.ylim(0, 10)
        plt.grid()
        plt.legend()
        filename = str(Path(os.getcwd()) /f"{save_prefix}_{eval_method}_samples_iter_{Iteration}.png")
        return self._save_fig(filename)

    def final_plots(self, O_train, O_tot, Output_indices, no_of_meas,
                    Labels, save_prefix, eval_method, HV, Pareto_front,
                    Init_pareto_front, Best_pareto_front, Reference_point):
        """Convenience method: HV evolution + PF plot."""
        self.plot_hypervolume_evolution(HV, eval_method, save_prefix)
        self.plot_pareto_front(Pareto_front, Reference_point, 'Final',
                               Best_pareto_front, Init_pareto_front,
                               O_train, O_tot, Output_indices, Labels, save_prefix)


    def plot_metrics_evolution(self, hypervolume: list, spacing: list, generational_distance: list, 
        diversity: list, num_pf_points: list,runtime: list, save_prefix: str):
        """
        Plot the evolution of key metrics over iterations.
        """
        iterations = np.arange(1, len(hypervolume) + 1)

        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        axes = axes.flatten()

        # 1. Hypervolume
        axes[0].plot(iterations, hypervolume, marker='o', color='tab:blue')
        axes[0].set_title("Hypervolume")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("HV")
        axes[0].grid(True)

        # 2. Spacing
        axes[1].plot(iterations, spacing, marker='o', color='tab:orange')
        axes[1].set_title("Spacing")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Spacing")
        axes[1].grid(True)

        # 3. Generational Distance
        axes[2].plot(iterations, generational_distance, marker='o', color='tab:green')
        axes[2].set_title("Generational Distance")
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("GD")
        axes[2].grid(True)

        # 4. Diversity
        axes[3].plot(iterations, diversity, marker='o', color='tab:red')
        axes[3].set_title("Diversity")
        axes[3].set_xlabel("Iteration")
        axes[3].set_ylabel("Diversity")
        axes[3].grid(True)

        # 5. Number of Pareto Points
        axes[4].plot(iterations, num_pf_points, marker='o', color='tab:purple')
        axes[4].set_title("Number of Pareto Points")
        axes[4].set_xlabel("Iteration")
        axes[4].set_ylabel("Num Points")
        axes[4].grid(True)

        # 6. Runtime
        axes[5].plot(iterations, runtime, marker='o', color='tab:brown')
        axes[5].set_title("Runtime per Iteration")
        axes[5].set_xlabel("Iteration")
        axes[5].set_ylabel("Time (s)")
        axes[5].grid(True)

        plt.tight_layout()
        filename = str(Path(os.getcwd()) /f"{save_prefix}_metrics_evolution.png")
        plt.savefig(filename, dpi=300)
        plt.close()
        return filename

class Metrics:
    """Centralised metrics for multi-objective optimization with scaling."""

    @staticmethod
    def _scale(pf: np.ndarray) -> np.ndarray:
        """Scale Pareto front points to zero mean, unit variance per objective."""
        if len(pf) == 0:
            return pf
        scaler = StandardScaler()
        return scaler.fit_transform(pf)

    @staticmethod
    def hypervolume(pf: np.ndarray, ref: np.ndarray) -> float:
        """Dominated hypervolume."""
        if len(pf) == 0:
            return 0.0
        hv = HV(ref_point=ref)
        return float(hv.do(pf))

    @staticmethod
    def generational_distance(pf: np.ndarray, pf_ref: np.ndarray) -> float:
        """Average distance from PF to nearest point in reference PF."""
        if pf.size == 0 or pf_ref.size == 0:
            return np.inf
        pf_scaled = Metrics._scale(pf)
        pf_ref_scaled = Metrics._scale(pf_ref)
        dists = cdist(pf_scaled, pf_ref_scaled)
        return float(np.mean(np.min(dists, axis=1)))

    @staticmethod
    def inverted_generational_distance(pf: np.ndarray, pf_ref: np.ndarray) -> float:
        """Average distance from reference PF to nearest point in PF."""
        if pf.size == 0 or pf_ref.size == 0:
            return np.inf
        pf_scaled = Metrics._scale(pf)
        pf_ref_scaled = Metrics._scale(pf_ref)
        dists = cdist(pf_ref_scaled, pf_scaled)
        return float(np.mean(np.min(dists, axis=1)))

    @staticmethod
    def diversity(pf: np.ndarray) -> float:
        """Spread of Pareto front distances."""
        if len(pf) < 2:
            return 0.0
        pf_scaled = Metrics._scale(pf)
        pf_sorted = pf_scaled[np.argsort(pf_scaled[:, 0])]
        distances = np.linalg.norm(np.diff(pf_sorted, axis=0), axis=1)
        return float(np.std(distances))

    @staticmethod
    def spacing(pf: np.ndarray) -> float:
        """Std deviation of nearest-neighbour distances."""
        if len(pf) < 2:
            return 0.0
        pf_scaled = Metrics._scale(pf)
        dists = cdist(pf_scaled, pf_scaled)
        np.fill_diagonal(dists, np.inf)
        min_dists = np.min(dists, axis=1)
        return float(np.std(min_dists))

    @staticmethod
    def num_pf_points(pf: np.ndarray) -> int:
        """Number of points on Pareto front."""
        return int(len(pf))

    @staticmethod
    def epsilon_indicator(pf: np.ndarray, pf_ref: np.ndarray) -> float:
        """Additive epsilon indicator."""
        if len(pf) == 0 or len(pf_ref) == 0:
            return np.inf
        epsilons = []
        for y_ref in pf_ref:
            epsilons.append(np.min([np.max(y - y_ref) for y in pf]))
        return float(np.max(epsilons))

    @staticmethod
    def coverage_metric(pf: np.ndarray, pf_ref: np.ndarray) -> float:
        """Proportion of reference PF points dominated by approximated PF."""
        if len(pf) == 0 or len(pf_ref) == 0:
            return 0.0
        dominated = 0
        for y_ref in pf_ref:
            if any(np.all(y <= y_ref) and np.any(y < y_ref) for y in pf):
                dominated += 1
        return dominated / len(pf_ref)