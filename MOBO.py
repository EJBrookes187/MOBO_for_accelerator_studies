from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from shutil import move
import argparse
import configparser
import json
import logging
import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from MOBO_optimiser_mp import (
    InitialSetup,
    BatchOptimiser,
    MOBOPlotter,
    EvaluatorFactory,
    compute_pareto_front_constrained,
)

from pymoo.config import Config
Config.warnings['not_compiled'] = False

warnings.filterwarnings("ignore", category=UserWarning)
np.seterr(invalid='ignore')

DERIVED_OBJECTIVES = {"4D_BRIGHTNESS"}


class InteractiveEvaluator:
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
        Y_rms = []

        print("\nEnter objective values:")
        for obj in self.objectives:
            v = self._ask_float(f"  {obj['name']} value: ")
            r = self._ask_float(f"  {obj['name']} RMS  : ")
            Y_obj.append(v)
            Y_rms.append(r)

        Y_con = []
        if self.constraints:
            print("\nEnter constraint values:")
            for con in self.constraints:
                v = self._ask_float(f"  {con['name']} value: ")
                Y_con.append(v)

        Y_all = np.array(Y_obj + Y_con, dtype=float)
        return Y_all, np.array(Y_rms, dtype=float).reshape(1, -1)

    def evaluate_batch(self, X):
        Ys, Yrms = [], []
        for x in X:
            y, r = self.evaluate(x.reshape(1, -1))
            Ys.append(y)
            Yrms.append(r[0])
        return np.array(Ys, dtype=float), np.array(Yrms, dtype=float)


@dataclass
class MOBOConfig:
    # Search space
    input_bounds: list
    inputs: list
    input_columns: list

    # Objective/constraint specs
    objectives: list = None
    objective_directions: list = None
    output_indices: list = None
    reference_point: list = None

    #initial samples
    initial_samples: list = None

    # Core run settings
    evaluation_method: str = 'MANUAL'
    aq: str = 'BATCH_UCB'
    penalise: bool = True
    weighting: bool = False
    weight: list = None
    iterations: int = 50
    no_of_meas: int = 5
    nu: float = 2.5
    batch: bool = True
    batch_size: int = 1
    save_name: str = ''

    # Working directory + folder names
    working_dir: str = ''
    outputs_dirname: str = 'Outputs'
    benchmarks_dirname: str = 'Benchmarks'

    # Optimisation controls
    resume: bool = False
    restart_from_iteration: int | None = None

    # Noise handling
    use_real_rms: bool = False
    default_objective_rms: float = 1e-10

    # Goal-function evaluator options
    goal_function_path: str = ""
    goal_function_name: str = "goal_function"

    # Model / training data paths (previously hardcoded)
    model_path: str = ""
    training_csv_path: str = ""

    # Constraint penalty
    feasible_tol: float = 0.0
    constraint_penalty_alpha: float = 0.0  # 0 disables

    # Plotting
    plot_acquisition: bool = False

    # GP hyperparameter optimisation controls
    auto_length_scales: bool = True
    gp_n_restarts_optimizer: int = 5
    gp_length_scale_lower_factor: float = 1e-6
    gp_length_scale_upper_factor: float = 10.0
    gp_length_scale_init_factor: float = 0.2



def parse_initial_samples(parser):
    samples = []

    if "INITIAL_SAMPLES" not in parser:
        return samples

    for key, val in parser["INITIAL_SAMPLES"].items():
        if not val.strip():
            continue

        # inputs | objectives | rms | constraints
        parts = [p.strip() for p in val.split("|")]

        x = [float(v) for v in parts[0].split(",")]
        y = [float(v) for v in parts[1].split(",")]

        rms = [float(v) for v in parts[2].split(",")] if len(parts) > 2 else None
        con = [float(v) for v in parts[3].split(",")] if len(parts) > 3 else None

        samples.append({
            "X": x,
            "Y": y,
            "RMS": rms,
            "C": con
        })

    return samples


def load_txt_config(path):
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(path)

    if "INPUTS" not in parser or "OBJECTIVES" not in parser:
        raise ValueError("Config must contain [INPUTS] and [OBJECTIVES] sections")

    # ---------- helpers ----------
    def get_general(key, default=None):
        if "GENERAL" not in parser:
            return default
        return parser["GENERAL"].get(key, fallback=default)

    def as_bool(v, default=False):
        if v is None:
            return default
        return str(v).strip().lower() in ("1", "true", "t", "yes", "y", "on")

    def as_int(v, default=0):
        if v is None or str(v).strip() == "":
            return default
        return int(v)

    def as_float(v, default=0.0):
        if v is None or str(v).strip() == "":
            return default
        return float(v)

    def as_list_floats(v, default=None):
        if v is None:
            return default
        s = str(v).strip()
        if not s:
            return default
        s = s.strip("[]")
        return [float(x.strip()) for x in s.split(",") if x.strip()]

    # ---------- INPUTS ----------
    inputs = []
    input_bounds = []
    for name, value in parser["INPUTS"].items():
        vmin, vmax, step = [float(x.strip()) for x in value.split(",")]
        inputs.append({"name": name.strip(), "min": vmin, "max": vmax, "step": step})
        input_bounds.append((vmin, vmax))
    input_columns = [inp["name"] for inp in inputs]
    print("INPUT_COLUMNS:", input_columns)

    # ---------- OBJECTIVES ----------
    objectives = []
    for name, value in parser["OBJECTIVES"].items():
        parts = [p.strip() for p in value.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid objective: {name} = {value}. Expected: direction, reference_point")
        direction, ref_val = parts
        direction = direction.lower()
        if direction not in ("min", "max"):
            raise ValueError(f"Invalid direction for objective '{name}': {direction}. Use min or max.")
        objectives.append({"name": name.strip(), "direction": direction, "ref_value": float(ref_val)})

    # ---------- CONSTRAINTS ----------
    constraints = []
    if "CONSTRAINTS" in parser:
        for name, value in parser["CONSTRAINTS"].items():
            parts = [v.strip() for v in value.split(",")]
            if len(parts) == 2:
                sense, thresh = parts
                scale = None
            elif len(parts) == 3:
                sense, thresh, scale = parts
            else:
                raise ValueError(
                    f"Constraint '{name}' must be 'sense, threshold' or 'sense, threshold, scale'. Got: {value}"
                )

            if sense not in ("<=", ">="):
                raise ValueError(f"Invalid sense for constraint '{name}': {sense}. Use <= or >=")

            constraints.append({
                "name": name.strip(),
                "sense": sense,
                "threshold": float(thresh),
                "scale": (float(scale) if scale is not None and str(scale).strip() != "" else None),
            })
    
    # initial samples
    initial_samples = parse_initial_samples(parser)
    

    # restart / resume
    resume = as_bool(get_general("resume", False))

    _restart_from_raw = get_general("restart_from_iteration", None)
    restart_from_iteration = None
    if _restart_from_raw is not None and str(_restart_from_raw).strip() != "":
        restart_from_iteration = int(str(_restart_from_raw).strip())
    
    goal_function_path = get_general("goal_function_path", None)
    if goal_function_path is None or str(goal_function_path).strip() == "":
        goal_function_path = Path(os.getcwd()).resolve()
    else:
        goal_function_path = Path(str(goal_function_path)).expanduser().resolve()

    working_dir_cfg = get_general("working_dir", None)

    # Standard subfolders under the working directory
    outputs_dirname = "Outputs"
    benchmarks_dirname = "Benchmarks"

    # Resolve working directory (default: current directory)
    if working_dir_cfg is None or str(working_dir_cfg).strip() == "":
        working_dir = Path(os.getcwd()).resolve()
    else:
        working_dir = Path(str(working_dir_cfg)).expanduser().resolve()

    # Ensure standard subfolders exist
    (working_dir / outputs_dirname).mkdir(parents=True, exist_ok=True)
    (working_dir / benchmarks_dirname).mkdir(parents=True, exist_ok=True)

    objective_names = [o["name"] for o in objectives]
    objective_directions = [o["direction"] for o in objectives]
    reference_point = [o["ref_value"] for o in objectives]

    cfg = MOBOConfig(
        evaluation_method=str(get_general("evaluation_method", "Manual")),
        weighting=as_bool(get_general("weighting", False)),
        weight=as_list_floats(get_general("weight", None), default=[1.0] * len(objective_names)),
        iterations=as_int(get_general("iterations", 10)),
        no_of_meas=as_int(get_general("no_of_meas", 5)),
        batch_size=as_int(get_general("batch_size", 1)),
        save_name=str(get_general("save_name", "mobo_run")),
        input_bounds=input_bounds,
        inputs=inputs,
        input_columns=input_columns,
        resume=bool(resume),
        restart_from_iteration=restart_from_iteration,
        reference_point=reference_point,
        use_real_rms=as_bool(get_general("use_real_rms", False)),
        default_objective_rms=as_float(get_general("default_objective_rms", 1e-10)),
        objectives=objective_names,
        objective_directions=objective_directions,
        initial_samples=initial_samples,
        goal_function_path=str(get_general("goal_function_path", "")),
        goal_function_name=str(get_general("goal_function_name", "goal_function")),
        model_path=str(get_general("model_path", "")),
        training_csv_path=str(get_general("training_csv_path", "")),
        feasible_tol=as_float(get_general("feasible_tol", 0.0)),
        constraint_penalty_alpha=as_float(get_general("constraint_penalty_alpha", 0.0)),
        auto_length_scales=True,
        gp_n_restarts_optimizer=5,
        gp_length_scale_lower_factor=1e-6,
        gp_length_scale_upper_factor=10.0,
        gp_length_scale_init_factor=0.2,
        working_dir=str(working_dir),
        outputs_dirname=str(outputs_dirname),
        benchmarks_dirname=str(benchmarks_dirname),
    )

    return cfg, inputs, objectives, constraints


def build_mobo_config(
    *,
    inputs,
    objectives,
    constraints=None,
    input_columns=[],
    initial_samples=None,

    evaluation_method="Manual",
    weighting=False,
    weight=None,
    iterations=10,
    no_of_meas=5,
    batch_size=1,
    save_name="mobo_run",

    resume=False,
    restart_from_iteration=None,

    use_real_rms=False,
    default_objective_rms=1e-10,

    goal_function_path="",
    goal_function_name="goal_function",

    model_path="",
    training_csv_path="",

    feasible_tol=0.0,
    constraint_penalty_alpha=0.0,

    working_dir=None,
    outputs_dirname="Outputs",
    benchmarks_dirname="Benchmarks",
):
    """
    Build MOBOConfig + validated inputs/objectives/constraints from Python arguments
    """
    constraints = constraints or []

    # INPUTS
    input_bounds = [(inp["min"], inp["max"]) for inp in inputs]
    input_columns = [inp["name"] for inp in inputs]

    # OBJECTIVES
    objective_names = [o["name"] for o in objectives]
    objective_directions = [o["direction"].lower() for o in objectives]
    reference_point = [o["ref_value"] for o in objectives]

    # Default weights
    if weight is None:
        weight = [1.0] * len(objective_names)

    # Resolve working dir
    if working_dir is None or str(working_dir).strip() == "":
        wd = Path(os.getcwd()).resolve()
    else:
        wd = Path(str(working_dir)).expanduser().resolve()

    (wd / outputs_dirname).mkdir(parents=True, exist_ok=True)
    (wd / benchmarks_dirname).mkdir(parents=True, exist_ok=True)

    cfg = MOBOConfig(
        evaluation_method=str(evaluation_method),
        weighting=bool(weighting),
        weight=list(weight),
        iterations=int(iterations),
        no_of_meas=int(no_of_meas),
        batch_size=int(batch_size),
        save_name=str(save_name),
        input_bounds=input_bounds,
        inputs=inputs,
        input_columns=input_columns,
        resume=bool(resume),
        restart_from_iteration=restart_from_iteration,
        reference_point=reference_point,
        use_real_rms=bool(use_real_rms),
        default_objective_rms=float(default_objective_rms),
        objectives=objective_names,
        objective_directions=objective_directions,
        initial_samples=initial_samples,
        goal_function_path=str(goal_function_path),
        goal_function_name=str(goal_function_name),
        model_path=str(model_path),
        training_csv_path=str(training_csv_path),
        feasible_tol=float(feasible_tol),
        constraint_penalty_alpha=float(constraint_penalty_alpha),
        auto_length_scales=True,
        gp_n_restarts_optimizer=5,
        gp_length_scale_lower_factor=1e-6,
        gp_length_scale_upper_factor=10.0,
        gp_length_scale_init_factor=0.2,
        working_dir=str(wd),
        outputs_dirname=str(outputs_dirname),
        benchmarks_dirname=str(benchmarks_dirname),
    )

    return cfg, inputs, objectives, constraints


class MOBOExperiment:
    def __init__(self, config: MOBOConfig, inputs, objectives, constraints, plotter: MOBOPlotter = None):
        self.config = config
        self.inputs = inputs
        self.objectives = objectives
        self.constraints = constraints
        self.inputs_spec = inputs
        self.input_columns = self.config.input_columns
        self.objectives_spec = objectives
        self.constraints_spec = constraints
        self.logger = logging.getLogger("MOBO_outputs")
        self.plotter = plotter or MOBOPlotter()
        self.timestamp = datetime.now().strftime("%Y%m%d")
        self.save_prefix = f"{self.timestamp}_{self.config.save_name}"

        self.evaluator = EvaluatorFactory.create(
            self.config.evaluation_method,
            model_path=self.config.model_path,
            inputs=self.inputs,
            objectives=self.objectives,
            constraints=self.constraints,
            goal_function_path=self.config.goal_function_path,
            goal_function_name=self.config.goal_function_name,
            default_objective_rms=self.config.default_objective_rms,
        )

        self.input_repository = None
        self.output_repository = None
        self.constraint_repository = None
        self.rms_repository = None

        # Extra repositories for full post-analysis
        self.raw_repository = None
        self.constraint_values_repository = None
        self.constraint_violations_repository = None
        self.gp_models = None
        self.reference_point = None
        self.best_pareto_front = None

    def setup(self):
        self.logger.info("Setting up MOBO experiment")
        self.all_labels = []
        self.input_cols = []
        self.output_cols = []

        

        setup = InitialSetup(
            self,
            csv_path=self.config.training_csv_path,
            input_columns=self.input_cols,
            all_labels=self.all_labels,
            objective_names=self.config.objectives,
            evaluator=self.evaluator,
            input_bounds=self.config.input_bounds,
            reference_point=self.config.reference_point,
            random_seed=np.random.randint(0, 1e6)
        )

        result = setup.run(no_of_init_samples=self.config.no_of_meas)

        # Attach results to optimisation
        self.input_repository = result.input_repository
        self.output_repository = result.output_repository
        self.gp_models = result.gp_models
        self.best_pareto_front = result.best_pareto_front
        self.best_pareto_inputs = result.best_pareto_inputs
        self.init_pareto_front = result.init_pareto_front
        self.init_pareto_inputs = result.init_pareto_inputs
        self.reference_point = setup.reference_point
        self.output_labels = self.output_cols

    def run(self):
        self.setup()

        optimiser = BatchOptimiser(self)
        results = optimiser.run(resume=self.config.resume)
        self.save_results(results)

        pf = np.asarray(results["pareto_front"])
        pf_inputs = np.asarray(results["pareto_front_inputs"])
        hv = results["hypervolume"]
        gd = results["generational_distance"]
        div = results["diversity"]
        spacing = results["spacing"]
        num_pf = results["num_pf_points"]
        runtime = results["timed_iterations"]
        X = results["X_repo"]
        Y = results["Y_repo"]

        # Plotting / summary
        outdir = Path(self.config.working_dir) / self.config.outputs_dirname
        outdir.mkdir(exist_ok=True)
        experiment_dir = outdir / self.save_prefix
        experiment_dir.mkdir(exist_ok=True)

        self.plotter.plot_hypervolume_evolution(
            hv, self.config.evaluation_method, experiment_dir, self.config.iterations, self.config.save_name
        )

        pf, pf_idx = compute_pareto_front_constrained(Y, self.constraint_repository)

        self.plotter.plot_pareto_comparison(
            pf, self.best_pareto_front, self.init_pareto_front,
            self.config.output_indices, self.all_labels,
            self.config.evaluation_method, experiment_dir, self.config.save_name
        )

        self.plotter.plot_pareto_front_colored(
            X, Y, pf_idx, self.config.iterations, self.config.iterations, self.config.save_name
        )

        self.plotter.plot_metrics_evolution(
            hypervolume=hv,
            spacing=spacing,
            generational_distance=gd,
            diversity=div,
            num_pf_points=num_pf,
            runtime=runtime,
            save_prefix=experiment_dir
        )

        return results

    def save_results(self, results: dict):
        outdir = Path(getattr(self.config, "working_dir", ".")) / str(getattr(self.config, "outputs_dirname", "Outputs"))
        outdir.mkdir(exist_ok=True)
        experiment_dir = outdir / self.save_prefix
        experiment_dir.mkdir(exist_ok=True)
        prefix = experiment_dir / f"{self.config.save_name}_{self.config.evaluation_method}"

        # Save PF
        np.savetxt(f"{prefix}_pareto_front.csv", results["pareto_front"], delimiter=",")

        # Save metrics dynamically
        for key, arr in results.items():
            if key in ["pareto_front", "X_repo", "Y_repo"]:
                continue
            arr = np.asarray(arr, dtype=float)
            np.savetxt(f"{prefix}_{key}.csv", arr, delimiter=",")

        # Save repositories
        np.savetxt(f"{prefix}_initial_pf.csv", self.init_pareto_front, delimiter=",")
        np.savetxt(f"{prefix}_input_samples.csv", results["X_repo"], delimiter=",")
        np.savetxt(f"{prefix}_samples.csv", results["Y_repo"], delimiter=",")
        self.save_metadata(prefix)
        self.logger.info(f"Saved results and metadata to {prefix.parent}")

        # Save GP Models
        gp_save_path = f"{prefix}_gp_models.pkl"
        with open(gp_save_path, "wb") as f:
            pickle.dump(self.gp_models, f)

        # Save Pareto Front Input Coordinates
        if "pareto_front_inputs" in results:
            pf_inputs = np.asarray(results["pareto_front_inputs"])
        if "pareto_front_rms" in results:
            pf_rms = np.asarray(results["pareto_front_rms"])
        else:
            CV_repo = np.asarray(results["CV_repo"], dtype=float)
            pf, pf_idx = compute_pareto_front_constrained(results["Y_repo"], CV_repo)
            pf_inputs = np.asarray(results["X_repo"])[pf_idx]
            pf_rms = np.asarray(results["rms_repo"])[pf_idx]

        np.savetxt(f"{prefix}_pareto_inputs.csv", pf_inputs, delimiter=",")
        np.savetxt(f"{prefix}_pareto_rms.csv", pf_rms, delimiter=",")
        self.logger.info(f"Saved results and metadata to {prefix.parent}")

    def save_metadata(self, prefix):
        outdir = Path(getattr(self.config, "working_dir", ".")) / str(getattr(self.config, "outputs_dirname", "Outputs"))
        outdir.mkdir(exist_ok=True)

        metadata = asdict(self.config)
        metadata.update({
            "timestamp": self.timestamp,
            "evaluation_method": self.config.evaluation_method,
        })

        metadata["objectives_spec"] = self.objectives_spec
        metadata["constraints_spec"] = self.constraints_spec
        metadata["inputs_spec"] = self.inputs_spec

        meta_path = f"{prefix}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)


#################
# MULTIPLE RUNS #
#################

def save_checkpoint(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"[Checkpoint] Saved to {filename}")


def load_checkpoint(filename):
    if Path(filename).exists():
        with open(filename, "rb") as f:
            data = pickle.load(f)
        print(f"[Checkpoint] Loaded from {filename}")
        return data
    return None


def benchmark_single_config(cfg, inputs, objectives, constraints, n_runs=1, output_dir="", checkpoint_file=""):
    """
    Run the optimiser multiple times (with random seeds) and save results
    """
    if not output_dir:
        output_dir = str(Path(getattr(cfg, "working_dir", ".")) / str(getattr(cfg, "outputs_dirname", "Outputs")))
    if not checkpoint_file:
        checkpoint_file = str(
            Path(getattr(cfg, "working_dir", ".")) /
            str(getattr(cfg, "benchmarks_dirname", "Benchmarks")) /
            f"{cfg.save_name}_checkpoint.pkl"
        )

    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d")

    start_run = 0
    progress = {"completed_runs": 0}
    start_iter = cfg.restart_from_iteration

    if cfg.resume:
        loaded_progress = load_checkpoint(checkpoint_file)
        if isinstance(loaded_progress, dict) and "completed_runs" in loaded_progress:
            progress = loaded_progress
        start_run = progress["completed_runs"]

    print(f"Starting run from iteration: {start_iter}")

    best_pf = None
    base_save_name = cfg.save_name

    for seed in range(start_run, n_runs):
        cfg.save_name = f"{base_save_name}_run{seed}"
        print(f"\n[Benchmark] Run {seed + 1}/{n_runs}")
        np.random.seed(seed)

        exp = MOBOExperiment(cfg, inputs, objectives, constraints)
        print("OBJ SPECS:", [o["name"] for o in exp.objectives_spec])
        print("CON SPECS:", [c["name"] for c in exp.constraints_spec])

        _ = exp.run()

        best_pf = exp.best_pareto_front

        progress["completed_runs"] += 1
        save_checkpoint(progress, checkpoint_file)

    # restore original save name
    cfg.save_name = base_save_name


def get_run_dir(base_output_dir, timestamp, save_name, run_id):
    return os.path.join(base_output_dir, f"{timestamp}_{save_name}_run{run_id}")


def summarize_results(cfg, n_runs, timestamp, output_dir="", best_pf=None):
    """Read saved CSV metric files and create summary plots and final PF plot."""
    save_name = cfg.save_name
    metrics = ["hypervolume", "generational_distance", "spacing", "diversity", "num_pf_points"]
    output_dir = str(Path(getattr(cfg, "working_dir", ".")) / str(getattr(cfg, "outputs_dirname", "Outputs")))
    summary_dir = os.path.join(output_dir, f"Summary_{timestamp}_{cfg.save_name}")
    os.makedirs(output_dir, exist_ok=True)

    for metric in metrics:
        all_runs = []
        for seed in range(n_runs):
            run_folder = f"{output_dir}/{timestamp}_{save_name}_run{seed}"
            filename = f"{run_folder}/{save_name}_run{seed}_{cfg.evaluation_method}_{metric}.csv"

            if os.path.exists(filename):
                data = pd.read_csv(filename, header=None).values.flatten()
                all_runs.append(data)
            else:
                print(f"[Warning] Missing metric file: {filename}")

        if not all_runs:
            print(f"[Warning] No data found for {metric}. Skipping.")
            continue

        max_len = max(len(r) for r in all_runs)
        padded = np.array([np.pad(r, (0, max_len - len(r)), mode='constant', constant_values=np.nan) for r in all_runs])
        mean, std = np.nanmean(padded, axis=0), np.nanstd(padded, axis=0)



def run_mobo(
    *,
    inputs,
    objectives,
    constraints=None,
    initial_samples=None,

    evaluation_method="Manual",
    weighting=False,
    weight=None,
    iterations=100,
    no_of_meas=20,
    batch_size=1,
    save_name="mobo_run",

    resume=False,
    restart_from_iteration=None,

    use_real_rms=False,
    default_objective_rms=1e-10,

    goal_function_path="",
    goal_function_name="goal_function",

    model_path=None, 
    training_csv_path="",

    feasible_tol=0.0,
    constraint_penalty_alpha=0.0,

    working_dir=None,
    outputs_dirname="Outputs",
    benchmarks_dirname="Benchmarks",

    n_runs=1,
):
    """
    Main importable function
    """
    cfg, inputs, objectives, constraints = build_mobo_config(
        inputs=inputs,
        objectives=objectives,
        constraints=constraints,
        evaluation_method=evaluation_method,
        weighting=weighting,
        weight=weight,
        iterations=iterations,
        no_of_meas=no_of_meas,
        batch_size=batch_size,
        save_name=save_name,
        resume=resume,
        restart_from_iteration=restart_from_iteration,
        use_real_rms=use_real_rms,
        default_objective_rms=default_objective_rms,
        goal_function_path=goal_function_path,
        goal_function_name=goal_function_name,
        model_path=model_path,
        training_csv_path=training_csv_path,
        feasible_tol=feasible_tol,
        constraint_penalty_alpha=constraint_penalty_alpha,
        working_dir=working_dir,
        outputs_dirname=outputs_dirname,
        benchmarks_dirname=benchmarks_dirname,
    )

    if getattr(cfg, "working_dir", None):
        os.makedirs(cfg.working_dir, exist_ok=True)
        os.chdir(cfg.working_dir)

    print("OBJECTIVES LOADED:", [o["name"] for o in objectives], "count =", len(objectives))

    checkpoint_file = str(Path(cfg.working_dir) / cfg.benchmarks_dirname / f"{cfg.save_name}.pkl")

    benchmark_single_config(
        cfg,
        inputs,
        objectives,
        constraints,
        n_runs=n_runs,
        checkpoint_file=checkpoint_file,
    )

    return cfg


def run_mobo_from_config(config_path, n_runs=1):
    """
    Backward-compatible file-based entrypoint.
    """
    cfg, inputs, objectives, constraints = load_txt_config(config_path)

    if getattr(cfg, "working_dir", None):
        os.makedirs(cfg.working_dir, exist_ok=True)
        os.chdir(cfg.working_dir)

    print("OBJECTIVES LOADED:", [o["name"] for o in objectives], "count =", len(objectives))

    checkpoint_file = str(Path(cfg.working_dir) / cfg.benchmarks_dirname / f"{cfg.save_name}.pkl")

    benchmark_single_config(
        cfg,
        inputs,
        objectives,
        constraints,
        n_runs=n_runs,
        checkpoint_file=checkpoint_file,
    )

    return cfg



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-runs", type=int, default=1)
    args = parser.parse_args()

    run_mobo_from_config(args.config, n_runs=args.n_runs)