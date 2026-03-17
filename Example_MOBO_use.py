from MOBO import run_mobo

inputs = [
    {"name": "Quad1 current norm", "min": 0.0, "max": 1.0, "step": 0.1},
    {"name": "Quad2 current norm", "min": 0.0, "max": 1.0, "step": 0.1},
    {"name": "Steerer1x current norm", "min": 0.0, "max": 1.0, "step": 0.1},
    {"name": "Steerer2x current norm", "min": 0.0, "max": 1.0, "step": 0.1},
    {"name": "Steerer1y current norm", "min": 0.0, "max": 1.0, "step": 0.1},
    {"name": "Steerer2y current norm", "min": 0.0, "max": 1.0, "step": 0.1},
]

objectives = [
    {"name": "y beam size [mm]", "direction": "min", "ref_value": 100.0},
    {"name": "beam intensity [A]", "direction": "max", "ref_value": 100.0},
]

constraints = [
    {"name": "x beam size [mm]", "sense": "<=", "threshold": 10.0, "scale": 10.0},
    {"name": "x beam position on screen [mm]", "sense": "<=", "threshold": 0.5, "scale": 0.5},
]

run_mobo(
    inputs=inputs,
    objectives=objectives,
    constraints=constraints,
    training_csv_path='/.../Example_initial_samples.csv',

    evaluation_method="GOAL_FUNCTION",
    weighting=False,
    weight=[1.0, 1.0],

    iterations=30,
    no_of_meas=10,
    batch_size=5,

    save_name="mobo_run0",
    working_dir="/.../MOBO_test/",

    resume=False,
    restart_from_iteration=0,

    use_real_rms=True,
    default_objective_rms=1e-10,

    goal_function_path="/.../Example_goal_function.py",
    goal_function_name="goal_function",

    feasible_tol=0.0,
    constraint_penalty_alpha=0.0
)