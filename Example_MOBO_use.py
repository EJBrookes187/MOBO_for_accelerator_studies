from MOBO import run_mobo
import numpy as np

inputs = [
    {"name": "Quad1_current_[A]", "min": 0.0, "max": 1.0, "step": 0.1},
    {"name": "Quad2_current_[A]", "min": 0.0, "max": 1.0, "step": 0.1},
    {"name": "Steerer1x_current_[A]", "min": 0.0, "max": 1.0, "step": 0.1},
    {"name": "Steerer2x_current_[A]", "min": 0.0, "max": 1.0, "step": 0.1},
    {"name": "Steerer1y_current_[A]", "min": 0.0, "max": 1.0, "step": 0.1},
    {"name": "Steerer2y_current_[A]", "min": 0.0, "max": 1.0, "step": 0.1},
]

objectives = [
    {"name": "y_beam_size_[mm]", "direction": "min", "ref_value": 100.0},
    {"name": "beam_intensity", "direction": "max", "ref_value": 100.0},
    {"name": "y_position_[mm]", "direction": "min", "ref_value": 100.0},
]

constraints = [
    {"name": "x_beam_size_[mm]", "sense": "<=", "threshold": 10.0, "scale": 10.0},
    {"name": "x_beam_position_[mm]", "sense": "<=", "threshold": 0.5, "scale": 0.5},
]

penalties = [
    {"name": "intensity", "lower": 0, "upper":np.inf, "scale": 2, "rate": 2},
    {"name": 'y_position_[mm]', "lower": 0.05, "upper":0.9, "scale": 1, "rate": 1},
]

repo, pf_repo, metrics_repo = run_mobo(
    inputs=inputs,
    objectives=objectives,
    constraints=constraints,
    penalties=penalties,
    training_csv_path='/mnt/Windows/Profile/y2326/Documents/BayesOpt/PITZ_MOBO_vs_MOGA/MOBO_test_scripts/Example_initial_samples.csv',

    evaluation_method="GOAL_FUNCTION",
    weighting=True,
    weight=[1.0, 1.0, 1.0],

    iterations=50,
    no_of_meas=2,
    batch_size=1,

    save_name="mobo_run1",
    working_dir="/mnt/Windows/Profile/y2326/Documents/BayesOpt/PITZ_MOBO_vs_MOGA/MOBO_test_scripts/MOBO_test",

    resume=False,
    restart_from_iteration=0,

    multiprocess_bool=True,

    use_real_error=True,
    default_objective_error=1e-10,

    goal_function_path="/mnt/Windows/Profile/y2326/Documents/BayesOpt/PITZ_MOBO_vs_MOGA/MOBO_test_scripts/Example_goal_function.py",
    goal_function_name="goal_function",
    goal_function_kwargs = {'alpha':0.5,'beta':2},

    feasible_tol=0.0,
    constraint_penalty_alpha=1.0
)




###############################
# SELECTION AND VISUALISATION #
###############################
import matplotlib.pyplot as plt

#Pareto front
objective_names = [o["name"] for o in objectives]
input_names = [o["name"] for o in inputs]
pareto_front = pf_repo[objective_names]

plt.clf()
plot1 = plt.figure()
plt.plot(pareto_front[objective_names[0]], pareto_front[objective_names[1]],'.',label='Pareto front with \nmaximisation inverted \n(as used in algorithm)')
for j, o in enumerate(objectives):
    if o["direction"] == "max":
        pareto_front[o['name']] = -pareto_front[o['name']]
plt.plot(pareto_front[objective_names[0]], pareto_front[objective_names[1]],'.',label='Pareto front with \nphysical values')



#Select Pareto front points
obj1_min=pf_repo.nsmallest(n=1, columns=objective_names[0])
obj1_inputs=obj1_min[input_names]
print('Inputs ofr objective 1 minimised: ',obj1_inputs)
obj2_max=pf_repo.nlargest(n=1, columns=objective_names[1])
obj2_inputs=obj2_max[input_names]
print('Inputs for objective 2 minimised: ',obj1_inputs)
plt.plot(obj1_min[objective_names[0]],obj1_min[objective_names[1]],'x', label=f'Minimum objective 1')
plt.plot(obj2_max[objective_names[0]],obj2_max[objective_names[1]],'x', label='Maximum objective 2')

plt.xlabel(objective_names[0])
plt.ylabel(objective_names[1])
plt.title('Final Pareto front')
plt.legend()
plot1.savefig('Outputs/Final_pareto_front.png',dpi=200)
plot1.show()