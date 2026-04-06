from MOBO import run_mobo
import numpy as np
from random import *



def goal_function(x, alpha=None, beta=None):
    """
    Takes in input parameter values from MOBO optimiser, outputs objective, constraint, penalties, error values
    """
    x=x[0]

    # Dummy objective functions
    y1 = abs(float(x[0]**2+x[1]*2-x[2]*np.cos(x[3])-x[3]/2*x[4]+x[5]*np.exp(x[0])))
    y2 = abs((float(np.sin(x[1])*x[0]**2+x[1]*6-x[2]-x[3]*np.sin(x[1])/2*x[4]*np.cos(x[2])+x[5])))
    y3 = abs(float(np.sin(x[0])*x[0]**2-x[1]*2-x[2]-x[3]*x[4]*np.cos(x[2])+x[5]))
    y4 = float(np.sin(x[1])*x[1]**2+x[1]*2-x[2]-x[1]*np.sin(x[1])/2*x[4]*np.cos(x[2])+x[5])
    y5 = float(np.sin(x[1])*x[0]**2+x[1]*2-x[0]-x[0]*np.sin(x[1])/2*x[4]*np.cos(x[2])+x[5])

    # Dummy rms values for objective measurements
    rms1=random()*2
    rms2=random()*1
    rms3=random()*3

    #Dummy constraint parameter values
    c1 = random()*0.5
    c2 = random()*0.9
    c3 = 0.4
    c4 = 0.06
    c5 = 0.04

    #Dummy penalty parameter values
    pen1 = random()
    pen2 = random()

    

    return {"objectives":[y1, y2,y3], "errors":[rms1,rms2,rms3], "constraints":[c1,c2]}#, "penalties":[pen1, pen2]}




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
    # penalties=penalties,
    # training_csv_path='./Example_initial_samples.csv',

    evaluation_method="GOAL_FUNCTION",
    weighting=True,
    weight=[1.0, 1.0, 2.0],

    iterations=10,
    no_of_meas=2,
    batch_size=1,

    save_name="mobo_run_test",
    working_dir="./MOBO_test",

    resume=False,
    restart_from_iteration=0,

    multiprocess_bool=False,

    use_real_error=True,
    default_objective_error=1e-10,

    goal_function_path="__main__",#"./Example_goal_function.py",
    goal_function_name="goal_function",
    goal_function_kwargs = {'alpha':0.5,'beta':2},

    feasible_tol=0.0,
    constraint_penalty_alpha=1.0
)




###############################
# SELECTION AND VISUALISATION #
###############################
# import matplotlib.pyplot as plt

# #Pareto front
# objective_names = [o["name"] for o in objectives]
# input_names = [o["name"] for o in inputs]
# pareto_front = pf_repo[objective_names]

# plt.clf()
# plot1 = plt.figure()
# plt.plot(pareto_front[objective_names[0]], pareto_front[objective_names[1]],'.',label='Pareto front with \nmaximisation inverted \n(as used in algorithm)')
# for j, o in enumerate(objectives):
#     if o["direction"] == "max":
#         pareto_front[o['name']] = -pareto_front[o['name']]
# plt.plot(pareto_front[objective_names[0]], pareto_front[objective_names[1]],'.',label='Pareto front with \nphysical values')



# #Select Pareto front points
# obj1_min=pf_repo.nsmallest(n=1, columns=objective_names[0])
# obj1_inputs=obj1_min[input_names]
# print('Inputs ofr objective 1 minimised: ',obj1_inputs)
# obj2_max=pf_repo.nlargest(n=1, columns=objective_names[1])
# obj2_inputs=obj2_max[input_names]
# print('Inputs for objective 2 minimised: ',obj1_inputs)
# plt.plot(obj1_min[objective_names[0]],obj1_min[objective_names[1]],'x', label=f'Minimum objective 1')
# plt.plot(obj2_max[objective_names[0]],obj2_max[objective_names[1]],'x', label='Maximum objective 2')

# plt.xlabel(objective_names[0])
# plt.ylabel(objective_names[1])
# plt.title('Final Pareto front')
# plt.legend()
# plot1.savefig('Outputs/Final_pareto_front.png',dpi=200)
# # plot1.show()