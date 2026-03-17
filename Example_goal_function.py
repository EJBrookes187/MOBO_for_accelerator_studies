import numpy as np
from random import *

def goal_function(x, X=None, inputs=None):
    """
    Takes in input parameter values from MOBO optimiser, outputs objective values
    """

    # Dummy objective functions
    y1 = float(x[0]**2+x[1]*2-x[2]*np.cos(x[3])-x[3]/2*x[4]+x[5]*np.exp(x[0]))
    y2 = float(np.sin(x[1])*x[0]**2+x[1]*2-x[2]-x[3]*np.sin(x[1])/2*x[4]*np.cos(x[2])+x[5])
    y3 = float(np.sin(x[0])*x[0]**2+x[1]*2-x[2]-x[3]*np.sin(x[1])/2*x[4]*np.cos(x[2])+x[5])
    y4 = float(np.sin(x[1])*x[1]**2+x[1]*2-x[2]-x[1]*np.sin(x[1])/2*x[4]*np.cos(x[2])+x[5])
    y5 = float(np.sin(x[1])*x[0]**2+x[1]*2-x[0]-x[0]*np.sin(x[1])/2*x[4]*np.cos(x[2])+x[5])

    # Dummy rms values for objective measurements
    rms1=random()*0.2
    rms2=random()*0.1

    #Dummy constraint parameter values
    c1 = random()*0.5
    c2 = random()*0.9
    c3 = 0.4
    c4 = 0.06
    c5 = 0.04

    

    return [y1, y2], [c1,c2], [rms1,rms2]