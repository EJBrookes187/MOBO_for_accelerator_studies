import numpy as np
from random import *

def goal_function(x, alpha=None, beta=None):
    """
    Takes in input parameter values from MOBO optimiser, outputs objective values
    """
    x=x[0]
    print('x:',x)

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