from cvxpy import *
from gekko import GEKKO
import numpy as np
import scipy as sp
import scipy.sparse as sparse
from scipy.optimize import fsolve

import matplotlib.pyplot as plt

SIM_TIME = 12
VEH_COUNT = 9
PRD_HRZ = 10
dt = 0.25
Ca = 0.57 # aerodynamic drag coefficient (kg/m)
f = 0.02 # coefficient of rolling resistance.
g = 9.8 # gravitational acceleration
ds = 5 # saftey gap

t=np.linspace(0,SIM_TIME,SIM_TIME/dt*50)

def simulation(vehicle_state,input):
    m = GEKKO()    # create GEKKO model
    x=m.Array(m.Var,(VEH_COUNT,1))
    v=m.Array(m.Var,(VEH_COUNT,1))
    u=m.Array(m.Var,(VEH_COUNT,1))
    mass=m.Array(m.Param,(VEH_COUNT,1))
    tau=m.Array(m.Param,(VEH_COUNT,1))

    for i in range(0, VEH_COUNT):
        x[i, 0].value = vehicle_state[i][0]  # create GEKKO variable
        v[i, 0].value = vehicle_state[i][1]
        u[i, 0].value = vehicle_state[i][3]
        mass[i, 0].value = vehicle_state[i][2]
        tau[i,0].value = vehicle_state[i][4]

        m.Equation(x[i,0].dt() == v[i, 0])  # create GEEKO equation
        m.Equation(u[i,0].dt() == 1/tau[i,0]*input[i] - 1/tau[i,0]*u[i,0])
        m.Equation(v[i,0].dt() == 1/mass[i,0]*(u[i,0]-Ca*v[i, 0]*v[i, 0]-mass[i,0]*g*f))
        m.time = np.linspace(0, dt)  # time points

    m.options.IMODE = 4
    m.solve(False)

    for i in range(0,VEH_COUNT):
        vehicle_state[i][0] = x[i,0].value[-1]
        vehicle_state[i][1] = v[i,0].value[-1]
        vehicle_state[i][3] = u[i,0].value[-1]


    print(str(x[0,0].value))

    print("x[0,0].value[-1]=" + str(x[0,0].value[-1]))
    print("v[0,0].value[-1]=" + str(v[0,0].value[-1]))

    x_ct = []
    v_ct = []

    for i in range(0,VEH_COUNT):
        x_ct.append(x[i,0].value)
        v_ct.append(v[i,0].value)


    return vehicle_state, x_ct, v_ct


if __name__ == "__main__":

########################## Initalize for simulation ###############

    vehicle_state=np.zeros((VEH_COUNT,5))

    mmin=1000  # kg for passenger car
    mmax=15000 # kg for heavy duty truck

    m=mmin*np.ones(VEH_COUNT)+np.multiply(np.random.rand(VEH_COUNT),(mmax-mmin)*np.ones(VEH_COUNT)) #1t ~ 15t
    alpha = np.divide(m-np.ones(VEH_COUNT),mmax*np.ones(VEH_COUNT))
    tau=0.2*(np.ones(VEH_COUNT)-alpha) + 0.6*alpha

    amin=-3.0*2.2 +3.0/mmax*m
    amax=-0.92*amin

    for i in range(0, VEH_COUNT):
        vehicle_state[i][0] = i * -60 # distance from origin
        vehicle_state[i][1] = np.random.rand(1)*6+31   # velocity : 31 +- 10% m/s , 31m/s = 111km/h)
        vehicle_state[i][2] = m[i]  # mass
        vehicle_state[i][3] = 0  # Initial F=0
        vehicle_state[i][4] = tau[i]

        print(str(i) + "th vehicle" + "int dis : " + str(vehicle_state[i][0])
                                    + "m ini vel : " + str(vehicle_state[i][1])
                                    + "m/s mass : " + str(vehicle_state[i][2]) + "kg")


    x_res = np.zeros((VEH_COUNT,t.size))
    v_res = np.zeros((VEH_COUNT,t.size))


    F_des=Variable((VEH_COUNT,PRD_HRZ))
    x=Variable((VEH_COUNT,PRD_HRZ+1))
    v=Variable((VEH_COUNT,PRD_HRZ+1))
    F=Variable((VEH_COUNT,PRD_HRZ+1))


    F_init=Parameter(VEH_COUNT)
    x_init=Parameter(VEH_COUNT)
    v_init=Parameter(VEH_COUNT)
    v_0 = Parameter(VEH_COUNT)

    objective = 0
    constraints = [x[:,0] == x_init] + [v[:,0]==v_init] + [F[:,0]==F_init]

    for j in range (0,PRD_HRZ):
        for i in range (0,VEH_COUNT):
            #print("i="+str(i)+", j="+str(j))
            constraints += [x[i,j+1] == x[i,j]+dt*v[i,j]]
            constraints += [v[i,j+1] == v[i,j] + dt*(1/m[i]*F[i,j]-1/m[i]*Ca*v_init[i]*v_init[i]-g*f)]
            constraints += [1/m[i]*F[i,j]-1/m[i]*Ca*v_init[i]*v_init[i]-g*f >= amin[i]]
            constraints += [1/m[i]*F[i,j]-1/m[i]*Ca*v_init[i]*v_init[i]-g*f <= amax[i]]
            #constraints += [v[i,j+1] == v[i,j] + dt*(1/m[i]*F[i,j]-1/m[i]*Ca*v[i,j]*v[i,j]-g*f)]

            constraints += [F[i,j+1] == (tau[i]-dt)/tau[i]*F[i,j] + dt/tau[i]*F_des[i,j]]
        constraints += [1/m[1]*F[1,j]-1/m[1]*Ca*v_init[1]*v_init[1]-g*f <= -amin[1]]
        constraints += [1/m[VEH_COUNT-1]*F[VEH_COUNT-1,j]-1/m[VEH_COUNT-1]*Ca*v_init[VEH_COUNT-1]*v_init[VEH_COUNT-1]-g*f <= -0.92*amin[VEH_COUNT-1]]

            #constraints += [amin[i] <= v[i,j+1]-v[i,j], v[i,j+1]-v[i,j] <= amax[i]]

    for i in range(1,VEH_COUNT):
        constraints += [x[i-1,1:PRD_HRZ+1] - x[i,1:PRD_HRZ+1] >= ds*np.ones(PRD_HRZ)]

        objective += 1/2*m[i]*quad_form(v[i-1,1:PRD_HRZ+1]-v[i,1:PRD_HRZ+1],sparse.eye(PRD_HRZ))
        objective += 1/2*m[i-1]*quad_form(v[i-1, 1:PRD_HRZ + 1], sparse.eye(PRD_HRZ))


    objective += 1/2*m[VEH_COUNT-1]*quad_form(v[VEH_COUNT-1-1, 1:PRD_HRZ + 1], sparse.eye(PRD_HRZ))
    prob = Problem(Minimize(objective),constraints)

    ################ Starting simulation using MPC

    for k in range(0,int(SIM_TIME/dt)):

        print("--------------------------------------------------------")
        print("** at t="+str(k)+"T")

        x_init.value = vehicle_state[:,0]
        v_init.value = vehicle_state[:,1]
        F_init.value = vehicle_state[:,3]

        prob.solve(solver=ECOS, verbose=True)

        input = F_des.value[:,0]
        #input[i] = m[i]*u.value[i,0] + Ca*v.value[i,0]*v.value[i,0] +g*f

        print("u.value" + str(F_des.value[:,0])+", input : " + str(input))

        vehicle_state, x_, v_ =simulation(vehicle_state,input)

        print("predicted x[0][k+1]: " + str(x.value[0,1]) + ", actual x[0][k+1]: " + str(vehicle_state[0,0]))
        print("predicted v[0][k+1]: " + str(v.value[0,1]) + ", actual v[0][k+1]: " + str(vehicle_state[0,1]))

        #print("predicted x[0][last]: " + str(x.value[0, -1]) + ", actual x[0][k+1]: " + str(vehicle_state[0, 0]))
        #print("predicted v[0][last]: " + str(v.value[0, -1]) + ", actual x[0][k+1]: " + str(vehicle_state[0, 1]))

        x_res[:,50*k:(k+1)*50]=x_
        v_res[:,50*k:(k+1)*50]=v_




    plt.subplot(211)
    plt.plot(t,x_res[0,:],t,x_res[1,:],t,x_res[2,:],t,x_res[3,:],t,x_res[4,:],
             t,x_res[5,:],t,x_res[6,:],t,x_res[7,:],t,x_res[8,:])
    plt.legend()
    plt.subplot(212)
    plt.plot(t,v_res[0,:],t,v_res[1,:],t,v_res[2,:],t,v_res[3,:],t,v_res[4,:],
             t,v_res[5,:],t,v_res[6,:],t,v_res[7,:],t,v_res[8,:])
    plt.legend()
    plt.savefig('result.png',dpi=1200)