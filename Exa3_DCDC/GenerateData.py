# import numpy as np
# import scipy as sp
# import matplotlib.pyplot as plt
# import pandas as pd
# import networkx as nx
# def sys(x, t, u, phi):
#     #phi=args[1]
#     tau1 = 1
#     tau2 = 1
#     A = np.array([0, 1,
#                   0, 0]).reshape(2, 2)
#     B = np.array([0, 1]).reshape(2, 1)
#     dx = A@x.reshape(2, 1)+B@np.array([u(t, phi)]).reshape(1, 1)+B@np.array([np.sin(x[0])]).reshape(1, 1)
#     return dx.flatten()

input_dim=3

import numpy as np
import matplotlib.pyplot as plt
from utils import *
from scipy.integrate import odeint
import torch
import os
import yaml
pwd=os.path.abspath(__file__)
par_pwd=os.path.dirname(pwd)
path = par_pwd+"/config/base.yaml" 
fig_path = par_pwd+'/figure'
with open(path, "r") as f:
    config = yaml.safe_load(f)

ndim=3

train_len = config['train']['train_len']
test_len = config['train']['test_len']
T_end=config['train']['T_end']
t = np.linspace(0,T_end,(train_len+test_len))
n_seq = config['train']['n_seq']



def dmyf(x,t,u):

    C1=0.1
    C2=0.2
    L3=0.5

    v1=x[0]
    v2=x[1]
    i3=x[2]

    d_v1=1/C1*(1-u(t))*i3
    d_v2=1/C2*u(t)*i3
    d_i3=1/L3*(-(1-u(t))*v1-u(t)*v2)

    return np.array([d_v1, d_v2, d_i3])




def getData(n_seq=n_seq, t=t, draw=True, u=my_u):
    y=np.zeros((n_seq, len(t), ndim))
    for i in range(n_seq):
        y0_=np.random.rand(ndim,)
        y[i]=odeint(dmyf, y0_, t, args=(u,))

    if draw:
        for i in range(n_seq):
            x=y[i,:,0]
            plt.plot(t,x)
        plt.show()

    return y











if  __name__ == '__main__':

    res=getData(u=my_u)
    plt.figure()
    for i in range(n_seq):
        x=res[i,:,0]
        plt.plot(t,x)
    fig_file=fig_path+'/tra/dim0.png'
    plt.savefig(fig_file)
    plt.show()
    
    plt.figure()
    for i in range(n_seq):
        x=res[i,:,1]
        plt.plot(t,x)
    fig_file=fig_path+'/tra/dim1.png'
    plt.savefig(fig_file)
    plt.show()



