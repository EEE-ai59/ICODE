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

input_dim=6

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

ndim=6

train_len = config['train']['train_len']
test_len = config['train']['test_len']
T_end=config['train']['T_end']
t = np.linspace(0,T_end,(train_len+test_len))
n_seq = config['train']['n_seq']



alpha1=0.077884314
theta14=0.66
theta16=1
beta1=1.06270825
mu11=1.53
mu12=-0.59
mu17=1
alpha2=0.585012402
theta21=0.95
theta22=-0.41
theta25=0.32
theta27=0.62
theta210=0.38
beta2=0.0007934561
alpha3=0.0007934561
mu22=3.97
theta32=3.97
mu23=-3.06
theta33=-3.06
mu28=1
theta38=1
beta3=1.05880847
mu33=0.3
mu39=1
x4=10
x5=5
x6=3
x7=40
x8=136
x9=2.86
x10=4
a1=alpha1*x6**theta16
a2=alpha2*x7**theta27*x10**theta210
a3=alpha3
b1=beta1*x7**mu17
b2=beta2
b3=beta3*x9**mu39




def dmyf(x, t, u):
    x1=x[0]
    x2=x[1]
    x3=x[2]
    f1=a1*x4**theta14 - b1*x1**mu11*x2**mu12
    f2=a2*x1**theta21*x2**theta22*x5**theta25 - b2*x2**mu22*x3**mu23*x8**mu28
    f3=a3*x2**theta32*x3**theta33*x8**theta38-b3*x3**mu33

    f=np.zeros((6,))
    f[0]=f1
    f[1]=f2
    f[2]=f3

    g=[[0, 0, 0, 1, 0, 0],
       [0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 1]]
    g=np.array(g)
    g=g.T

    # h=[[1, 0, 0, 0, 0, 0],
    #    [0, 1, 0, 0, 0, 0],
    #    [0, 0, 1, 0, 0, 0]]
    # h=np.array(h)
    # h=h.T

    ###
    u1=u(t)
    u2=u(t)
    u3=u(t)
    u=np.array([u1, u2, u3]).reshape(3,1)
    dx=f+(g@u).flatten()

    return dx



def getData(n_seq=n_seq, t=t, draw=True, u=my_u, my_y0=None):

    if my_y0 is not None:
        y=np.zeros((my_y0.shape[0], len(t), ndim))
        for i in range(my_y0.shape[0]):
            y0_=my_y0
            y[i]=odeint(dmyf, y0_, t, args=(u,))
        return y

    y=np.zeros((n_seq, len(t), ndim))
    np.random.seed(2)
    for i in range(n_seq):
        y0_=10*np.random.rand(ndim,)+0.1 ###
        y[i]=odeint(dmyf, y0_, t, args=(u,))

    # if draw:
    #     plt.figure()
    #     for i in range(n_seq):
    #         x=y[i,:,0]
    #         plt.plot(t,x)
    #     plt.show()

    #     plt.figure()
    #     for i in range(n_seq):
    #         x=y[i,:,1]
    #         plt.plot(t,x)
    #     plt.show()

    return y











if  __name__ == '__main__':

    res=getData(u=my_u)

    # plt.figure()
    # for i in range(n_seq):
    #     x=res[i,:,0]
    #     plt.plot(t,x)
    # fig_file=fig_path+'/tra/dim0.png'
    # plt.savefig(fig_file)
    # plt.show()
    
    # plt.figure()
    # for i in range(n_seq):
    #     x=res[i,:,1]
    #     plt.plot(t,x)
    # fig_file=fig_path+'/tra/dim1.png'
    # plt.savefig(fig_file)
    # plt.show()

    fig = plt.figure()
    ax = fig.add_axes(Axes3D(fig)) 
    for i in range(n_seq):
        x=res[i,:,0]
        y=res[i,:,1]
        z=res[i,:,2]
        ax.plot(x,y,z,label=str(i))
    fig_file=fig_path+'/tra/dim6.png'
    plt.legend()
    # plt.savefig(fig_file)
    plt.show()



