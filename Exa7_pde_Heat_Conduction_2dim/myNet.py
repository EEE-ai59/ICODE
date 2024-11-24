
import numpy as np
# from scipy.integrate import odeint as scipy_odeint
# import plotly.graph_objects as go
# from scipy.interpolate import interp2d
import torch
import torch.nn as nn
import torch.optim as optim
# from fastdtw import fastdtw
# from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from thop import profile
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter, uniform_filter
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error
# from skimage.metrics import structural_similarity as ssim # 计算图片相似度
from torchdiffeq import odeint, odeint_adjoint
import os
from Heat_Conduction_Equation import getData,input_dim,Tnum,M,r
hidden_dim = 200
num_layers = 3
aug_ini_hiddendim=60
u_dim = 1 ###

h=1/M
k=r*h**2
T=k*Tnum
tL_all= np.arange(0,T,k) 
tL_all=np.append(tL_all, T)
train_len=int(len(tL_all)/3)*2-2
test_len = int(train_len * 0.5) 
tL = tL_all[:train_len+test_len]

tSwitchPoint=T*np.array([0.06, 0.43, 0.78, 0.9])
tSwitchLen=T*0.01
# tSwitchPoint=[0.06, 0.3, 0.6]
# tSwitchLen=0.01
xVal=[0, 1]
def my_u(t, tSwitchPoint=tSwitchPoint, tSwitchLen=tSwitchLen, xVal=xVal):
    # 二值切换
    if t<tSwitchPoint[0]:
        return xVal[0]
    count=-1
    for i,it in enumerate(tSwitchPoint):
        if t<it:
            break
        else:
            count+=1
    if t>=tSwitchPoint[count]+tSwitchLen:
        temp=(count+1)%2
        return xVal[temp]
    
    t0=tSwitchPoint[count]
    k=-2/tSwitchLen**3 * (t-t0)**3 + 3/tSwitchLen**2 * (t-t0)**2
    x0=xVal[count%2]
    x1=xVal[(count+1)%2]
    res=x0+k*(x1-x0)
    return res

def my_du(t, tSwitchPoint=tSwitchPoint, tSwitchLen=tSwitchLen, xVal=xVal):
    # 二值切换
    if t<tSwitchPoint[0]:
        return 0
    count=-1
    for i,it in enumerate(tSwitchPoint):
        if t<it:
            break
        else:
            count+=1
    if t>=tSwitchPoint[count]+tSwitchLen:
        return 0
    
    t0=tSwitchPoint[count]
    k=-6/tSwitchLen**3 * (t-t0)**2 + 6/tSwitchLen**2 * (t-t0)
    x0=xVal[count%2]
    x1=xVal[(count+1)%2]
    res=x0+k*(x1-x0)
    return res


USE_BASIC_SOLVER = True
# if USE_BASIC_SOLVER = False, the code uses ode solver in torchdiffeq

train_mode = True
initial_mode = True # False

whole_seq = None
augment_dim = 10     
def basic_euler_ode_solver(func, y0, t, gu_update_func=None):
    dt = t[1] - t[0]
    y = y0
    ys = [y0]
    for i in range(len(t) - 1):
        t_start, t_end = t[i], t[i+1]


        y = y + func(t_start, y) * dt
        t_start += dt
        ys.append(y)
    return torch.stack(ys) 

# class TrainInf():
#     def __init__(self, num_epochs=num_epochs):        
#         self.loss=np.zeros((num_epochs,))
#         self.msePred=[]
#         self.msePredStep=[]
        
   


# class NeuralODE(nn.Module):
#     def __init__(self):
#         super(NeuralODE, self).__init__()
#         self.func = ODEFunc()
        
#     def forward(self, y0, t):
#         if USE_BASIC_SOLVER:
#             out = basic_euler_ode_solver(self.func, y0, t)
#         else:
#             out = odeint(self.func, y0, t)
#         # out = out.view(-1, t.shape[0], input_dim)
#         out = out.transpose(0,1)
#         out = torch.squeeze(out, dim=2)
#         return out

class AugmentedODEFunc(nn.Module):
    def __init__(self, hidden_dim=hidden_dim, num_layers=num_layers, this_input_dim=augment_dim+input_dim):
        super(AugmentedODEFunc, self).__init__()
        
        layers = [nn.Linear(this_input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, this_input_dim)) ###
        
        self.net = nn.Sequential(*layers)

    def forward(self, t, y):
        y = self.net(y)

        return y


class SecondOrderAugmentedODEFunc(nn.Module):
    def __init__(self, hidden_dim=hidden_dim, num_layers=num_layers):
        super(SecondOrderAugmentedODEFunc, self).__init__()
        
        layers = [nn.Linear(2*(input_dim + augment_dim), hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, input_dim + augment_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, t, z):
        cutoff = int(len(z)/2)
        y = z[:cutoff]
        v = z[cutoff:]
        t_vec = torch.ones(y.shape[0], 1).to(device) * t
        t_and_y_and_v = torch.cat([t_vec, y, v], 1)
        out = self.net(t_and_y_and_v)
        return torch.cat((v, out[:, :input_dim-1]))
    
class AugmentedNeuralODE(nn.Module):
    def __init__(self, augment_dim=augment_dim, use_second_order=False):
        super(AugmentedNeuralODE, self).__init__()
        self.use_second_order = use_second_order
        
        if use_second_order:
            self.func = SecondOrderAugmentedODEFunc(hidden_dim=int(hidden_dim))
        else:
            self.func = AugmentedODEFunc(hidden_dim=int(hidden_dim), this_input_dim=augment_dim+input_dim)
        
        self.augment_dim = augment_dim
        # layers = [nn.Linear(input_dim, aug_ini_hiddendim), nn.Tanh()]
        # layers.append(nn.Linear(aug_ini_hiddendim, augment_dim))
        layers = [nn.Linear(input_dim, augment_dim)]
        self.augment_net = nn.Sequential(*layers)


    def forward(self, y0, t):
        y_aug=0*self.augment_net(y0) ###
        y0=torch.cat((y0, y_aug), 2)
        if USE_BASIC_SOLVER:
            out = basic_euler_ode_solver(self.func, y0, t)
        else:
            out = odeint(self.func, y0, t)
            # out = solver.integrate(t)
        # out = out.view(-1, t.shape[0], input_dim)
        out = out.transpose(0,1)
        out = out[:,-1,:,:input_dim]
        return out

    def forward_seq(self, y0, t):
        y_aug=self.augment_net(y0)
        y0=torch.cat((y0, y_aug), 2)
        if self.use_second_order:
            v_aug = torch.zeros_like(y_aug)
            z0 = torch.cat((y_aug, v_aug), dim=1)
        else:
            z0 = y_aug
        
        if USE_BASIC_SOLVER:
            out = basic_euler_ode_solver(self.func, y0, t)
        else:
            out = odeint(self.func, y0, t)
        
        if self.use_second_order:
            out = out[:, :, :input_dim] ###

        out = out.transpose(0,1)
        out = out[:, :, :, :input_dim]
        
        out = torch.squeeze(out, dim=2)
        return out

class CSNeuralODE(nn.Module):
    def __init__(self,input_dim=input_dim, u=None, u_dim=input_dim):
        super(CSNeuralODE, self).__init__() ###

        if u is None:
            self.u=my_u
        else:
            self.u=u
        # self.Gfunc = [ODEFunc(hidden_dim=2, input_dim=1, output_dim=1), ODEFunc(hidden_dim=2, input_dim=1, output_dim=1), ODEFunc(hidden_dim=2, input_dim=1, output_dim=1)] # put u into the system
        ### change 6.19
        self.func_f = ODEFunc(this_hidden_dim=int(hidden_dim))
        self.func_g = ODEFunc(this_hidden_dim=int(hidden_dim/5)+1) ###

    def func(self, t, x):
        dx=self.func_f(t, x)+self.func_g(t,x)*self.u(t) ####
        return dx


    def forward(self, y0, t):
        if USE_BASIC_SOLVER:
            out = basic_euler_ode_solver(self.func, y0, t)
        else:
            out = odeint(self.func, y0, t)
            # out = solver.integrate(t)
        # out = out.view(-1, t.shape[0], input_dim)
        out = out.transpose(0,1)
        out = out[:,-1,:,:]
        return out
    
    def forward_seq(self, y0, t):
        if USE_BASIC_SOLVER:
            out = basic_euler_ode_solver(self.func, y0, t)
        else:
            out = odeint(self.func, y0, t)
            # out = solver.integrate(t)
        # out = out.view(-1, t.shape[0], input_dim)
        out = out.transpose(0,1)
        out = torch.squeeze(out, dim=2)
        return out
    
class CNeuralODE(nn.Module):
    def __init__(self,input_dim=input_dim, u=None, u_dim=input_dim):
        super(CNeuralODE, self).__init__() ###

        if u is None:
            self.u=my_du
        else:
            self.u=u
        # self.Gfunc = [ODEFunc(hidden_dim=2, input_dim=1, output_dim=1), ODEFunc(hidden_dim=2, input_dim=1, output_dim=1), ODEFunc(hidden_dim=2, input_dim=1, output_dim=1)] # put u into the system
        ### change 6.19
        self.func_f = ODEFunc(this_hidden_dim=int(hidden_dim))
        self.func_g = ODEFunc(this_hidden_dim=int(hidden_dim/5)+1) ###

    def func(self, t, x):
        dx=self.func_f(t, x)+self.func_g(t,x)*self.u(t) ####
        return dx


    def forward(self, y0, t):
        if USE_BASIC_SOLVER:
            out = basic_euler_ode_solver(self.func, y0, t)
        else:
            out = odeint(self.func, y0, t)
            # out = solver.integrate(t)
        # out = out.view(-1, t.shape[0], input_dim)
        out = out.transpose(0,1)
        out = out[:,-1,:,:]
        return out
    
    def forward_seq(self, y0, t):
        if USE_BASIC_SOLVER:
            out = basic_euler_ode_solver(self.func, y0, t)
        else:
            out = odeint(self.func, y0, t)
            # out = solver.integrate(t)
        # out = out.view(-1, t.shape[0], input_dim)
        out = out.transpose(0,1)
        out = torch.squeeze(out, dim=2)
        return out

class NeuralODE(nn.Module):
    def __init__(self):
        super(NeuralODE, self).__init__()
        self.func = ODEFunc()

    def forward(self, y0, t):
        if USE_BASIC_SOLVER:
            out = basic_euler_ode_solver(self.func, y0, t)
        else:
            out = odeint(self.func, y0, t)
            # out = solver.integrate(t)
        # out = out.view(-1, t.shape[0], input_dim)
        out = out.transpose(0,1)
        out = out[:,-1,:,:]
        return out
      
    def forward_seq(self, y0, t):
        if USE_BASIC_SOLVER:
            out = basic_euler_ode_solver(self.func, y0, t)
        else:
            out = odeint(self.func, y0, t)
        # out = out.view(-1, t.shape[0], input_dim)
        out = out.transpose(0,1)
        out = torch.squeeze(out, dim=2)
        return out
    
class simple_fc_layer(nn.Module):
    def __init__(self, dim=input_dim):
        super(simple_fc_layer, self).__init__()
        self.linear = nn.Linear(dim, dim)
        
    def forward(self, t, y):

        t_vec = torch.ones(y.shape[0], 1).to(device) * t
        t_and_y = torch.cat([t_vec, y], 1)
        y =  self.linear(t_and_y)[:, :input_dim-1]
        return y
    

    


class ODEFunc(nn.Module):
    def __init__(self, this_hidden_dim=hidden_dim, 
                 num_layers=num_layers, input_dim=input_dim, output_dim=input_dim
                 ):
        super(ODEFunc, self).__init__()
        
        layers = [nn.Linear(input_dim, this_hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(this_hidden_dim, this_hidden_dim))
            layers.append(nn.Softplus()) ###
        layers.append(nn.Linear(this_hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        # self.net.cuda()
        self.net.to(device)
        self.input_dim = input_dim
    

    def forward(self, t, y):
        y1 = self.net(y)
        return y1

