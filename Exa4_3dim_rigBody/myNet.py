import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import yaml
import importlib.util
import sys
from GenerateData import getData,input_dim 
from utils import u_dim,my_du,my_u
pwd=os.path.abspath(__file__)
par_pwd=os.path.dirname(pwd)
path = par_pwd+"/config/base.yaml" 
with open(path, "r") as f:
    config = yaml.safe_load(f)

draw=False
num_epochs = config['train']['num_epochs']
num_epochs_show= config['train']['num_epochs_show']
lrList = config['train']['lrList']
n_seq = config['train']['n_seq']
USE_BASIC_SOLVER = config['train']['USE_BASIC_SOLVER']

hidden_dim = config['model']['hidden_dim']
num_layers = config['model']['num_layers']
augment_dim=config['model']['augment_dim']
aug_ini_hiddendim=config['model']['aug_ini_hiddendim']
# 把u的信息融合到neuralODE中
# torch.sign(torch.sin(0.4*t))
tSwitchPoint=config['input']['tSwitchPoint']
tSwitchLen=config['input']['tSwitchLen']
xVal=config['input']['xVal']

run_my=config['select_model']['my']
run_CDE=config['select_model']['CDE']
run_NeuralODE=config['select_model']['NeuralODE']
run_AugNeuralODE=config['select_model']['AugNeuralODE']

show_mse=config['select_metrics']['mse']
show_absErr=config['select_metrics']['absErr']
show_maxErr=config['select_metrics']['maxErr']

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


class TrainInf():
    def __init__(self, num_epochs=num_epochs):        
        self.loss=np.zeros((num_epochs,))
        self.msePred=[]
        self.msePredStep=[]
        self.finalPredErr={}
        
   


class NeuralODE(nn.Module):
    def __init__(self):
        super(NeuralODE, self).__init__()
        self.func = ODEFunc()
        
    def forward(self, y0, t):
        if USE_BASIC_SOLVER:
            out = basic_euler_ode_solver(self.func, y0, t)
        else:
            out = odeint(self.func, y0, t)
        # out = out.view(-1, t.shape[0], input_dim)
        out = out.transpose(0,1)
        out = torch.squeeze(out, dim=2)
        return out

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
        layers = [nn.Linear(input_dim, aug_ini_hiddendim), nn.Tanh()]
        layers.append(nn.Linear(aug_ini_hiddendim, augment_dim))
        self.augment_net = nn.Sequential(*layers)


    def forward(self, y0, t):
        y_aug=self.augment_net(y0)
        y0=torch.cat((y0, y_aug), 2)
        if USE_BASIC_SOLVER:
            out = basic_euler_ode_solver(self.func, y0, t)
        else:
            out = odeint(self.func, y0, t)
            # out = solver.integrate(t)
        # out = out.view(-1, t.shape[0], input_dim)
        out = out[-1,:,:,:]
        out = out[:,:,:input_dim]
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
        
        out = out[:, :, :, :input_dim]
        
        # out = out.view(-1, t.shape[0], input_dim) ###
        out = out.transpose(0,1)
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
        self.func_f = ODEFunc(hidden_dim=int(hidden_dim))
        self.func_g = ODEFunc(hidden_dim=int(hidden_dim)) ###

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
        out = out[-1,:,:,:]
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
        self.func_f = ODEFunc(hidden_dim=int(hidden_dim))
        self.func_g = ODEFunc(hidden_dim=int(hidden_dim)) ###

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
        out = out[-1,:,:,:]
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
        out = out[-1,:,:,:]
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
    def __init__(self, hidden_dim=hidden_dim, 
                 num_layers=num_layers, input_dim=input_dim, output_dim=input_dim
                 ):
        super(ODEFunc, self).__init__()
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Softplus()) ###
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        # self.net.cuda()
        self.net.to(device)
        self.input_dim = input_dim
    

    def forward(self, t, y):
        y1 = self.net(y)
        return y1
