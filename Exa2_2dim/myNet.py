import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from thop import profile
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, r2_score
from torchdiffeq import odeint, odeint_adjoint
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

# torch.sign(torch.sin(0.4*t))
T_end=config['train']['T_end']
tSwitchPoint=T_end*np.array(config['input']['tSwitchPoint'])
tSwitchLen=T_end*config['input']['tSwitchLen']
xVal=config['input']['xVal']

run_my=config['select_model']['my']
run_CDE=config['select_model']['CDE']
run_NeuralODE=config['select_model']['NeuralODE']
run_AugNeuralODE=config['select_model']['AugNeuralODE']

show_mse=config['select_metrics']['mse']
show_absErr=config['select_metrics']['absErr']
show_maxErr=config['select_metrics']['maxErr']


# if USE_BASIC_SOLVER = False, the code uses ode solver in torchdiffeq

train_mode = True
initial_mode = True # False


global_min_vals = None
global_range_vals = None

train_len = config['train']['train_len']
test_len = config['train']['test_len']
T_end=config['train']['T_end']
tL_all = np.linspace(0,T_end,(train_len+test_len))
tL = tL_all[:train_len+test_len]
num_time_points = test_len + train_len
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





def normalize_array(arr): ###
    min_vals = arr.min(axis=2)
    range_vals = arr.max(axis=2) - min_vals
    range_vals[range_vals == 0] = 1  # avoid deviding by 0
    norm_arr = (arr - min_vals) / range_vals
    return norm_arr, min_vals, range_vals

def unnormalize_array(norm_arr, min_vals, range_vals):
    original_arr = norm_arr * range_vals + min_vals
    return original_arr



        
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

# Transformer
# class ODEFunc(nn.Module):
#     def __init__(self, hidden_dim=hidden_dim, 
#                  num_layers=num_layers, input_dim=input_dim,
#                  d_model=512, nhead=4, transformer_num_layers=2):
#         super(ODEFunc, self).__init__()
        
#         layers = [nn.Linear(input_dim+1, hidden_dim), nn.Tanh()]
#         for _ in range(num_layers - 1):
#             layers.append(nn.Linear(hidden_dim, hidden_dim))
#             layers.append(nn.Softplus()) ###
#         layers.append(nn.Linear(hidden_dim, input_dim))
        
#         self.net = nn.Sequential(*layers)
#         self.input_dim = input_dim
#         self.transformer_layer = \
#             TransformerLayer(input_dim, d_model, nhead, transformer_num_layers)

#     def forward(self, t, y):
#         t_vec = torch.ones(y.shape[0], 1, 1).to(device) * t
#         t_and_y = torch.cat([t_vec, y], 2)
#         y1 = self.net(t_and_y)[:, :self.input_dim-1]
#         y2 = self.transformer_layer(t_and_y)
#         y2 = y2[:, :, :self.input_dim]
#         return y1+y2





class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=4, num_layers=2):
        super(Transformer, self).__init__()
        self.transformer_layer = \
            TransformerLayer(input_dim, d_model, nhead, num_layers)
    def forward(self, y0, t):
        pred_seq = y0.unsqueeze(0)
        pred_seq = torch.cat([pred_seq, t[0].view(1, -1, 1)], dim=2)
        for i in range(t.shape[0] - 1):
            output = self.transformer_layer(pred_seq)
            output = output[:, :, :-1]
            next_t = t[:i + 1].view(-1, 1, 1)
            output = torch.cat([output, next_t], dim=2)
            pred_seq = torch.cat([pred_seq, output[-1:]], dim=0)
        pred_seq = pred_seq[:,:,:input_dim-1].squeeze(1)
        return pred_seq.view(-1, t.shape[0], input_dim)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerLayer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers):
        super(TransformerLayer, self).__init__()
        self.embedding = nn.Linear(input_dim+1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, input_dim+1)

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output