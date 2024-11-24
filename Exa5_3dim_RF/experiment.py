import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from thop import profile
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error
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



whole_seq = None




    

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





# def normalize_array(arr): ###
#     min_vals = arr.min(axis=2)
#     range_vals = arr.max(axis=2) - min_vals
#     range_vals[range_vals == 0] = 1  # avoid deviding by 0
#     norm_arr = (arr - min_vals) / range_vals
#     return norm_arr, min_vals, range_vals

# def unnormalize_array(norm_arr, min_vals, range_vals):
#     original_arr = norm_arr * range_vals + min_vals
#     return original_arr



        
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





def train(model, model_name, modelInf):
    
    
    
    optimizer = optim.Adam(model.parameters(), lr=lrList[0])  

    print(model_name + f": initial time")
    with torch.no_grad():
        inputs, targets = whole_seq[:,train_len-1,:].to(device),  whole_seq[:,:train_len+test_len,:].to(device) ###
        inputs = inputs.reshape(n_seq, -1, input_dim)   
        t=tL_all[train_len-1:2*train_len]
        pred = model.forward_seq(inputs, torch.from_numpy(t).to(device).detach())
        Z = pred.cpu().detach().numpy().reshape(n_seq, -1, t.shape[0], input_dim)
        # Z = unnormalize_array(Z, global_min_vals, global_range_vals)
        mse=0
        for i in range(test_len):
            mse += mean_squared_error(Z[:,0,i,:], sol_fine[:,train_len+i,:])
        print(model_name + " Test MSE: ", mse, "\n")
        modelInf.msePred.append(mse)
        modelInf.msePredStep.append(0)

    count_change=1
    numlr=num_epochs/len(lrList)

    for epoch in range(num_epochs):

        num_=int(epoch/numlr)
        if num_>=len(lrList):
            num_=len(lrList)-1
        optimizer.param_groups[0]['lr']=lrList[num_]
        model.train()   ###
        optimizer.zero_grad()  
        for time_k in range(train_len-1):
            # if epoch == epoch_change*count_change:
            #     for parm in optimizer.param_groups:
            #         if parm['lr']> lr_min:
            #             parm['lr']=parm['lr']*lr_decay
            #             count_change*=2
            inputs, targets = whole_seq[:,time_k,:].to(device),  whole_seq[:,time_k+1,:].to(device)
            inputs = inputs.reshape(n_seq, -1, input_dim).to(device) 
            targets = targets.reshape(n_seq, -1, input_dim).to(device)
            t=tL[time_k:time_k+2] ##
            pred = model(inputs, torch.from_numpy(t).to(device).detach()) ### t
            loss = criterion(pred, targets)                      
            modelInf.loss[epoch]+=loss.item()
            loss.backward()
        optimizer.step()
        if (epoch + 1) % num_epochs_show == 0:
            if (epoch + 1) % 50*num_epochs_show == 0:
                print(model_name + f": Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}", end='') 
            model.eval() ###
            with torch.no_grad():
                inputs, targets = whole_seq[:,train_len-1,:].to(device),  whole_seq[:,:train_len+test_len,:].to(device) ###
                inputs = inputs.reshape(n_seq, -1, input_dim)   
                t=tL_all[train_len-1:2*train_len]
                pred = model.forward_seq(inputs, torch.from_numpy(t).to(device).detach())
                Z = pred.cpu().detach().numpy().reshape(n_seq, -1, t.shape[0], input_dim)
                # Z = unnormalize_array(Z, global_min_vals, global_range_vals)
                mse=0
                for i in range(test_len):
                    mse += mean_squared_error(Z[:,0,i,:], sol_fine[:,train_len+i,:])
                if (epoch + 1) % 50*num_epochs_show == 0:
                    print("     Test MSE: ", mse, "\n") 
                modelInf.msePred.append(mse)
                modelInf.msePredStep.append(epoch+1)

def eval_model(model, model_name):

    inputs, targets = whole_seq[:,train_len-1,:].to(device),  whole_seq[:,:train_len+test_len,:].to(device)
    inputs = inputs.reshape(n_seq, -1, input_dim)   
    t=tL_all[train_len-1:2*train_len]
    pred = model.forward_seq(inputs, torch.from_numpy(t).to(device).detach())
    Z = pred.cpu().detach().numpy().reshape(n_seq, -1, t.shape[0], input_dim)
    # Z = unnormalize_array(Z, global_min_vals, global_range_vals)
    flops, params = profile(model, inputs=(inputs, torch.from_numpy(t).to(device).detach()), verbose=False)
    print(model_name + " flops and params:", flops, params)
    return Z[:,0,0:test_len,:]

cs_ode = None
c_ode = None
neural_ode = None
aug_ode = None

if initial_mode:
    train_mode = True
    sol_fine = getData(n_seq, tL, False, my_u)

    whole_seq = sol_fine
    np.save(current_directory+"/saveModel/data/sol.npy", sol_fine)
    cs_ode = CSNeuralODE().to(device)
    c_ode = CNeuralODE().to(device)
    neural_ode = NeuralODE().to(device)
    aug_ode = AugmentedNeuralODE().to(device)


    cs_ode_Inf=TrainInf()
    c_ode_Inf=TrainInf()
    neural_ode_Inf=TrainInf()
    aug_ode_Inf=TrainInf()

    # aug_ode = AugmentedNeuralODE().to(device)
    # torch.save(cs_ode, current_directory+"/cs_ode.pth")
    # torch.save(neural_ode, current_directory+"/neural_ode.pth")
    # torch.save(aug_ode, current_directory+"/aug_ode.pth")

# sol_fine = np.load(current_directory + "/sol.npy")
print(sol_fine.shape)
# whole_seq, global_min_vals, global_range_vals = normalize_array(sol_fine[:,:train_len,:]) ###
whole_seq=sol_fine[:,:train_len,:]

whole_seq = torch.tensor(whole_seq, dtype=torch.float32)   


class CombinedLoss(nn.Module):
    def __init__(self, weight_mse=0.5, weight_l1=0.5):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.weight_mse = weight_mse
        self.weight_l1 = weight_l1

    def forward(self, output, target):
        loss_mse = self.mse_loss(output, target)
        loss_l1 = self.l1_loss(output, target)
        combined_loss = self.weight_mse * loss_mse + self.weight_l1 * loss_l1
        return combined_loss


criterion = CombinedLoss(weight_mse=1, weight_l1=0.5).to(device)



#------ 本没有注释的 ----------
if run_my:
    if train_mode:
        # cs_ode = torch.load(current_directory+"/cs_ode.pth").to(device)
        train(cs_ode, "cs_ode", cs_ode_Inf)
        torch.save(cs_ode, current_directory+"/saveModel/cs_ode.pth")
    # cs_ode = torch.load(current_directory+"/cs_ode.pth").to(device)
    Z_cs_ode = eval_model(cs_ode, "cs_ode")

if run_CDE:
    if train_mode: 
        # neural_ode = torch.load(current_directory+"/neural_ode.pth").to(device)
        train(c_ode, "c_ode", c_ode_Inf)
        torch.save(c_ode, current_directory+"/saveModel/c_ode.pth")
    # neural_ode = torch.load(current_directory+"/neural_ode.pth").to(device)
    Z_C_ode = eval_model(c_ode, "c_ode")

if run_NeuralODE:
    if train_mode:
        # neural_ode = torch.load(current_directory+"/neural_ode.pth").to(device)
        train(neural_ode, "neural_ode", neural_ode_Inf)
        torch.save(neural_ode, current_directory+"/saveModel/neural_ode.pth")
    Z_neural_ode = eval_model(neural_ode, "neural_ode")

if run_AugNeuralODE:
    if train_mode:
        # aug_ode = torch.load(current_directory+"/neural_ode.pth").to(device)
        train(aug_ode, "aug_ode", aug_ode_Inf)
        torch.save(aug_ode, current_directory+"/saveModel/aug_ode.pth")
    Z_aug_ode = eval_model(aug_ode, "aug_ode")

# if train_mode:
#     aug_ode = torch.load(current_directory+"/aug_ode.pth").to(device)
#     train(aug_ode, "aug_ode")
#     torch.save(aug_ode, current_directory+"/aug_ode.pth")
# augo_de = torch.load(current_directory+"/aug_ode.pth").to(device)
# Z_aug_ode = eval_model(aug_ode, "aug_ode")


def chamfer_distance(set1, set2):
    dist1 = np.sqrt(((set1[:, np.newaxis, :] - set2[np.newaxis, :, :]) ** 2).sum(axis=2))
    dist2 = np.sqrt(((set2[:, np.newaxis, :] - set1[np.newaxis, :, :]) ** 2).sum(axis=2))  
    nearest_dist1 = np.min(dist1, axis=1)
    nearest_dist2 = np.min(dist2, axis=0)
    return np.mean(nearest_dist1) + np.mean(nearest_dist2)




Zsta = sol_fine[:,0:train_len+test_len,:] 

# Z_initial = sol_fine[:,train_len-1,:]

if run_my:
    mse_cs_ode=0
    absErr_cs_ode=0
    maxErr_cs_ode=0
    for i in range(test_len):
        mse_cs_ode += mean_squared_error(Z_cs_ode[:,i,:], sol_fine[:,train_len+i,:])
        absErr_cs_ode += mean_absolute_error(Z_cs_ode[:,i,:], sol_fine[:,train_len+i,:])
        maxErr_cs_ode += max_error(Z_cs_ode[:,i,:].ravel(), sol_fine[:,train_len+i,:].ravel())
    cs_ode_Inf.finalPredErr['mse']=mse_cs_ode
    cs_ode_Inf.finalPredErr['absErr']=absErr_cs_ode
    cs_ode_Inf.finalPredErr['maxErr']=maxErr_cs_ode

if run_CDE:
    mse_C_ode=0
    absErr_C_ode=0
    maxErr_C_ode=0
    for i in range(test_len):
        mse_C_ode += mean_squared_error(Z_C_ode[:,i,:], sol_fine[:,train_len+i,:])
        absErr_C_ode += mean_absolute_error(Z_C_ode[:,i,:], sol_fine[:,train_len+i,:])
        maxErr_C_ode += max_error(Z_C_ode[:,i,:].ravel(), sol_fine[:,train_len+i,:].ravel())
    c_ode_Inf.finalPredErr['mse']=mse_C_ode
    c_ode_Inf.finalPredErr['absErr']=absErr_C_ode
    c_ode_Inf.finalPredErr['maxErr']=maxErr_C_ode

if run_NeuralODE:
    mse_n_ode=0
    absErr_n_ode=0
    maxErr_n_ode=0
    for i in range(test_len):
        mse_n_ode += mean_squared_error(Z_neural_ode[:,i,:], sol_fine[:,train_len+i,:])
        absErr_n_ode += mean_absolute_error(Z_neural_ode[:,i,:], sol_fine[:,train_len+i,:])
        maxErr_n_ode += max_error(Z_neural_ode[:,i,:].ravel(), sol_fine[:,train_len+i,:].ravel())
    neural_ode_Inf.finalPredErr['mse']=mse_n_ode
    neural_ode_Inf.finalPredErr['absErr']=absErr_n_ode
    neural_ode_Inf.finalPredErr['maxErr']=maxErr_n_ode

if run_AugNeuralODE:
    mse_aug_ode=0
    absErr_aug_ode=0
    maxErr_aug_ode=0
    for i in range(test_len):
        mse_aug_ode += mean_squared_error(Z_aug_ode[:,i,:], sol_fine[:,train_len+i,:])
        absErr_aug_ode += mean_absolute_error(Z_aug_ode[:,i,:], sol_fine[:,train_len+i,:])
        maxErr_aug_ode += max_error(Z_aug_ode[:,i,:].ravel(), sol_fine[:,train_len+i,:].ravel())
    aug_ode_Inf.finalPredErr['mse']=mse_aug_ode
    aug_ode_Inf.finalPredErr['absErr']=absErr_aug_ode
    aug_ode_Inf.finalPredErr['maxErr']=maxErr_aug_ode


if show_mse:
    print('')
    if run_my:
        print("MSE between Input Concomitant ODE Prediction and GT:", mse_cs_ode)
    if run_CDE:
        print("MSE between CDE Prediction and GT:", mse_C_ode)
        
    if run_NeuralODE:
        print("MSE between Neural ODE Prediction and GT:", mse_n_ode)

    if run_AugNeuralODE:
        print("MSE between Augmented Neural ODE Prediction and GT:", mse_aug_ode)

if show_absErr:
    print('')
    if run_my:
        print("MAE between Input Concomitant ODE Prediction and GT:", absErr_cs_ode)
    if run_CDE:
        print("MAE between CDE Prediction and GT:", absErr_C_ode)
        
    if run_NeuralODE:
        print("MAE between Neural ODE Prediction and GT:", absErr_n_ode)

    if run_AugNeuralODE:
        print("MAE between Augmented Neural ODE Prediction and GT:", absErr_aug_ode)

if show_maxErr:
    print('')
    if run_my:
        print("Max Error between Input Concomitant ODE Prediction and GT:", maxErr_cs_ode)
    if run_CDE:
        print("Max Error between CDE Prediction and GT:", maxErr_C_ode)
        
    if run_NeuralODE:
        print("Max Error between Neural ODE Prediction and GT:", maxErr_n_ode)

    if run_AugNeuralODE:
        print("Max Error between Augmented Neural ODE Prediction and GT:", maxErr_aug_ode)

if draw==True:
    plt.plot(cs_ode_Inf.msePredStep,cs_ode_Inf.msePred,label='cs_ode')
    # plt.plot(c_ode_Inf.msePredStep,c_ode_Inf.msePred,label='c_ode_Inf')
    plt.plot(neural_ode_Inf.msePredStep,neural_ode_Inf.msePred,label='neural_ode_Inf')
    plt.plot(aug_ode_Inf.msePredStep,aug_ode_Inf.msePred,label='aug_ode_Inf')
    plt.legend()
    plt.show()

    plt.plot(cs_ode_Inf.loss,label='cs_ode')
    # plt.plot(c_ode_Inf.loss,label='c_ode')
    plt.plot(neural_ode_Inf.loss,label='neural_ode')
    plt.plot(aug_ode_Inf.loss,label='aug_ode')
    plt.legend()
    plt.show()



import os
import shutil

current_path = os.path.dirname(os.path.abspath(__file__))

def count_files_in_directory(directory):
    files = os.listdir(directory)
    return len(files)

directory_path = current_path+'/experimentData'

if not os.path.exists(directory_path):
    os.makedirs(directory_path)

file_count = count_files_in_directory(directory_path)
# file_count+=1

import pickle

folder_path=current_path+'/experimentData/num'+str(file_count)
os.mkdir(folder_path)

filename=folder_path+'/cs_ode_Inf.pkl'
with open(filename, 'wb') as file:
    pickle.dump(cs_ode_Inf, file)

filename=folder_path+'/c_ode_Inf.pkl'
with open(filename, 'wb') as file:
    pickle.dump(c_ode_Inf, file)

filename=folder_path+'/neural_ode_Inf.pkl'
with open(filename, 'wb') as file:
    pickle.dump(neural_ode_Inf, file)

filename=folder_path+'/aug_ode_Inf.pkl'
with open(filename, 'wb') as file:
    pickle.dump(aug_ode_Inf, file)
 

# with open('my_class.pkl', 'rb') as file:
#     loaded_instance = pickle.load(file)



