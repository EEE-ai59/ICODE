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
# from skimage.metrics import structural_similarity as ssim 
from torchdiffeq import odeint, odeint_adjoint
import os
import importlib.util
import sys
from config.param import *
from Heat_Conduction_Equation import getData,input_dim,Tnum,M,r
# from one_equ  import u

lr_decay=0.1
lr_min=5e-4
epoch_change=5






h=1/M
k=r*h**2
T=k*Tnum
tL_all= np.arange(0,T,k) 
tL_all=np.append(tL_all, T)
train_len=int(len(tL_all)/3)*1-2
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





global_min_vals = None
global_range_vals = None

# train_len = 50
# test_len = int(train_len * 0.5) # int(train_len * 0.5)

# tL_all = np.linspace(0,1,2*train_len)
# tL = tL_all[:train_len+test_len]
# L = 2.5
# num_time_points = test_len + train_len


u_dim = 1 ###


whole_seq = None


# def my_u(t):
#     return 0.5+(4*t)**2+2*t

# def my_du(t, tSwitchPoint=tSwitchPoint, tSwitchLen=tSwitchLen, xVal=xVal):
#     return (8*t)+2

    

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
        self.lossStep=[]
        
   
class MLP(nn.Module):
    def __init__(self, hidden_dim=hidden_dim, num_layers=num_layers, input_dim=input_dim, output_dim=input_dim):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.output_dim = output_dim
    # def forward(self, y0, t):
    #     ys = []
    #     y = y0
    #     ys.append(y)
    #     for _ in range(t.shape[0] - 1):
    #         y = self.net(y)
    #         ys.append(y)
    #     out = torch.stack(ys)
    #     return out.view(-1, t.shape[0], self.input_dim) ### ?
    def forward(self, y0):
        y = self.net(y0)        
        return y



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
            self.func = AugmentedODEFunc(hidden_dim=int(hidden_dim+aug_hidden_dim), this_input_dim=augment_dim+input_dim)
        
        self.augment_dim = augment_dim
        # layers = [nn.Linear(input_dim, aug_ini_hiddendim), nn.Tanh()]
        # layers.append(nn.Linear(aug_ini_hiddendim, augment_dim))
        layers = [nn.Linear(input_dim, augment_dim)]
        self.augment_net = nn.Sequential(*layers)


    def forward(self, y0, t):
        y_aug=self.augment_net(y0) ###
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
        self.func_g = ODEFunc(this_hidden_dim=int(hidden_dim/5)+1, num_layers=0) ###
        
        self.encoder=MLP(hidden_dim=zdim, num_layers=0, input_dim=input_dim, output_dim=zdim)
        self.decoder=MLP(hidden_dim=zdim, num_layers=0, input_dim=zdim, output_dim=input_dim)

    def func(self, t, x):
        dx=self.func_f(t, x)+self.func_g(t,x)*self.u(t) ####
        return dx


    def forward(self, y0, t):
        # y0=self.encoder(y0)
        if USE_BASIC_SOLVER:
            out = basic_euler_ode_solver(self.func, y0, t)
        else:
            out = odeint(self.func, y0, t)
            # out = solver.integrate(t)
        # out = out.view(-1, t.shape[0], input_dim)
        out = out.transpose(0,1)
        out = out[:,-1,:,:]
        # out=self.decoder(out)
        return out
    
    def forward_seq(self, y0, t):
        # y0=self.encoder(y0)
        if USE_BASIC_SOLVER:
            out = basic_euler_ode_solver(self.func, y0, t)
        else:
            out = odeint(self.func, y0, t)
            # out = solver.integrate(t)
        # out = out.view(-1, t.shape[0], input_dim)
        out = out.transpose(0,1)
        out = torch.squeeze(out, dim=2)
        # out=self.decoder(out)
        return out

class CSNeuralODE_MLP(nn.Module):
    def __init__(self,input_dim=input_dim, u=None, u_dim=input_dim):
        super(CSNeuralODE, self).__init__() ###

        if u is None:
            self.u=my_u
        else:
            self.u=u
        # self.Gfunc = [ODEFunc(hidden_dim=2, input_dim=1, output_dim=1), ODEFunc(hidden_dim=2, input_dim=1, output_dim=1), ODEFunc(hidden_dim=2, input_dim=1, output_dim=1)] # put u into the system
        ### change 6.19
        
        self.func_f = ODEFunc(input_dim=zdim, output_dim=zdim, this_hidden_dim=int(hidden_dim))
        self.func_g = ODEFunc(input_dim=zdim, output_dim=zdim, this_hidden_dim=int(hidden_dim/5)+1) ###
        
        self.encoder=MLP(hidden_dim=zdim, num_layers=0, input_dim=input_dim, output_dim=zdim)
        self.decoder=MLP(hidden_dim=zdim, num_layers=0, input_dim=zdim, output_dim=input_dim)

    def func(self, t, x):
        dx=self.func_f(t, x)+self.func_g(t,x)*self.u(t) ####
        return dx


    def forward(self, y0, t):
        y0=self.encoder(y0)
        if USE_BASIC_SOLVER:
            out = basic_euler_ode_solver(self.func, y0, t)
        else:
            out = odeint(self.func, y0, t)
            # out = solver.integrate(t)
        # out = out.view(-1, t.shape[0], input_dim)
        out = out.transpose(0,1)
        out = out[:,-1,:,:]
        out=self.decoder(out)
        return out
    
    def forward_seq(self, y0, t):
        y0=self.encoder(y0)
        if USE_BASIC_SOLVER:
            out = basic_euler_ode_solver(self.func, y0, t)
        else:
            out = odeint(self.func, y0, t)
            # out = solver.integrate(t)
        # out = out.view(-1, t.shape[0], input_dim)
        out = out.transpose(0,1)
        out = torch.squeeze(out, dim=2)
        out=self.decoder(out)
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





def train(model, model_name, modelInf):
    
    train_period=2
    train_jump=1

    ###
    if model_name == "aug_ode":
        lrList[0] = 5*lrList[0] # 16000
    # else:
    #     num_epochs = 2



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
        # modelInf.msePredStep.append(0)

    count_change=1
    numlr=num_epochs/len(lrList)

    for epoch in range(num_epochs):
        num_=int(epoch/numlr)
        if num_>=len(lrList):
            num_=len(lrList)-1
        optimizer.param_groups[0]['lr']=lrList[num_]
        model.train()   ###
        optimizer.zero_grad()

        inputs, targets = whole_seq[:,0,:].to(device),  whole_seq[:, :train_len, :].to(device)
        inputs = inputs.reshape(n_seq, -1, input_dim).to(device)
        targets = targets.reshape(n_seq, -1, input_dim).to(device)
        t=tL[0:train_len]
        pred = model.forward_seq(inputs, torch.from_numpy(t).to(device).detach()) 
        loss = criterion(pred, targets)
        loss.backward()
        modelInf.loss[epoch]=loss.item()
        modelInf.lossStep.append(epoch+1)



        # for time_k in range(0,train_len-1-train_period,train_jump):
        #     # if epoch == epoch_change*count_change:
        #     #     for parm in optimizer.param_groups:
        #     #         if parm['lr']> lr_min:
        #     #             parm['lr']=parm['lr']*lr_decay
        #     #             count_change*=2
        #     inputs, targets = whole_seq[:,time_k,:].to(device),  whole_seq[:,time_k+1+train_period,:].to(device)
        #     inputs = inputs.reshape(n_seq, -1, input_dim).to(device) 
        #     targets = targets.reshape(n_seq, -1, input_dim).to(device)
        #     t=tL[time_k:time_k+2+train_period] ##
        #     pred = model(inputs, torch.from_numpy(t).to(device).detach()) ### t
        #     loss = criterion(pred, targets)
        #     #optimizer.zero_grad()
        #     loss.backward()
        #     modelInf.loss[epoch]+=loss.item()

        optimizer.step()
        if (epoch + 1) % num_epochs_show == 0:
            print(model_name + f": Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}",end='') 
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
                print(" Test MSE: ", mse) 
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
    sol_fine = getData(n_seq)
    # sol_fine=sol_fine.reshape(n_seq, sol_fine.shape[0], sol_fine.shape[1]*sol_fine.shape[2])
    np.save(current_directory+"/saveModel/sol_fine_save.npy", sol_fine)
    whole_seq = sol_fine
    # np.save(current_directory+"/sol.npy", sol_fine)
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


criterion = CombinedLoss(weight_mse=1, weight_l1=0).to(device)




if train_mode:
    # cs_ode = torch.load(current_directory+"/cs_ode.pth").to(device)
    train(cs_ode, "cs_ode", cs_ode_Inf)
    torch.save(cs_ode, current_directory+"/saveModel/cs_ode.pth")
# cs_ode = torch.load(current_directory+"/cs_ode.pth").to(device)
Z_cs_ode = eval_model(cs_ode, "cs_ode")

if train_mode:
    # neural_ode = torch.load(current_directory+"/neural_ode.pth").to(device)
    train(c_ode, "c_ode", c_ode_Inf)
    torch.save(neural_ode, current_directory+"/saveModel/neural_ode.pth")
# neural_ode = torch.load(current_directory+"/neural_ode.pth").to(device)
Z_C_ode = eval_model(c_ode, "c_ode")

if train_mode:
    # neural_ode = torch.load(current_directory+"/neural_ode.pth").to(device)
    train(neural_ode, "neural_ode", neural_ode_Inf)
Z_neural_ode = eval_model(neural_ode, "neural_ode")

if train_mode:
    # aug_ode = torch.load(current_directory+"/neural_ode.pth").to(device)
    train(aug_ode, "aug_ode", aug_ode_Inf)
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


# X = np.linspace(0, L, N_fine)
# Y = np.linspace(0, L, N_fine)
Z1 = Z_C_ode
# Z2 = Z_aug_ode
Z5 = Z_cs_ode
# Z3 = Z_csode[:, :, 0] 
Zsta = sol_fine[:,0:train_len+test_len,:] 

Z_initial = sol_fine[:,train_len-1,:]


mse_Z1_Zsta=0
for i in range(test_len):
    mse_Z1_Zsta += mean_squared_error(Z1[:,i,:], sol_fine[:,train_len+i,:])

mse_Z5_Zsta=0
for i in range(test_len):
    mse_Z5_Zsta += mean_squared_error(Z5[:,i,:], sol_fine[:,train_len+i,:])

mse_Znode_Zsta=0
for i in range(test_len):
    mse_Znode_Zsta += mean_squared_error(Z_neural_ode[:,i,:], sol_fine[:,train_len+i,:])

mse_Zaug_Zsta=0
for i in range(test_len):
    mse_Zaug_Zsta += mean_squared_error(Z_aug_ode[:,i,:], sol_fine[:,train_len+i,:])


# mse_Z5_Zsta = mean_squared_error(Z5, Zsta)
# mse_Znode_Zsta = mean_squared_error(Z_neural_ode, Zsta)
# mse_Zaug_Zsta = mean_squared_error(Z_aug_ode, Zsta)


# mae_Z1_Zsta = mean_absolute_error(Z1, Zsta)
# mae_Z5_Zsta = mean_absolute_error(Z5, Zsta)
# mae_Z2_Zsta = mean_absolute_error(Z2, Zsta)


# max_err_Z1_Zsta = max_error(Z1.ravel(), Zsta.ravel())
# max_err_Z5_Zsta = max_error(Z5.ravel(), Zsta.ravel())
# # max_err_Z2_Zsta = max_error(Z2.ravel(), Zsta.ravel())




print("MSE between ControlSynth Neural ODE Prediction and GT:", mse_Z5_Zsta)
print("MSE between C ODE Prediction and GT:", mse_Z1_Zsta)
print("MSE between Neural ODE Prediction and GT:", mse_Znode_Zsta)
print("MSE between Augmented Neural ODE Prediction and GT:", mse_Zaug_Zsta)
# # print("MSE between Agumented Neural ODE Prediction and GT:", mse_Z2_Zsta)
# print()
# print("MAE between ControlSynth Neural ODE Prediction and GT:", mae_Z5_Zsta)
# print("MAE between Neural ODE Prediction and GT:", mae_Z1_Zsta)
# # print("MAE between Agumented Neural ODE Prediction and GT:", mae_Z2_Zsta)
# print()
# print("Max Error between ControlSynth Neural ODE Prediction and GT:", max_err_Z5_Zsta)
# print("Max Error between Neural ODE Prediction and GT:", max_err_Z1_Zsta)
# # print("Max Error between Agumented Neural ODE Prediction and GT:", max_err_Z2_Zsta)
if draw==True:
    plt.plot(cs_ode_Inf.msePredStep,cs_ode_Inf.msePred,label='cs_ode')
    plt.plot(c_ode_Inf.msePredStep,c_ode_Inf.msePred,label='c_ode_Inf')
    plt.plot(neural_ode_Inf.msePredStep,neural_ode_Inf.msePred,label='neural_ode_Inf')
    plt.plot(aug_ode_Inf.msePredStep,aug_ode_Inf.msePred,label='aug_ode_Inf')
    plt.legend()
    plt.show()

    plt.plot(cs_ode_Inf.loss,label='cs_ode')
    plt.plot(c_ode_Inf.loss,label='c_ode')
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



