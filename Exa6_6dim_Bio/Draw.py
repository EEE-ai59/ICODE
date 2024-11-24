import numpy as np
import os
import pickle
import matplotlib
# matplotlib.use('CMYK') 
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
import yaml
from utils import TrainInf, ZoomIn

fontsize_legend=12
fontsize_xylabel=15
# font = {'family' : 'Times New Roman',
# 'weight' : 'normal',
# 'size'   : 15,
#         }
def cmyk_to_rgb(c, m, y, k):
    r = 1 - min(1, c * (1 - k) + k)
    g = 1 - min(1, m * (1 - k) + k)
    b = 1 - min(1, y * (1 - k) + k)
    return r, g, b

# Example CMYK color (C, M, Y, K)
cmyk_color_cs = (1, 1, 0, 0)
rgb_color_cs = cmyk_to_rgb(*cmyk_color_cs)
cmyk_color_c = (0, 0.4, 0.6, 0)
rgb_color_c = cmyk_to_rgb(*cmyk_color_c)
cmyk_color_n = (0, 1, 1, 0.45)
rgb_color_n = cmyk_to_rgb(*cmyk_color_n)
cmyk_color_aug = (1, 0, 1, 0.25)
rgb_color_aug = cmyk_to_rgb(*cmyk_color_aug)

pwd=os.path.abspath(__file__)
par_pwd=os.path.dirname(pwd)
path = par_pwd+"/config/base.yaml" 
with open(path, "r") as f:
    config = yaml.safe_load(f)

num_epochs = config['train']['num_epochs']
show_mse=config['select_metrics']['mse']
show_absErr=config['select_metrics']['absErr']
show_maxErr=config['select_metrics']['maxErr']
show_R2=config['select_metrics']['R2']
run_my=config['select_model']['my']
run_CDE=config['select_model']['CDE']
run_NeuralODE=config['select_model']['NeuralODE']
run_AugNeuralODE=config['select_model']['AugNeuralODE']
draw_len=num_epochs
draw_alpha=0.1


def count_files_in_directory(directory):
    files = os.listdir(directory)
    return len(files)

current_path = os.path.dirname(os.path.abspath(__file__))
directory_path = current_path+'/experimentData'
file_count = count_files_in_directory(directory_path)


# file_count=1


# with open(filename, 'rb') as file:
#     loaded_instance = pickle.load(file)
# file_count=int(file_count)
i=0
folder_path=current_path+'/experimentData/num'+str(i)
if run_my:
    filename=folder_path+'/cs_ode_Inf.pkl'
elif run_CDE:
    filename=folder_path+'/c_ode_Inf.pkl'
elif run_NeuralODE:
    filename=folder_path+'/neural_ode_Inf.pkl'
elif run_AugNeuralODE:
    filename=folder_path+'/aug_ode_Inf.pkl'
else:
    print('No Model !!')
    exit(0)


with open(filename, 'rb') as file:
    loaded_instance = pickle.load(file)
    len_loss=len(loaded_instance.loss)
    msePredStep=loaded_instance.msePredStep
    len_msePred=len(loaded_instance.msePred)


Loss_cs_ode=np.zeros((file_count,len_loss))
Loss_c_ode=np.zeros((file_count,len_loss))
Loss_n_ode=np.zeros((file_count,len_loss))
Loss_aug_ode=np.zeros((file_count,len_loss))


msePred_cs_ode=np.zeros((file_count,len_msePred))
msePred_c_ode=np.zeros((file_count,len_msePred))
msePred_n_ode=np.zeros((file_count,len_msePred))
msePred_aug_ode=np.zeros((file_count,len_msePred))

absErr_cs_ode=np.zeros((file_count,))
maxErr_cs_ode=0
R2_cs_ode=0
absErr_c_ode=np.zeros((file_count,))
maxErr_c_ode=0
R2_c_ode=0
absErr_n_ode=np.zeros((file_count,))
maxErr_n_ode=0
R2_n_ode=0
absErr_aug_ode=np.zeros((file_count,))
maxErr_aug_ode=0
R2_aug_ode=0

for i in range(0,file_count):
    folder_path=current_path+'/experimentData/num'+str(i)

    if run_my:
        filename=folder_path+'/cs_ode_Inf.pkl'
        with open(filename, 'rb') as file:
            loaded_instance = pickle.load(file)
            Loss_cs_ode[i]=loaded_instance.loss 
            msePred_cs_ode[i]=loaded_instance.msePred
            absErr_cs_ode[i]=loaded_instance.finalPredErr['absErr']
            maxErr_cs_ode+=loaded_instance.finalPredErr['maxErr']
            R2_cs_ode+=loaded_instance.finalPredErr['R2']

    if run_CDE:
        filename=folder_path+'/c_ode_Inf.pkl'
        with open(filename, 'rb') as file:
            loaded_instance = pickle.load(file)
            Loss_c_ode[i]=loaded_instance.loss ###
            msePred_c_ode[i]=loaded_instance.msePred
            absErr_c_ode[i]=loaded_instance.finalPredErr['absErr']
            maxErr_c_ode+=loaded_instance.finalPredErr['maxErr']
            R2_c_ode+=loaded_instance.finalPredErr['R2']

    if run_NeuralODE:
        filename=folder_path+'/neural_ode_Inf.pkl'
        with open(filename, 'rb') as file:
            loaded_instance = pickle.load(file)
            Loss_n_ode[i]=loaded_instance.loss 
            msePred_n_ode[i]=loaded_instance.msePred
            absErr_n_ode[i]=loaded_instance.finalPredErr['absErr']
            maxErr_n_ode+=loaded_instance.finalPredErr['maxErr']
            R2_n_ode+=loaded_instance.finalPredErr['R2']

    if run_AugNeuralODE:
        filename=folder_path+'/aug_ode_Inf.pkl'
        with open(filename, 'rb') as file:
            loaded_instance = pickle.load(file)
            Loss_aug_ode[i]=loaded_instance.loss ###
            msePred_aug_ode[i]=loaded_instance.msePred
            absErr_aug_ode[i]=loaded_instance.finalPredErr['absErr']
            maxErr_aug_ode+=loaded_instance.finalPredErr['maxErr']
            R2_aug_ode+=loaded_instance.finalPredErr['R2']

# absErr_cs_ode/=file_count
maxErr_cs_ode/=file_count
R2_cs_ode/=file_count
# absErr_c_ode/=file_count
maxErr_c_ode/=file_count
R2_c_ode/=file_count
# absErr_n_ode/=file_count
maxErr_n_ode/=file_count
R2_n_ode/=file_count
# absErr_aug_ode/=file_count
maxErr_aug_ode/=file_count
R2_aug_ode/=file_count
# plt.plot(cs_ode_Inf.loss,label='cs_ode')
# plt.plot(Loss_n_ode[:,0],label='n_ode_Inf')
# plt.legend()
# plt.show()

# for i in range(0,file_count):
#     plt.plot(Loss_n_ode[i],label='n_ode_Inf')

# plt.show()
if run_my:
    Loss_cs_ode_min=Loss_cs_ode.min(axis=0)
    Loss_cs_ode_max=Loss_cs_ode.max(axis=0)
    Loss_cs_mean=Loss_cs_ode.mean(axis=0)
    Loss_cs_std=Loss_cs_ode.std(axis=0)
    msePred_cs_ode_min=msePred_cs_ode.min(axis=0)
    msePred_cs_ode_max=msePred_cs_ode.max(axis=0)
    msePred_cs_mean=msePred_cs_ode.mean(axis=0)
    msePred_cs_std=msePred_cs_ode.std(axis=0)

if run_CDE:
    Loss_c_ode_min=Loss_c_ode.min(axis=0)
    Loss_c_ode_max=Loss_c_ode.max(axis=0)
    Loss_c_mean=Loss_c_ode.mean(axis=0)
    Loss_c_std=Loss_c_ode.std(axis=0)
    msePred_c_ode_min=msePred_c_ode.min(axis=0)
    msePred_c_ode_max=msePred_c_ode.max(axis=0)
    msePred_c_mean=msePred_c_ode.mean(axis=0)
    msePred_c_std=msePred_c_ode.std(axis=0)

if run_NeuralODE:
    Loss_n_ode_min=Loss_n_ode.min(axis=0)
    Loss_n_ode_max=Loss_n_ode.max(axis=0)
    Loss_n_mean=Loss_n_ode.mean(axis=0)
    Loss_n_std=Loss_n_ode.std(axis=0)
    msePred_n_ode_min=msePred_n_ode.min(axis=0)
    msePred_n_ode_max=msePred_n_ode.max(axis=0)
    msePred_n_mean=msePred_n_ode.mean(axis=0)
    msePred_n_std=msePred_n_ode.std(axis=0)

if run_AugNeuralODE:
    Loss_aug_ode_min=Loss_aug_ode.min(axis=0)
    Loss_aug_ode_max=Loss_aug_ode.max(axis=0)
    Loss_aug_mean=Loss_aug_ode.mean(axis=0)
    Loss_aug_std=Loss_aug_ode.std(axis=0)
    msePred_aug_ode_min=msePred_aug_ode.min(axis=0)
    msePred_aug_ode_max=msePred_aug_ode.max(axis=0)
    msePred_aug_mean=msePred_aug_ode.mean(axis=0)
    msePred_aug_std=msePred_aug_ode.std(axis=0)









fig_path = current_path+'/figure'

fig, ax = plt.subplots(1, 1)
# plt.fill_between(msePredStep[:draw_len], msePred_cs_ode_min[:draw_len], msePred_cs_ode_max[:draw_len], facecolor='blue', alpha=draw_alpha)
# # plt.fill_between(msePredStep[:draw_len], msePred_c_ode_min[:draw_len], msePred_c_ode_max[:draw_len], facecolor='c', alpha=draw_alpha)
# plt.fill_between(msePredStep[:draw_len], msePred_n_ode_min[:draw_len], msePred_n_ode_max[:draw_len], facecolor='red', alpha=draw_alpha)
# plt.fill_between(msePredStep[:draw_len], msePred_aug_ode_min[:draw_len], msePred_aug_ode_max[:draw_len], facecolor='g', alpha=draw_alpha)
if run_my:
    ax.plot(msePredStep[:draw_len], msePred_cs_mean[:draw_len], color=rgb_color_cs, label='ICODE', linewidth=3, marker='*', markevery=int(num_epochs/5))
if run_CDE:
    ax.plot(msePredStep[:draw_len], msePred_c_mean[:draw_len], color=rgb_color_c, label='CDE', marker='.', markevery=int(num_epochs/5))
if run_NeuralODE:
    ax.plot(msePredStep[:draw_len], msePred_n_mean[:draw_len], color=rgb_color_n, label='NODE', marker='+', markevery=int(num_epochs/5))
if run_AugNeuralODE:
    ax.plot(msePredStep[:draw_len], msePred_aug_mean[:draw_len], color=rgb_color_aug, label='ANODE', marker='x', markevery=int(num_epochs/5))
ax.legend(fontsize=fontsize_legend)
# plt.gca().set_facecolor('#f0f0f0')
ax.grid(color='w')
ax.set_xlim([0,draw_len-1])
# ax.set_ylim([0, 1])
# plt.ylim([0,2]) 
ax.set_xlabel('Epoch', fontsize=fontsize_xylabel)
ax.set_ylabel('Prediction error', fontsize=fontsize_xylabel)
width=0.3
height=0.2
pos=[0.6, 0.3]
y_ratio=200
# if run_my and run_CDE and run_NeuralODE and run_AugNeuralODE:
#     ZoomIn(ax, msePredStep, msePred_cs_mean, msePred_c_mean, msePred_n_mean, msePred_aug_mean, width=width, height=height, pos=pos, draw_len=num_epochs, y_ratio = y_ratio)
    
fig_file=fig_path+'/slrobot-testError.png'
plt.savefig(fig_file, dpi=300)
plt.show()


fig, ax = plt.subplots(1, 1)
# plt.fill_between(msePredStep[:draw_len],Loss_cs_ode_min[:draw_len], Loss_cs_ode_max[:draw_len], facecolor='blue', alpha=draw_alpha)
# plt.fill_between(msePredStep[:draw_len], Loss_c_ode_min[:draw_len], Loss_c_ode_max[:draw_len], facecolor='c', alpha=draw_alpha)
# plt.fill_between(msePredStep[:draw_len],Loss_n_ode_min[:draw_len], Loss_n_ode_max[:draw_len], facecolor='red', alpha=draw_alpha)
# plt.fill_between(msePredStep[:draw_len], Loss_aug_ode_min[:draw_len], Loss_aug_ode_max[:draw_len], facecolor='g', alpha=draw_alpha)
if run_my:
    ax.plot(msePredStep[:draw_len], Loss_cs_mean[:draw_len], color=rgb_color_cs, label='ICODE', linewidth=3, marker='*', markevery=int(num_epochs/5))
if run_CDE:
    ax.plot(msePredStep[:draw_len], Loss_c_mean[:draw_len], color=rgb_color_c, label='CDE', marker='s', markevery=int(num_epochs/5))
if run_NeuralODE:
    ax.plot(msePredStep[:draw_len], Loss_n_mean[:draw_len], color=rgb_color_n, label='NODE', marker='o', markevery=int(num_epochs/5))
if run_AugNeuralODE:
    ax.plot(msePredStep[:draw_len], Loss_aug_mean[:draw_len], color=rgb_color_aug, label='ANODE', marker='d', markevery=int(num_epochs/5))
ax.set_xlabel('Epoch', fontsize=fontsize_xylabel)
ax.set_ylabel('Train loss', fontsize=fontsize_xylabel)
# plt.gca().set_facecolor('#f0f0f0')
ax.legend(fontsize=fontsize_legend)
ax.set_xlim([0,draw_len-1])
# ax.set_ylim([0, 0.01])
ax.grid(color='w')
fig_file=fig_path+'/slrobot-trainError.png'
plt.savefig(fig_file, dpi=300)
plt.show()


print('mean result')
if run_my:
    print("MSE between Input Concomitant ODE Prediction and GT:", msePred_cs_mean[-1], "      standard deviation:", msePred_cs_std[-1])
if run_CDE:
    print("MSE between Controlled Differential Equation Prediction and GT:", msePred_c_mean[-1], "      standard deviation:", msePred_c_std[-1])
if run_NeuralODE:
    print("MSE between Neural ODE Prediction and GT:", msePred_n_mean[-1], "      standard deviation:", msePred_n_std[-1])
if run_AugNeuralODE:
    print("MSE between Augmented Neural ODE Prediction and GT:", msePred_aug_mean[-1], "      standard deviation:", msePred_aug_std[-1])

print("")
if run_my:
    print("RMSE between Input Concomitant ODE Prediction and GT:", np.sqrt(msePred_cs_mean[-1]))
if run_CDE:
    print("RMSE between Controlled Differential Equation Prediction and GT:", np.sqrt(msePred_c_mean[-1]))
if run_NeuralODE:
    print("RMSE between Neural ODE Prediction and GT:", np.sqrt(msePred_n_mean[-1]))
if run_AugNeuralODE:
    print("RMSE between Augmented Neural ODE Prediction and GT:", np.sqrt(msePred_aug_mean[-1]))

if show_absErr:
    print("")
    if run_my:
        print("MAE between Input Concomitant ODE Prediction and GT:", absErr_cs_ode.mean(), "      standard deviation:", absErr_cs_ode.std())
    if run_CDE:
        print("MAE between Controlled Differential Equation Prediction and GT:", absErr_c_ode.mean(), "      standard deviation:", absErr_c_ode.std())
    if run_NeuralODE:
        print("MAE between Neural ODE Prediction and GT:", absErr_n_ode.mean(), "      standard deviation:", absErr_n_ode.std())
    if run_AugNeuralODE:
        print("MAE between Augmented Neural ODE Prediction and GT:", absErr_aug_ode.mean(), "      standard deviation:", absErr_aug_ode.std())

if show_maxErr:
    print("")
    if run_my:
        print("Max Error between Input Concomitant ODE Prediction and GT:", maxErr_cs_ode)
    if run_CDE:
        print("Max Error between Controlled Differential Equation Prediction and GT:", maxErr_c_ode)
    if run_NeuralODE:
        print("Max Error between Neural ODE Prediction and GT:", maxErr_n_ode)
    if run_AugNeuralODE:
        print("Max Error between Augmented Neural ODE Prediction and GT:", maxErr_aug_ode)

if show_R2:
    print("")
    if run_my:
        print("R2 between Input Concomitant ODE Prediction and GT:", R2_cs_ode)
    if run_CDE:
        print("R2 between Controlled Differential Equation Prediction and GT:", R2_c_ode)
    if run_NeuralODE:
        print("R2 between Neural ODE Prediction and GT:", R2_n_ode)
    if run_AugNeuralODE:
        print("R2 between Augmented Neural ODE Prediction and GT:", R2_aug_ode)