import os
import yaml
import numpy as np
pwd=os.path.abspath(__file__)
par_pwd=os.path.dirname(pwd)
path = par_pwd+"/config/base.yaml" 
with open(path, "r") as f:
    config = yaml.safe_load(f)

tSwitchPoint=config['input']['tSwitchPoint']
tSwitchLen=config['input']['tSwitchLen']
xVal=config['input']['xVal']

u_dim = 1
def my_u(t, tSwitchPoint=tSwitchPoint, tSwitchLen=tSwitchLen, xVal=xVal):
    
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

num_epochs = config['train']['num_epochs']
class TrainInf():
    def __init__(self, num_epochs=num_epochs):        
        self.loss=np.zeros((num_epochs,))
        self.msePred=[]
        self.msePredStep=[]
        self.finalPredErr={}


def ZoomIn(ax, msePredStep, msePred_cs_mean, msePred_c_mean, msePred_n_mean, msePred_aug_ode_mean,width=0.3, height=0.2, pos=[0.6, 0.3], draw_len=num_epochs, y_ratio = 60):

    axins = ax.inset_axes((pos[0], pos[1], width, height), facecolor='#f0f0f0')
    axins.plot(msePredStep[:draw_len], msePred_cs_mean[:draw_len], color='blue', label='Input Concomitant ODE', linewidth=3, marker='*', markevery=10)
    axins.plot(msePredStep[:draw_len], msePred_c_mean[:draw_len], '--', color='c', label='CDE')
    axins.plot(msePredStep[:draw_len], msePred_n_mean[:draw_len], '-.', color='red', label='Neural ODE')
    axins.plot(msePredStep[:draw_len], msePred_aug_ode_mean[:draw_len], ':', color='g', label='Augmented ODE')
    # fig.gca().set_facecolor('#f0f0f0')

    
    zone_right = num_epochs-1
    zone_left = zone_right-2
   
    x_ratio = 0  
    # y_ratio = 60  


    xlim0 = msePredStep[zone_left]-(msePredStep[zone_right]-msePredStep[zone_left])*x_ratio
    xlim1 = msePredStep[zone_right]+(msePredStep[zone_right]-msePredStep[zone_left])*x_ratio


    y = np.hstack((msePred_cs_mean[zone_left:zone_right], msePred_cs_mean[zone_left:zone_right],
                msePred_cs_mean[zone_left:zone_right], msePred_cs_mean[zone_left:zone_right],
                msePred_cs_mean[zone_left:zone_right]))
    ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
    ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio


    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(ylim0, ylim1)


    tx0 = xlim0
    tx1 = xlim1
    ty0 = ylim0
    ty1 = ylim1
    sx = [tx0,tx1,tx1,tx0,tx0]
    sy = [ty0,ty0,ty1,ty1,ty0]
    ax.plot(sx,sy,"black")

    # xy = (xlim0,ylim0)
    # xy2 = (xlim0,ylim1)
    # con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
    #         axesA=axins,axesB=ax)
    # axins.add_artist(con)

    # xy = (xlim1,ylim0)
    # xy2 = (xlim1,ylim1)
    # con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
    #         axesA=axins,axesB=ax)
    # axins.add_artist(con)
    ax.arrow((zone_left+zone_right)/2, np.max(y)+(np.max(y)-np.min(y))*y_ratio*2, -pos[0]-width/2, pos[1]*0.6, width=0.03, zorder=2)
    