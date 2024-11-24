import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import interp2d
from config.param import *
from scipy.interpolate import interp1d
# M=50
# r=0.6 # 0.6
# Tnum=50
# input_dim=(M+1)*(M+1)

# def u(t):
#     return 0.5+(4*t)**2+2*t
# h=10/M
# k=r*h**2
# T=k*Tnum

# x=np.linspace(0,10,M+1)
# U=np.sin(np.pi/10*x)

r=0.1
h=0.25
T_num=300
M=int(10/h)
input_dim=M+1
k=r*h**2
def f(t):
    # ans=-np.pi/10*np.exp(-np.pi**2/10**2*t)
    # ans=1
    ans=2*np.sin(np.pi*2*t)*np.exp(-t/5)+0.1
    return ans

def explicit(h,r,T_num):
    
    k=r*h**2
    T=T_num*k
    M=int(10/h)
    x=np.linspace(0,10,M+1)

    xx = np.linspace(0, 10, 5)
    U0_temp=2*np.random.rand(5,)-1
    interp_U0 = interp1d(xx, U0_temp, kind='cubic')
    U0 = interp_U0(x) 

    # U0=np.sin(np.pi/10*x)#+np.random.rand(M+1,)
    # U0=np.random.rand(M+1,)
    U=np.empty((T_num, M+1))
    U[0,:]=U0
    B=np.zeros((M+1, M+1))

    


    for t in range(T_num-1):
        for j in range(M+1):
            if j==0:
                B[j,j]=1-2*r*(1+h)
                B[j,j+1]=2*r
            elif j==M:
                B[j,j-1]=2*r
                B[j,j]=1-2*r*(1+h)
            else:
                B[j,j-1]=r
                B[j,j]=1-2*r
                B[j,j+1]=r
        
        e=np.zeros((M+1,))
        e[0]=2*r*h*f(t)
        e[M]=2*r*h*f(t)
        U[t+1]=(B@U[t].reshape(-1,1)).flatten()+e
    
    return U

def getData_1(r, h, T_num):
    res=explicit(h,r,T_num)
    return res

def getData(n_seq=n_seq, r=r, h=h, T_num=T_num):
    M=int(10/h)
    res=np.zeros((n_seq, T_num, M+1))
    for i in range(n_seq):
        temp=getData_1(r=r, h=h, T_num=T_num)
        # temp=temp.reshape(1, T_num, (M+1)*(M+1))
        res[i,:,:]=temp

    return res


if __name__ == '__main__':
    res=getData()
    print(res.shape)
    
    # res=res.reshape(n_seq, res.shape[1], (M+1), (M+1))
    # res=res[0]
    # U0=res[0]
    # U=res[-1]

    X = np.linspace(0,k*T_num,T_num)
    
    Y = np.linspace(0,10,M+1)
    
    X, Y = np.meshgrid(X, Y)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, res[0,:,:].T, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # surf = ax.plot_surface(X, Y, U[1:-1,1:-1], cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # surf = ax.plot_surface(X, Y, U[1:-1,1:-1]-U0[1:-1,1:-1], cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()
    # err=U0-U
































