import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import interp2d

M=49
r=0.6 # 0.6
Tnum=50
input_dim=(M+1)*(M+1)
n_seq=3
# def u(t):
#     return 0.5+(4*t)**2+2*t
h=1/M
k=r*h**2
T=k*Tnum
tSwitchPoint=T*np.array([0.06, 0.43, 0.78, 0.9])
tSwitchLen=T*0.01
# tSwitchPoint=[0.06, 0.3, 0.6]
# tSwitchLen=0.01
xVal=[-0.1, 1]
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

# def my_du(t, tSwitchPoint=tSwitchPoint, tSwitchLen=tSwitchLen, xVal=xVal):
#     # 二值切换
#     if t<tSwitchPoint[0]:
#         return 0
#     count=-1
#     for i,it in enumerate(tSwitchPoint):
#         if t<it:
#             break
#         else:
#             count+=1
#     if t>=tSwitchPoint[count]+tSwitchLen:
#         return 0
    
#     t0=tSwitchPoint[count]
#     k=-6/tSwitchLen**3 * (t-t0)**2 + 6/tSwitchLen**2 * (t-t0)
#     x0=xVal[count%2]
#     x1=xVal[(count+1)%2]
#     res=x0+k*(x1-x0)
#     return res


def chaseMethod(A,d):

    # return np.linalg.inv(A)@d
    M=A.shape[0]

    g=np.zeros((M,1))
    w=np.zeros((M,1))
    U=np.zeros((M,1))


    
    for m in range(0,M):
        if m==0:
            A[m,m+1]=-A[m,m+1]
        elif m==M-1:
            A[m,m-1]=-A[m,m-1]
        else:
            A[m,m-1]=-A[m,m-1]
            A[m,m+1]=-A[m,m+1]

    for m in range(0,M):
        if m==0:
            g[m]=d[m]/A[m,m]
            w[m]=A[m,m+1]/A[m,m]
        elif m==M-1:
            g[m]=(d[m]+A[m,m-1]*g[m-1])/(A[m,m]-A[m,m-1]*w[m-1])
        else:
            g[m]=(d[m]+A[m,m-1]*g[m-1])/(A[m,m]-A[m,m-1]*w[m-1])
            w[m]=A[m,m+1]/(A[m,m]-A[m,m-1]*w[m-1])

    U[M-1]=g[M-1]
    for m in range(M-2,-1,-1):
        U[m]=g[m]+w[m]*U[m+1]

    return U




def getData(n_seq, M=M,r=r,Tnum=Tnum):
    h=1/M
    k=r*h**2
    T=k*Tnum
    tL= np.arange(0,T,k) 
    tL=np.append(tL, T)
    N=len(tL)
    res=np.zeros((n_seq, N, (M+1)*(M+1)))
    for i in range(n_seq):
        temp=getData_1(M=M,r=r,Tnum=Tnum)
        temp=temp.reshape(1, N, (M+1)*(M+1))
        res[i,:,:]=temp
    
       
    return res





def getData_1(M=M,r=r,Tnum=Tnum):
    # use the PR format
    h=1/M
    k=r*h**2
    T=k*Tnum

    tL= np.arange(0,T,k) 
    tL=np.append(tL, T)
    N=len(tL) ### ?
    U0=np.zeros((M+1,M+1))
    res=np.zeros((N,M+1,M+1))

    # 赋初值 version 1 one peak
    # for l in range(0,M+1):
    #     for m in range(0,M+1):
    #         U0[l,m]=np.sin(np.pi/10*l*h)*np.sin(np.pi/10*m*h)

    # 赋初值 version 2 multi peak
    N_U0=int((M+1)/5)
    x = np.linspace(0, 1, N_U0)
    y = np.linspace(0, 1, N_U0)
    x_fine = np.linspace(0, 1, M+1)
    y_fine = np.linspace(0, 1, M+1)
    # np.random.seed(0)
    U0_temp=np.random.rand(N_U0, N_U0)
    interp_U0 = interp2d(x, y, U0_temp, kind='cubic')
    U0 = interp_U0(x_fine, y_fine) 
    
    # for l in range(0,M+1):
    #     for m in range(0,M+1):
    #         if l < M/2 and m < M/2:
    #             U0[l,m]=0.1*np.cos(np.pi/10*l*h)*np.cos(np.pi/10*m*h)
    #         elif l > M/2 and m > M/2:
    #             U0[l,m]=0.1*np.sin(np.pi/5*(l-M/2)*h)*np.sin(np.pi/5*(m-M/2)*h)
    #         elif l > M/2 and m < M/2:
    #             U0[l,m]=0.2*np.sin(np.pi/5*(l-M/2)*h)*np.sin(np.pi/5*m*h)
    #         elif l < M/2 and m > M/2:
    #             U0[l,m]=-0.1*np.sin(np.pi/5*l*h)*np.sin(np.pi/5*(m-M/2)*h)


    U=U0.copy()
    V=U0.copy()
    Nn=0
    res[Nn]=U.copy()
    Nn+=1


    A=np.zeros((M+1,M+1)) 

    for t in tL[1:]:

        # U[:int(M/2),:int(M/2)]+=my_u(t)*1000*k ##############
        U[:,:]+=my_u(t)*10*k # 1000
    

        # print('t')
        for m in range(0,M+1):
            if m==0:
                b=U[:,m]+r/2*(-2*U[:,m]+U[:,m+1])
            elif m==M:
                b=U[:,m]+r/2*(U[:,m-1]-2*U[:,m])
            else:
                b=U[:,m]+r/2*(U[:,m-1]-2*U[:,m]+U[:,m+1])

            for i in range(0,M+1):
                if i==0:
                    A[i,i]=1+r
                    A[i,i+1]=-r/2
                elif i==M:
                    A[i,i-1]=-r/2
                    A[i,i]=1+r
                else:
                    A[i,i-1]=-r/2
                    A[i,i]=1+r
                    A[i,i+1]=-r/2
    
            V[:,m]=chaseMethod(A,b).flatten()

        for l in range(0,M+1):
            if l==0:
                b=V[l,:]+r/2*(-2*V[l,:]+V[l+1,:])
            elif l==M:
                b=V[l,:]+r/2*(V[l-1,:]-2*V[l,:])
            else:
                b=V[l,:]+r/2*(V[l-1,:]-2*V[l,:]+V[l+1,:])

            for i in range(0,M+1):
                if i==0:
                    A[i,i]=1+r
                    A[i,i+1]=-r/2
                elif i==M:
                    A[i,i-1]=-r/2
                    A[i,i]=1+r
                else:
                    A[i,i-1]=-r/2
                    A[i,i]=1+r
                    A[i,i+1]=-r/2
    
            U[l,:]=chaseMethod(A,b).flatten()

           
        res[Nn]=U.copy()
        Nn+=1

    return res






if  __name__ == '__main__':
    # M=10 # number of intervals # 50
    # r=0.5 #0.1
    # # T=0.2  # 0.17
    # Tnum=500
    res=getData(n_seq, M,r,Tnum)
    res=res.reshape(n_seq, res.shape[1], (M+1), (M+1))
    res=res[0]
    U0=res[0]
    U=res[-1]

    X = np.linspace(0,10,M)
    X=X[1:]
    Y = np.linspace(0,10,M)
    Y=Y[1:]
    X, Y = np.meshgrid(X, Y)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, U0[1:-1,1:-1], cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, U[1:-1,1:-1], cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, U[1:-1,1:-1]-U0[1:-1,1:-1], cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    err=U0-U
