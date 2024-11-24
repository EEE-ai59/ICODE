draw=False
num_epochs = 1000 # 6000 # 8000 # 150 #### 15 ----------------- 1000
num_epochs_show= 100
lrList = [5e-4] #[1e-2, 5e-3, 1e-3] # 5e-4

n_seq = 5 # 5

hidden_dim = 500
num_layers = 2
augment_dim = 100
# aug_ini_hiddendim=1
aug_hidden_dim = 100
# 把u的信息融合到neuralODE中
# torch.sign(torch.sin(0.4*t))
USE_BASIC_SOLVER = True
# if USE_BASIC_SOLVER = False, the code uses ode solver in torchdiffeq

train_mode = True
initial_mode = True # False

zdim=1000