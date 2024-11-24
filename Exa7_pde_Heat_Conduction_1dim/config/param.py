draw=False
num_epochs = 400 # 8000 # 150 #### 15 ----------------- 1000
num_epochs_show= 5
lrList = [2e-2] #[1e-1] #[1e-2, 5e-3, 1e-3] # 5e-4

n_seq = 100 # 5

hidden_dim = 60
num_layers = 2
augment_dim = 10
# aug_ini_hiddendim=1
aug_hidden_dim = 10
# 把u的信息融合到neuralODE中
# torch.sign(torch.sin(0.4*t))
USE_BASIC_SOLVER = True
# if USE_BASIC_SOLVER = False, the code uses ode solver in torchdiffeq

train_mode = True
initial_mode = True # False

zdim=1000