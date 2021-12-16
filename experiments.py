import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tqdm
import math
from scipy.special import softmax,expit
import torch
from torch.autograd import Variable

from binary_ts import *

d = 2
T = 2
N = 10

true_A = np.identity(d) + np.random.randn(d,d)
true_B = np.diag(np.random.uniform(low=-0.8,high=0.8,size=d))
true_sigma = np.diag(np.random.uniform(low=1,high=4,size=d))

bints = BinaryTS(d=d,T=T,N=N,true_A = true_A,true_B=true_B,true_sigma = true_sigma,R=1000)

bints.block_cavi(latent_lrs=[1e-3,1e-3,1e-3],mu_lr=1e-4,var_lr=1e-4,max_iter=5000,its=200,batchsize=1)
