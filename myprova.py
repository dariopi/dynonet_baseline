import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import config as cfg  
import nonlinear_benchmarks
from nonlinear_benchmarks.error_metrics import RMSE, NRMSE, R_squared, MAE, fit_index
from dynonet.lti import MimoLinearDynamicalOperator
from dynonet.static import MimoStaticNonLinearity
import os
import copy
import time


train, test = nonlinear_benchmarks.Cascaded_Tanks(atleast_2d=True, always_return_tuples_of_datasets=True)

print(train)

