# Use the following format to run the script from command line, replacing 'benchmark_name' and other default parameters as needed:
# python dynonet_baseline.py --benchmark_name <benchmark_name> --<parameter> <value>
# Example:
# python dynonet_baseline.py --benchmark_name Silverbox --lr 0.001 --batch_size 16
# This script configures and trains a dynoNet model on specified benchmarks, allowing the user to adjust training parameters dynamically.
# 
# Authors: Dario Piga, Marco Forgione, Lugano, 22nd May, 2024.
#
# Copyright (C) [2024] [Dario Piga, Marco Forgione]
# 
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
# 
# You are free to:
# - Share — copy and redistribute the material in any medium or format
# - Adapt — remix, transform, and build upon the material
# 
# The licensor cannot revoke these freedoms as long as you follow the license terms.
# 
# Under the following terms:
# - Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
#   You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
# - NonCommercial — You may not use the material for commercial purposes.
# 
# No additional restrictions — You may not apply legal terms or technological measures that legally restrict others
# from doing anything the license permits.
# 
# Notices:
# You do not have to comply with the license for elements of the material in the public domain or where your use is
# permitted by an applicable exception or limitation.
# 
# No warranties are given. The license may not give you all of the permissions necessary for your intended use. For
# example, other rights such as publicity, privacy, or moral rights may limit how you use the material.


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

default_benchmark = 'Silverbox'

def parse_args():
    """
    Parses command-line arguments to configure the model and training settings.
    """

    parser = argparse.ArgumentParser(description='dynoNet training on various datasets')
    parser.add_argument('--benchmark_name', type=str, default=default_benchmark, help='Type of model to train')
    
    # Temporary args parsing to get model type
    temp_args, _ = parser.parse_known_args()
    config = cfg.get_config(temp_args.benchmark_name)
    
    # Add arguments dynamically based on current model configuration
    for key, val in config.items():
        if key != 'command_load' and not any(action.dest == key for action in parser._actions):  # Prevent duplication
            parser.add_argument(f'--{key}', type=type(val), default=val, help=f'Set {key} (default: {val})')

    args = parser.parse_args()
    return args, config

def get_config():
    """
    Loads configuration using command-line arguments and updates global configuration.
    """
    args, config = parse_args()

    # Update the model configuration with command line arguments
    for key in config:
        if hasattr(args, key):
            config[key] = getattr(args, key)

    print("Configuration used:", config)

    return config



def set_seed(seed=42):
    """Set seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SubSeqDataset(Dataset):
    """
    Custom dataset to handle sub-sequences of input-output pairs for training.
    """
        
    def __init__(self, train, seq_len):
        # Initialize lists to store sequences
        self.u_list = []
        self.y_list = []

        # Process each (u_train, y_train) pair in the training data
        for u_train, y_train in train:
            N, nu = u_train.shape  # Get the total number of samples and features in u
            _, ny = y_train.shape  # Get the total number of features in y

            # Calculate the number of full sequences possible, minus the last potentially shorter sequence
            n = int(np.ceil(N / seq_len))

            # Iterate through each sequence index
            for ind in range(n):
                start_idx = ind * seq_len
                end_idx = start_idx + seq_len

                # Adjust the start index of the last sequence if it's shorter than seq_len
                if end_idx > N:
                    start_idx = max(0, N - seq_len)  # Start the last sequence so that it ends exactly at the last data point
                    end_idx = N

                # Extract the sequences
                u_subseq = u_train[start_idx:end_idx, :]
                y_subseq = y_train[start_idx:end_idx, :]

                # Store the sequences
                self.u_list.append(u_subseq)
                self.y_list.append(y_subseq)

        # Convert lists to numpy arrays and then to PyTorch tensors
        self.u = torch.from_numpy(np.array(self.u_list, dtype=np.float32))
        self.y = torch.from_numpy(np.array(self.y_list, dtype=np.float32))

    def __len__(self):
        return len(self.u_list)

    def __getitem__(self, idx):
        return self.u[idx], self.y[idx]

class Scaler:
    """
    Scaler class for normalizing and denormalizing data.
    """

    def __init__(self):
        self.mean = None
        self.std = None
        

    def fit(self, data):
        """Fit the scaler to the data by calculating mean and standard deviation."""
        self.mean = data.mean()
        self.std = data.std()

    def transform(self, data):
        """Normalize the data using the calculated mean and standard deviation."""
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """Revert the normalization to return to the original scale."""
        return data * self.std + self.mean

class my_dynoNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_a, n_b):
        super(my_dynoNet, self).__init__()

        self.G1 = MimoLinearDynamicalOperator(
            in_channels=input_size, out_channels=4, n_b=n_b, n_a=n_a
        )
        self.F = MimoStaticNonLinearity(4, 3, n_hidden=hidden_size)
        self.G2 = MimoLinearDynamicalOperator(
            in_channels=3, out_channels=output_size, n_b=n_b, n_a=n_a
        )
        self.Glin = MimoLinearDynamicalOperator(
            in_channels=input_size, out_channels=output_size, n_b=n_b, n_a=n_a
        )

    def forward(self, x_in):

        x = self.G1(x_in)
        x = self.F(x)
        x = self.G2(x)
        y = x + self.Glin(x_in)

        return y
    

def train_model(train):
    """
    Trains a model according to the given configuration and dataset.

    Parameters:
        train (list): The training dataset.

    Returns:
        torch.nn.Module: The trained model.
    """

    # Extracting configuration parameters
    lr = config['lr']
    max_epochs = config['max_epochs']
    n_skip = config['n_skip']
    print_frequency = config['print_frequency']
    save_frequency = config['save_frequency']
    save_path = config['save_path']
    seq_len = config['seq_len']
    batch_size = config['batch_size']
    shuffle = config['shuffle']
    n_a = config['n_a']
    n_b = config['n_b']

    # data normalization
    train, u_scaler, y_scaler = data_normalizer(train)

    # Initialize the dataset and dataloader
    my_dataset = SubSeqDataset(train, seq_len)
    loader = DataLoader(my_dataset, shuffle=shuffle, batch_size=batch_size)

    # Model initialization
    model = my_dynoNet(input_size=1, hidden_size=config['hidden_size'], output_size=1, n_a=n_a, n_b=n_b)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Model Configuration:\n{model}\nTraining starts with lr={lr} and max_epochs={max_epochs}")

    # Load checkpoint if needed

    start_epoch = 0
    if config.get('load', False):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Model loaded')


    # Training loop
    for epoch in range(start_epoch, max_epochs):
        loss_total = 0.0

        for u, y in loader:
            y_hat = model(u)
            loss = torch.mean((y_hat[:, n_skip:, :] - y[:, n_skip:, :]) ** 2)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total += loss.item()
    

        # Periodic logging
        if epoch % print_frequency == 0:
            print(f'Epoch [{epoch + 1}/{max_epochs}], Loss: {loss_total:.4f}')


        # Periodic saving 
        if epoch % save_frequency == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(checkpoint, save_path)
            print(f"Checkpoint saved at epoch {epoch + 1} with loss {loss_total:.4f}")

    
    return {'model': model, 'u_scaler': u_scaler, 'y_scaler': y_scaler}


def apply_model(dict_model, u_test, y_in, u_train = None):

    
    # Model prediction on scaled inputs; scaled outputs used only for initial conditions
    model = dict_model['model']
    u_scaler = dict_model['u_scaler'] 
    y_scaler = dict_model['y_scaler'] 

    if config['simulate_train']:
        # Concatenate training and testing inputs for the model in case of training simulation
        u_conct = np.concatenate((u_scaler.transform(u_train), u_scaler.transform(u_test)))
        y_hat_conc = model(torch.tensor(u_conct).unsqueeze(0).float()) #.squeeze(dim = 0)

        # Append outputs after the training portion
        y_hat_norm = y_hat_conc[[0], u_train.shape[0]:,:]

        # Convert normalized model outputs back to original scale for evaluation
            

    else:
        u_test = u_scaler.transform(u_test)
        y_hat_norm = model(torch.tensor(u_test).unsqueeze(0).float())

    y_hat = y_scaler.inverse_transform(y_hat_norm) 

    return y_hat.squeeze(0).detach().numpy()


def data_normalizer(train):
   
    train_norm = copy.deepcopy(train)

    u_scaler = Scaler()
    y_scaler = Scaler()

    # Concatenate all u and y data from the train set for fitting scalers

    conc_u_train = np.concatenate([t.u for t in train])
    conc_y_train = np.concatenate([t.y for t in train])
                                  
    u_scaler.fit(conc_u_train)
    y_scaler.fit(conc_y_train)

    # Transform each train data point using the fitted scalers
    for ind in range(len(train)):
        train_norm[ind].u = u_scaler.transform(train[ind].u)
        train_norm[ind].y  = y_scaler.transform(train[ind].y)
    

    return train_norm, u_scaler, y_scaler

def print_and_plot(test_data, model_outputs):
    """
    Plots the test results against the model outputs.

    Args:
    test_data (list-like): List containing the test datasets.
    model_outputs (list-like): List containing the model outputs.
    """
    rmse_summary = []  # Initialize an empty list to store RMSE summary strings.


    for ind, test in enumerate(test_data):
        y_test = test.y
        y_model = model_outputs[ind]
        n = test.state_initialization_window_length

        for output in range(y_test.shape[1]):
            rmse = RMSE(y_test[n:, output], y_model[n:, output])
            nrmse = NRMSE(y_test[n:, output], y_model[n:, output])
            r2 = R_squared(y_test[n:, output], y_model[n:, output])
            mae = MAE(y_test[n:, output], y_model[n:, output])
            fit = fit_index(y_test[n:, output], y_model[n:, output])
            rmse_summary.append(rmse)


            # Print all metrics for each test and output.
            print(f'Test {ind}, Output {output}: RMSE: {rmse:.8f}, R^2: {r2:.2f}, Fit Index: {fit:.2f}')


            if config['plot']:
                plt.figure(figsize=(10, 4))
                plt.plot(y_test[:, output], label='Original y_test', color='blue')
                plt.plot(y_model[:, output], label='Model y_test', linestyle='--', color='red')
                plt.plot(y_test[:, output] - y_model[:, output], label='Error', color='green')
                plt.title(f'Comparison and Error for Test {ind}, Output {output}')
                plt.axvline(n, color = 'grey')
                plt.xlabel('Sample Index')
                plt.ylabel('Values / Error')
                plt.legend()
                plt.show()

    # Print the RMSE summary for all tests and outputs.
    rmse_summary_str = ', '.join([f'{value:.8f}' for value in rmse_summary])
    print("All RMSEs: ", rmse_summary_str)


if __name__ == "__main__":
    # Load the configuration settings specific to the selected benchmark or model type.
    config = get_config()

    # Set a random seed to ensure that all random operations are reproducible, particularly important for training neural networks.
    set_seed(seed=config['seed']) 

    # Load the training and testing data according to the specifications in the configuration.
    # This ensures that the data is always loaded in a consistent format (at least 2D, and as tuples of datasets).
    train, test = config['command_load'](atleast_2d=True, always_return_tuples_of_datasets=True)

    # Train the model on the loaded and normalized training data.
    start_time = time.time()
    dict_model = train_model(train)
    train_time = time.time() - start_time

    # Initialize a list to store the output predictions from the model for each test dataset.
    y_test_model = []

    # Iterate through each test dataset, applying the trained model to obtain predictions.
    for ind, test_case in enumerate(test):
        # Retrieve the initial condition window length for the state from the test data.
        n = test_case.state_initialization_window_length

        # Depending on the 'simulate_train' configuration, the model may also use training data for simulation. 
        # Useful for good initialization of the test dataset if training and test are concatenated
        # If so, the model applies itself to both the training inputs and the test inputs.
        # Note: even if 'simulate_train'== True, only the output of the test dataset is returned 
        if config['simulate_train']:
            y_test_model_s = apply_model(dict_model, test_case.u, test_case.y[:n], train[ind].u)
        else:
            # Otherwise, apply the model only to the test inputs.
            y_test_model_s = apply_model(dict_model, test_case.u, test_case.y[:n])
        
        # Append the scaled model outputs to the list for comparison and plotting.
        y_test_model.append(y_test_model_s)

    # After processing all test cases, display and plot the results.
    print_and_plot(test, y_test_model)
    print(f'Training time: {train_time:.2f} s')
