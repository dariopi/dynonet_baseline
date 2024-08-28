# configuration file

import nonlinear_benchmarks

def get_config(model_type, **kwargs):
    # Deafult configuration for all models: contains common settings that apply to almost all scenarios.
    base_config = {
        'lr': 0.001,  # Learning rate for the optimizer.
        'seq_len': 2048,  # Length of each sub-sequence in the dataset.
        'batch_size': 16,  # Batch size.
        'max_epochs': 10001,  # Maximum number of epochs to run during training.
        'shuffle': True,  # Whether to shuffle the dataset each epoch.
        'n_skip': 20,  # Number of initial timesteps to skip in loss calculation.
        'print_frequency': 50,  # Frequency of epochs to print training progress.
        'save_frequency': 8000,  # Frequency of epochs at which to save a checkpoint.
        'hidden_size': 20,  # Size of the hidden layers in the dynonet model.
        'load': False,  # Flag to indicate whether to load a saved model.
        'simulate_train': False,  # Flag to simulate training data in predictions.
        'seed': 42,  # Seed for random number generators to ensure reproducibility.
        'n_a': 10,  # Output lag in the dynamical model.
        'n_b': 10,  # Input lag in the dynamical model.
        'plot': True  # Whether to plot results during evaluation.
    }

    # Model-specific configurations: contains settings unique to each type of model or experiment.
    model_configs = {
        'Silverbox': {
            'save_path': 'checkpoints/silverbox_checkpoint.pth',  # Path to save the model checkpoint.
            'command_load': nonlinear_benchmarks.Silverbox  # Function to load Silverbox dataset.
        },
        'WienerHammerBenchMark': {
            'save_path': 'checkpoints/wienerhammerstein_checkpoint.pth',  # Path for checkpoint.
            'command_load': nonlinear_benchmarks.WienerHammerBenchMark  # Function to load dataset.
        },
        'Cascaded_Tanks': {
            'save_path': 'checkpoints/cascaded_tanks_checkpoint.pth',
            'command_load': nonlinear_benchmarks.Cascaded_Tanks
        },
        'EMPS': {
            'save_path': 'checkpoints/emps_checkpoint.pth',
            'command_load': nonlinear_benchmarks.EMPS,
            'seq_len': 25000,  # Very long sequence length for the EMPS dataset.
            'max_epochs': 25001,  # Increased number of epochs due to the complexity of the dataset.
            'lr': 1e-4  # Lower learning rate for potentially finer convergence in complex models.
        },
        'CED': {
            'save_path': 'checkpoints/ced_checkpoint.pth',
            'command_load': nonlinear_benchmarks.CED,
            'simulate_train': True,  # Simulate the train to have good initial conditions for the test.
            'max_epochs': 10001
        }
    }

    # Error handling to ensure the specified model type is supported.
    if model_type not in model_configs:
        raise ValueError("Unsupported model type specified")

    # Merge the base config with the model-specific one, and then with any custom overrides
    config = {**base_config, **model_configs[model_type], **kwargs}

    return config
