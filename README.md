# DynoNet Baseline Model for [nonlinear benchmarks](https://www.nonlinearbenchmark.org/)

The DynoNet Baseline Model project implements a machine learning model designed to simulate and predict the behavior of various dynamical systems. This project utilizes the dynoNet library to build, train, and evaluate models capable of handling nonlinear dynamics with a specific focus on real-world benchmarks like Silverbox, Wiener-Hammerstein systems, and others.

## Features

- Implementation of dynamical models using [dynoNet](https://github.com/forgi86/dynonet).
- Ability to handle multiple benchmarks such as Silverbox, Wiener-Hammerstein, CED, EMPS, Cascaded Tanks
- Custom dataset handling for training models on sub-sequences of input-output pairs.
- Data normalization and scaling utilities.
- Evaluation metrics included (RMSE, NRMSE, R-squared, MAE, fit index).
- Visualization of model predictions against actual data.

## Installation

Clone this repository and install the required dependencies:

git clone https://github.com/dariopi/benchmarks_baseline.git
cd dynonet_baseline
pip install -r requirements.txt

## Usage

To run the training script, navigate to the project directory and execute:

```bash
python dynonet_baseline.py --benchmark_name Silverbox --lr 0.001 --batch_size 16
```

Replace 'Silverbox' with your desired benchmark and adjust other parameters such as the learning rate (--lr) and batch size (--batch_size)
as needed to suit your training requirements.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. To view a copy of this license, visit [http://creativecommons.org/licenses/by-nc/4.0/](http://creativecommons.org/licenses/by-nc/4.0/) or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

## Authors

- **Dario Piga** (dario.piga@idsia.ch)
- **Marco Forgione** (marco.forgione@idsia.ch)


