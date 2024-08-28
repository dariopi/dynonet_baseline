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

git clone https://github.com/dariopi/dynonet_baseline.git

cd name_of_the_folder

pip install -r requirements.txt

In our tests, we used Python 3.11.9

## Usage

To run the training script, navigate to the project directory and execute:

```bash
python dynonet_baseline.py --benchmark_name Silverbox
```
or 

```bash
python dynonet_baseline.py --benchmark_name EMPS
```
or

```bash
python dynonet_baseline.py --benchmark_name WienerHammerBenchMark
```

or

```bash
python dynonet_baseline.py --benchmark_name CED
```
or

```bash
python dynonet_baseline.py --benchmark_name Cascaded_Tanks
```

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. To view a copy of this license, visit [http://creativecommons.org/licenses/by-nc/4.0/](http://creativecommons.org/licenses/by-nc/4.0/) or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

## Authors

- **Dario Piga** (dario.piga@idsia.ch)
- **Marco Forgione** (marco.forgione@idsia.ch)

## Citing

If you find this project useful, we encourage you to cite the [paper](https://onlinelibrary.wiley.com/doi/abs/10.1002/acs.3216):


```
@article{forgione2021dyno,
  title={\textit{dyno{N}et}: A neural network architecture for learning dynamical systems},
  author={Forgione, M. and Piga, D.},
  journal={International Journal of Adaptive Control and Signal Processing},
  volume={35},
  number={4},
  pages={612--626},
  year={2021},
  publisher={Wiley}
}
```


