## About

This application is built to create, train, and test artificial neural networks with the following characteristics:

* can have multiple hidden layers
* feedforward
* multi-input / single-output
* non-linear hidden nodes with hyperbolic tangent sigmoid transfer function
* linear output node with linear transfer function

The GUI is built using python's native tkinter library, the math is done using numpy, and the plots are generated with matplotlib. The application was developed in Python 3.10 on a Windows 10 system.

### <br />The training algorithm

The forward propagation of input signals to the output of a layer of nodes is:

$$ y = f(W_n*x + W_b) $$

where $W_n$ is the matrix of connection weights with previous nodes, $W_b$ is the bias weight vector, $x$ is the vector of input signals, and *f()* is the transfer function. The choice of training algorithm is gradient descent with a fixed learning rate $\alpha$ and momentum coefficient $\beta$. The weight updates $\Delta W^m$ are a function of the network error gradient and the previous weight update for each layer $m$:

$$\Delta W^m = -\alpha\frac{\partial E}{\partial W^m} + \beta(\Delta W^m)_{prev}$$

The error performance criteria is the mean squared error between the target ($T$) and the network output ($O$) for $N$ datasets:

$$MSE = \frac{1}{N}\sum_{k=1}^N(T_k-O_k)^2 $$

This application performs batch training where all datasets are processed per training iteration (or epoch) therefore the error criteria for training becomes $E=(T-O)^2$ and the weight updates are averaged before updating the network weights:

$$\Delta W^m = \frac{1}{N}\sum_{k=1}^N\Delta W_k^m $$

## <br />Installation

The installation process will create a vitrual environment into which the necessary libraries from the requirements file will be installed and with which the application will be launched. Be sure to have Python installed and added to PATH under environment variables.

1. Download the project files (extracted from zip)
2. Execute `Installation.bat`

The folder size after installation is about 180 MB

## <br />Usage

After running the installation, the application may be launched with `Run.bat`

1. The first step is to select a training data file. This is a spreadsheet file with the following requirements:
    - Have a unique sheet titled "Data"
    - The first row is reserved for input and output labels
    - The labels must start with either "x" to designate input or "y" to designate output
    - There must be at least one "x" column and only one "y" column<br />
    \**Spreadsheet formats supported: .ods, .xls, .xlsx*
2. Choose whether to use all the datasets provided in the file for training or to use a part of the datasets as untouched data to test and validate the network during training
3. Adjust the fixed training parameters:
    - HL architecture: number of nodes per hidden layer separated by commas
    - Maximum epochs: max number of training iterations
    - Validation break: number of training iterations for which when reached with increasing validation error the training will stop (divergence break)
    - Learning rate
    - Momentum coefficient
4. Choose whether to generate and run a single network training or a batch where:
    - A network is created and trained
    - Network weights are reinitialized and the *new* network is trained again in repetition untill reaching the sequence number entered
    - The best network weight configuration is chosen based on final error (validation or training, depending on step 2 above)
5. Train new network: training will be carried out and the accompanying performance plots will be displayed on the right window
    - The MSE and weight variation plots are in log scale
    - The weight variation is the absolute value of the average total weight update (bias and node connections) per iteration
    - The correlation R is Pearson's correlation coefficient

The user is then free to carry out the following actions as desired:

- Retrain the current network in an attempt to further improve the performance and make the network fit the training data much closer
- Save the network
- Load a network
- Test the network with different datasets by clicking on the testing window button
    - A testing data file should be selected
    - The testing data file should respect the same format as the training data file only it is no longer necessary to have a "y" column
    - If the testing data contains targets (a "y" column) then the correlation coefficient will be automatically calculated and displayed
    - The training data file can be selected here and results from the training window may be found and confirmed

### <br />Tutorial

Create and train a network on the following model:
$$f(x) = 20(x_1)^2 + x_2 - 0.7(x_3)^3 + \sqrt{x_4}$$

Prepare the training file with randomized $x$ values and plug the above formula for $f(x)$ into the "y" column for the model targets

<p align="center"><br />
  <img src="/images/training-file.png" />
</p><br />

Run the application. Consider the following parameters (some of which are already set by default):
- Train with 20% of the datasets as validation data
- HL architecture: 5
- Maximum epochs: 100
- Validation break: 6
- Learning rate: 0.2
- Momentum coefficient: 0.1
- Single network training

Numerous training sequences can be run untill producing a network with satisfactory results:

<p align="center"><br />
  <img src="/images/tuto-training.png" />
</p><br />

For each training sequence there is randomness resulting from network weight initialization and random selection of validation datasets

A simple test would be to randomize the $x$ values again. Selecting the test file in the testing window and clicking on the network response button we see the output of the last trained network on this new data:

<p align="center"><br />
  <img src="/images/tuto-testing.png" />
</p><br />

## <br />License

Distributed under the GPLv3 License. See `LICENSE.txt` for more information.
