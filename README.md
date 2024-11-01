This is based on deepxde (https://github.com/lululxvi/deepxde). The loss function and network structures are modified for our specific use.
Use $env:DDE_BACKEND = "tensorflow" to run the main.py for survival curve estimation.

The codes added and modified: 

./deepxde/losses.py 
add custom_loss() and custom_loss_discrete()

./deepxde/nn/tensorflow/deeponet.py
add CNN structure in class DeepONet(), add class DeepONetFunctional()

./deepxde/data/simulator.py
the file is written for our simulations
