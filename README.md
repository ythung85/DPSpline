# DPS - Pytorch Implementation 
This is a PyTorch implementation of "Deep P-Spline: Fast tuning and Theory", where we propose the novel structure as follow compared to traditional Deep Neuron Networks (DNNs).

![](./imgs/DPS.png)
# Quick Start
## Dependency
You can install the Python dependencies with
```
pip3 install -r requirements.txt
```

# Simulation
## 100 individual trials for Example 1
```math
    y = \exp\{2\sin(0.5\pi x_1) + 0.5\cos(2.5\pi x_2) + \epsilon\}
```
The user can customize the specific size for training and testing. For instance, we assume the training size = 800 and the testing size = 200

For 1 Layer DPS:
```
python3 DPS_simulation.py --data A --trainsize 800 --testsize 200 --Fin 2 --Fout 1 --nk 15 --nm 50 --rep 100
```

For 2 Layer DPS:
```
python3 2DPS_simulation.py --data A --trainsize 800 --testsize 200 --Fin 2 --Fout 1 --nk 15 --nm 50 --rep 100
```

Afterwards, we will calculate the MSPE and its standard deviation over 100 trials with respect to different training size, and the result is summarized below.

Trainsize\ Method | DNN-S | DPS | 2DNN-S | 2DPS | 
--- | --- | --- | --- |--- 
200 | 0.086 (0.147) | 0.048 (0.024) | 0.088 (0.081) | 0.051 (0.031) |
400 | 0.061 (0.116) | 0.031 (0.013) | 0.078 (0.217) | 0.028 (0.008) |
800 | 0.074 (0.141) | 0.034 (0.017) | 0.044 (0.076) | 0.024 (0.008) | 






