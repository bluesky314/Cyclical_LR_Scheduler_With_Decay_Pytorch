# Cyclical Learning Rate Scheduler With Decay in Pytorch

Adapted from: https://github.com/Harshvardhan1/cyclic-learning-schedulers-pytorch

Reach multiple minimas to create a powerful ensemble or just to find the best one using Cyclical Learning Rates with Decay. Ideally decay milestones should intersect with cyclical milestones for smooth transition as shown below. Can be used with any optimizer such as Adam.




# Cyclic learning rate schedulers -PyTorch

### Implementation
Cyclic learning rate schedules -
- cyclic cosine annealing - CycilcCosAnnealingLR()
- cyclic linear decay - CyclicLinearLR()

### Requirements
- numpy 
- python >= 2.7
- PyTorch >= 0.4.0

### Reference
<a href= https://arxiv.org/pdf/1608.03983.pdf> *SGDR: Stochastic Gradient Descent with Warm Restarts* </a>

"

### Usage
Sample - (follow similarly for CyclicLinearLR)
milestones specifies when learning rate should shoot back up and decay_milestones when learning rate should be decayed.
```
from cyclicLR import CyclicCosAnnealingLR
import torch

optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)
scheduler = CyclicCosAnnealingLR(optimizer,milestones=[10,25,60,80,120,180,240,320,400,480],decay_milestones=[60, 120, 240, 480, 960],eta_min=1e-6)
for epoch in range(100):
  scheduler.step()
  train(..)
  validate(..)
```
>Note: scheduler.step() shown is called at every epoch. It can be called even in every batch. Remember to specify milestones in number of batches (and not number of epochs) in such as case. For only cyclical lr with no decay, do not pass a decay list. eta_min is the minimum lr it will go to and continue on that once cyclical shedule is over which is by default 1e-6.


### Visualization
Cyclic Cosine Annealing Learning Rate Schedule

![Cosine LR](https://github.com/bluesky314/Cyclical_LR_Scheduler_With_Decay_Pytorch/blob/master/cyc.png)


Cyclic Linear Annealing Learning Rate Schedule

![Linear LR](https://github.com/bluesky314/Cyclical_LR_Scheduler_With_Decay_Pytorch/blob/master/linear.png)
<img src="https://github.com/bluesky314/Cyclical_LR_Scheduler_With_Decay_Pytorch/blob/master/linear.png" alt="alt text" width=100 height=100>
