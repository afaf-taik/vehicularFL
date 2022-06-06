# vehicularFL

Implementation of _"Clustered Vehicular Federated Learning: Process and Optimization"_

This code builds on the implementation of  _Communication-Efficient Learning of Deep Networks from Decentralized Data_ available in [here](https://github.com/AshwinRJ/Federated-Learning-PyTorch) .
The vehicular environment parameters used for the simulation are similar to [Resource allocation for D2D-enabled vehicular communications] (https://ieeexplore.ieee.org/abstract/document/7913583/?casa_token=PSYPrzyfBd0AAAAA:-SKi1BKb0ZVc692haL3bvuAlZoaqF5BHOgwdQ5_0XtMkwPHoaWz_hcqTko426MtS56XQ5JN_zA).


To run the different experiments, choose the desired options in the options file and then run the experiments as follows :

To install the requirements 
``` pip install -r requirements```

To consider mobility only run 
``` python mobility_only.py``` 
To consider mobility and cluster-shift run 
```python vehicular_main.py```
