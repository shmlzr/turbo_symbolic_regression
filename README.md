
Turbo Symbolic Regression
================
Tools for symbolic regression to discover turbulence models from data.


Two examples are given:

    - script_simple.py: Performs a symbolic regression using elastic net regularization for a simple problem, i.e.
                        regression of a simple 1D function.
    - script_turbulence.py: Performs a symbolic regression using elastic net regularization for a problem of
                            RANS turbulence modelling.


INPUT.

    - script_simple.py: The target data is produced within the script, no input is needed.  
    - script_turbulence.py: Data is loaded from /data/PH10595_frozen_var.p


OUTPUT.

    Both scripts output ...
    - strings of the models saved in /data/models/<NAME>_model_str.txt
    - figures of the best model results in /data/fig/<NAME>_model_result.png
    - figures of the model structure vs mean-squared error (MSE) in /data/fig/<NAME>_model_vs_mse.png


## Relevant Publications

Martin Schmelzer, Richard P. Dwight, Paola Cinnella: **Discovery of Algebraic Reynolds-Stress Models Using Sparse Symbolic Regression.** 
Flow, Turbulence and Combustion 104, 579–603, 2020
https://doi.org/10.1007/s10494-019-00089-x

Ismaïl Ben Hassan Saïdi, Martin Schmelzer, Paola Cinnella, Francesco Grasso: **CFD-driven symbolic identification of algebraic Reynolds-stress models.**
Journal of Computational Physics, Volume 457, 2022
https://doi.org/10.1016/j.jcp.2022.111037

Yu Zhang, Richard P. Dwight, Martin Schmelzer, Javier F. Gómez, Zhong-hua Han, Stefan Hickel: **Customized data-driven RANS closures for bi-fidelity LES–RANS optimization.** Journal of Computational Physics, Volume 432, 2021
https://doi.org/10.1016/j.jcp.2021.110153

Jasper P. Huijing, Richard P. Dwight, Martin Schmelzer: **Data-driven RANS closures for three-dimensional flows around bluff bodies.**
Computers & Fluids, Volume 225, 2021
https://doi.org/10.1016/j.compfluid.2021.104997

Further applications:
https://www.researchgate.net/profile/Martin-Schmelzer


##
2021, Martin Schmelzer, m.schmelzer@tudelft.nl
