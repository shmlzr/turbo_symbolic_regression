Examples
================

Two examples are given:

    - script_simple.py: Performs a symbolic regression using elastic net regularization for a simple problem, i.e.
                        regression of a simple 1D function.
    - script_turbulence.py: Performs a symbolic regression using elastic net regularization for a problem of
                            RANS turbulence modelling.


INPUT.
    - script_simple.py: The target data is produced within the script, no input is needed.
    - script_turbulence.py: Data is loaded from /examples/data/PH10595_frozen_var.p


OUTPUT.
    Both scripts output ...
    - strings of the models saved in /examples/models/<NAME>_model_str.txt
    - figures of the best model results in fig/<NAME>_model_result.png
    - figures of the model structure vs mean-squared error (MSE) in fig/<NAME>_model_vs_mse.png
