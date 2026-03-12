# ML for iCLIP
CHEMENG 277 Final Project W26

This repository contains code for processing computational fluid dynamics (CFD) simulation data for injection CLIP (iCLIP) printing and developing machine learning models to predict dead volume fraction in porous structures.

- runs.csv is the list of initial simulation runs

- extract_dead_volume.py is used to process the OpenFOAM simulation results and extract the volume fraction of cells below a critical velocity (1e-05 m/s).

- This data is stored in dead_volume_combined.csv, where dead volumes from both the original and additional simulation cases (selected using uncertainty-based sampling) are contained. Each row corresponds to the results from one simulation, with geometric/process parameters detailed in the columns 

- ridge_learning_curve.py was used to perform the initial ridge regression on the original 104 point dataset. This script also tests model performance as a function of dataset size.

- active_learning.py was used to implement uncertainty-based active learning using a bootstrap ensemble from the ridge regression model developed from the initial data set.

- The outputs from this script that were submitted for the additional 20 simulations are stored in runs_active.csv.
 
-  



