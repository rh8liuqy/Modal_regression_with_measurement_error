@echo hello world
@CALL "C:\Users\Kevin_Liu\anaconda3\condabin\conda.bat" activate %*
@CALL conda activate env2
@CALL cd /d "C:\Users\Kevin_Liu\OneDrive - University of South Carolina\Research\Beta Measurement Error\paper\simulation\Table3"
@CALL python 19simulation_N200_corrected_variance1_robust.py
@CALL python 20simulation_N200_corrected_variance_unknown_robust.py
@CALL python 21simulation_N200_naive_variance1_robust.py
pause