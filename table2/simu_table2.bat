@echo hello world
@CALL "C:\Users\Kevin_Liu\anaconda3\condabin\conda.bat" activate %*
@CALL conda activate env2
@CALL cd /d "C:\Users\Kevin_Liu\OneDrive - University of South Carolina\Research\Beta Measurement Error\paper\simulation\Table2"
@CALL python 07simulation_N200_corrected_large_variance.py
@CALL python 08simulation_N200_naive_large_variance.py
@CALL python 15simulation_N200_corrected_large_variance_unknown.py
pause