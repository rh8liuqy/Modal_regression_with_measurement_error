@echo hello world
@CALL "C:\Users\Kevin_Liu\anaconda3\condabin\conda.bat" activate %*
@CALL conda activate env2
@CALL cd /d "C:\Users\Kevin_Liu\OneDrive - University of South Carolina\Research\Beta Measurement Error\paper\simulation\figure1"
@CALL python 09simulation_N2000_corrected_large_variance.py
@CALL python 10simulation_N2000_naive_large_variance.py
@CALL python 11simulation_N2000_corrected_large_variance_unknown.py
@CALL python 16simulation_N2000_corrected_large_variance_ind.py
@CALL python 17simulation_N2000_naive_large_variance_ind.py
@CALL python 18simulation_N2000_corrected_large_variance_unknown.py
pause