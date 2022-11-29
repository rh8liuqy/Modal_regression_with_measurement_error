@echo hello world
@CALL "C:\Users\Kevin_Liu\anaconda3\condabin\conda.bat" activate %*
@CALL conda activate env2
@CALL cd /d "C:\Users\Kevin_Liu\OneDrive - University of South Carolina\Research\Beta Measurement Error\paper\simulation\Table1"
@CALL python 01simulation_N200_corrected.py
@CALL python 02simulation_N200_naive.py
@CALL python 03simulation_N100_corrected.py
@CALL python 04simulation_N100_naive.py
@CALL python 05simulation_N100_corrected_large_variance.py
@CALL python 06simulation_N100_naive_large_variance.py
@CALL python 07simulation_N200_corrected_large_variance.py
@CALL python 08simulation_N200_naive_large_variance.py
@CALL python 12simulation_N100_corrected_unknown.py
@CALL python 13simulation_N200_corrected_unknown.py
@CALL python 14simulation_N100_corrected_large_variance_unknown.py
@CALL python 15simulation_N200_corrected_large_variance_unknown.py
pause