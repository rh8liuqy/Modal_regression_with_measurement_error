README

###############################################################################
Source code and data for the manuscript "Parametric Modal Regression with Error
in Covariates”, by Qingyang Liu and Xianzheng Huang.
###############################################################################

For questions, comments or remarks about the code please contact Qingyang Liu (qingyang@email.sc.edu).

Functions for fitting parametric modal regression with error in covariates are provided in the file ./Beta_reg.py.

Models m1 - m4 and all data generation codes are provided in ./simulation_setting.py.

The files with extension .sh are SLURM scripts.

Folders Figure1, Figure2, Figure3, Table1, Table2, Table3, Table4, Table5, Table6 contain intermediate results and R codes for creating plots or tables.

To reproduce the results presented in the manuscript (i.e Table 1 - Table 6 and Figure 1 - Figure 3), please run the SLURM scripts if possible, otherwise run the python scripts.

The python version information can be found in ./environment.yml and ./Table6/Table6.yml.
.
├── Beta_reg.py #Functions for fitting parametric modal regression with error in covariates
├── BJadni.csv #Alzheimer’s disease data in Section 6.2, downloaded from Alzheimer’s Disease Neuroimaging Initiative (ADNI) database (http://adni.loni.usc.edu/). 
├── environment.yml #Version information
├── Figure1 #Intermediate results
│   ├── 01_Figure1.csv
│   ├── 02_Figure1.csv
│   ├── 03_Figure1.csv
│   ├── 04_Figure1.csv
│   ├── FFQ-1.pdf
│   └── Figure1.R #R script to generate Figure 1 using intermediate results
├── Figure1.py
├── Figure1.sh
├── Figure2 #Intermediate results
│   ├── diagnosis-1.pdf
│   ├── Figure2.R #R script to generate Figure 2 using intermediate results
│   ├── Figure2_n1000seed1001_1250.csv
│   ├── Figure2_n1000seed1251_1500.csv
│   ├── Figure2_n1000seed1501_1750.csv
│   ├── Figure2_n1000seed1751_2000.csv
│   ├── Figure2_n1000seed1_250.csv
│   ├── Figure2_n1000seed2001_2250.csv
│   ├── Figure2_n1000seed2251_2500.csv
│   ├── Figure2_n1000seed2501_2750.csv
│   ├── Figure2_n1000seed251_500.csv
│   ├── Figure2_n1000seed2751_3000.csv
│   ├── Figure2_n1000seed3001_3250.csv
│   ├── Figure2_n1000seed3251_3500.csv
│   ├── Figure2_n1000seed3501_3750.csv
│   ├── Figure2_n1000seed3751_4000.csv
│   ├── Figure2_n1000seed4001_4250.csv
│   ├── Figure2_n1000seed4251_4500.csv
│   ├── Figure2_n1000seed4501_4750.csv
│   ├── Figure2_n1000seed4751_5000.csv
│   ├── Figure2_n1000seed501_750.csv
│   ├── Figure2_n1000seed751_1000.csv
│   ├── Figure2_n100seed1001_1250.csv
│   ├── Figure2_n100seed1251_1500.csv
│   ├── Figure2_n100seed1501_1750.csv
│   ├── Figure2_n100seed1751_2000.csv
│   ├── Figure2_n100seed1_250.csv
│   ├── Figure2_n100seed2001_2250.csv
│   ├── Figure2_n100seed2251_2500.csv
│   ├── Figure2_n100seed2501_2750.csv
│   ├── Figure2_n100seed251_500.csv
│   ├── Figure2_n100seed2751_3000.csv
│   ├── Figure2_n100seed3001_3250.csv
│   ├── Figure2_n100seed3251_3500.csv
│   ├── Figure2_n100seed3501_3750.csv
│   ├── Figure2_n100seed3751_4000.csv
│   ├── Figure2_n100seed4001_4250.csv
│   ├── Figure2_n100seed4251_4500.csv
│   ├── Figure2_n100seed4501_4750.csv
│   ├── Figure2_n100seed4751_5000.csv
│   ├── Figure2_n100seed501_750.csv
│   ├── Figure2_n100seed751_1000.csv
│   ├── Figure2_n200seed1001_1250.csv
│   ├── Figure2_n200seed1251_1500.csv
│   ├── Figure2_n200seed1501_1750.csv
│   ├── Figure2_n200seed1751_2000.csv
│   ├── Figure2_n200seed1_250.csv
│   ├── Figure2_n200seed2001_2250.csv
│   ├── Figure2_n200seed2251_2500.csv
│   ├── Figure2_n200seed2501_2750.csv
│   ├── Figure2_n200seed251_500.csv
│   ├── Figure2_n200seed2751_3000.csv
│   ├── Figure2_n200seed3001_3250.csv
│   ├── Figure2_n200seed3251_3500.csv
│   ├── Figure2_n200seed3501_3750.csv
│   ├── Figure2_n200seed3751_4000.csv
│   ├── Figure2_n200seed4001_4250.csv
│   ├── Figure2_n200seed4251_4500.csv
│   ├── Figure2_n200seed4501_4750.csv
│   ├── Figure2_n200seed4751_5000.csv
│   ├── Figure2_n200seed501_750.csv
│   ├── Figure2_n200seed751_1000.csv
│   ├── Figure2_n500seed1001_1250.csv
│   ├── Figure2_n500seed1251_1500.csv
│   ├── Figure2_n500seed1501_1750.csv
│   ├── Figure2_n500seed1751_2000.csv
│   ├── Figure2_n500seed1_250.csv
│   ├── Figure2_n500seed2001_2250.csv
│   ├── Figure2_n500seed2251_2500.csv
│   ├── Figure2_n500seed2501_2750.csv
│   ├── Figure2_n500seed251_500.csv
│   ├── Figure2_n500seed2751_3000.csv
│   ├── Figure2_n500seed3001_3250.csv
│   ├── Figure2_n500seed3251_3500.csv
│   ├── Figure2_n500seed3501_3750.csv
│   ├── Figure2_n500seed3751_4000.csv
│   ├── Figure2_n500seed4001_4250.csv
│   ├── Figure2_n500seed4251_4500.csv
│   ├── Figure2_n500seed4501_4750.csv
│   ├── Figure2_n500seed4751_5000.csv
│   ├── Figure2_n500seed501_750.csv
│   └── Figure2_n500seed751_1000.csv
├── Figure2.py
├── Figure2.sh
├── Figure3 #Intermediate results (Run ./Table6/Table6.py to get the intermediate results)
│   ├── data_plot_wishreg.csv
│   ├── figure3.pdf
│   └── Figure3.R #R script to generate Figure 3 using intermediate results
├── README.txt
├── simulation_setting.py
├── Table1 #Intermediate results
│   ├── 01_Table1.csv
│   ├── 02_Table1.csv
│   ├── 03_Table1.csv
│   ├── 04_Table1.csv
│   ├── 05_Table1.csv
│   ├── 06_Table1.csv
│   ├── 07_Table1.csv
│   ├── 08_Table1.csv
│   ├── 09_Table1.csv
│   ├── 10_Table1.csv
│   ├── 11_Table1.csv
│   ├── 12_Table1.csv
│   └── Table1.r #R script to generate Table 1 using intermediate results
├── Table1.py
├── Table1.sh
├── Table2 #Intermediate results
│   ├── 01_Table2.csv
│   ├── 02_Table2.csv
│   └── Table2.r #R script to generate Table 2 using intermediate results
├── Table2.py
├── Table2.sh
├── Table3 #Intermediate results
│   ├── 01_Table3.csv
│   ├── 02_Table3.csv
│   ├── 03_Table3.csv
│   └── Table3.R #R script to generate Table 3 using intermediate results
├── Table3.py
├── Table3.sh
├── Table4 #Intermediate results
│   ├── Table4.r #R script to generate Table 4 using intermediate results
│   ├── Table4_modelm2n200seed121_150.csv
│   ├── Table4_modelm2n200seed151_180.csv
│   ├── Table4_modelm2n200seed181_210.csv
│   ├── Table4_modelm2n200seed1_30.csv
│   ├── Table4_modelm2n200seed211_240.csv
│   ├── Table4_modelm2n200seed241_270.csv
│   ├── Table4_modelm2n200seed271_300.csv
│   ├── Table4_modelm2n200seed31_60.csv
│   ├── Table4_modelm2n200seed61_90.csv
│   ├── Table4_modelm2n200seed91_120.csv
│   ├── Table4_modelm2n300seed121_150.csv
│   ├── Table4_modelm2n300seed151_180.csv
│   ├── Table4_modelm2n300seed181_210.csv
│   ├── Table4_modelm2n300seed1_30.csv
│   ├── Table4_modelm2n300seed211_240.csv
│   ├── Table4_modelm2n300seed241_270.csv
│   ├── Table4_modelm2n300seed271_300.csv
│   ├── Table4_modelm2n300seed31_60.csv
│   ├── Table4_modelm2n300seed61_90.csv
│   ├── Table4_modelm2n300seed91_120.csv
│   ├── Table4_modelm2n400seed121_150.csv
│   ├── Table4_modelm2n400seed151_180.csv
│   ├── Table4_modelm2n400seed181_210.csv
│   ├── Table4_modelm2n400seed1_30.csv
│   ├── Table4_modelm2n400seed211_240.csv
│   ├── Table4_modelm2n400seed241_270.csv
│   ├── Table4_modelm2n400seed271_300.csv
│   ├── Table4_modelm2n400seed31_60.csv
│   ├── Table4_modelm2n400seed61_90.csv
│   ├── Table4_modelm2n400seed91_120.csv
│   ├── Table4_modelm2n500seed121_150.csv
│   ├── Table4_modelm2n500seed151_180.csv
│   ├── Table4_modelm2n500seed181_210.csv
│   ├── Table4_modelm2n500seed1_30.csv
│   ├── Table4_modelm2n500seed211_240.csv
│   ├── Table4_modelm2n500seed241_270.csv
│   ├── Table4_modelm2n500seed271_300.csv
│   ├── Table4_modelm2n500seed31_60.csv
│   ├── Table4_modelm2n500seed61_90.csv
│   ├── Table4_modelm2n500seed91_120.csv
│   ├── Table4_modelm3n200seed121_150.csv
│   ├── Table4_modelm3n200seed151_180.csv
│   ├── Table4_modelm3n200seed181_210.csv
│   ├── Table4_modelm3n200seed1_30.csv
│   ├── Table4_modelm3n200seed211_240.csv
│   ├── Table4_modelm3n200seed241_270.csv
│   ├── Table4_modelm3n200seed271_300.csv
│   ├── Table4_modelm3n200seed31_60.csv
│   ├── Table4_modelm3n200seed61_90.csv
│   ├── Table4_modelm3n200seed91_120.csv
│   ├── Table4_modelm3n300seed121_150.csv
│   ├── Table4_modelm3n300seed151_180.csv
│   ├── Table4_modelm3n300seed181_210.csv
│   ├── Table4_modelm3n300seed1_30.csv
│   ├── Table4_modelm3n300seed211_240.csv
│   ├── Table4_modelm3n300seed241_270.csv
│   ├── Table4_modelm3n300seed271_300.csv
│   ├── Table4_modelm3n300seed31_60.csv
│   ├── Table4_modelm3n300seed61_90.csv
│   ├── Table4_modelm3n300seed91_120.csv
│   ├── Table4_modelm3n400seed121_150.csv
│   ├── Table4_modelm3n400seed151_180.csv
│   ├── Table4_modelm3n400seed181_210.csv
│   ├── Table4_modelm3n400seed1_30.csv
│   ├── Table4_modelm3n400seed211_240.csv
│   ├── Table4_modelm3n400seed241_270.csv
│   ├── Table4_modelm3n400seed271_300.csv
│   ├── Table4_modelm3n400seed31_60.csv
│   ├── Table4_modelm3n400seed61_90.csv
│   ├── Table4_modelm3n400seed91_120.csv
│   ├── Table4_modelm3n500seed121_150.csv
│   ├── Table4_modelm3n500seed151_180.csv
│   ├── Table4_modelm3n500seed181_210.csv
│   ├── Table4_modelm3n500seed1_30.csv
│   ├── Table4_modelm3n500seed211_240.csv
│   ├── Table4_modelm3n500seed241_270.csv
│   ├── Table4_modelm3n500seed271_300.csv
│   ├── Table4_modelm3n500seed31_60.csv
│   ├── Table4_modelm3n500seed61_90.csv
│   ├── Table4_modelm3n500seed91_120.csv
│   ├── Table4_modelm4n200seed121_150.csv
│   ├── Table4_modelm4n200seed151_180.csv
│   ├── Table4_modelm4n200seed181_210.csv
│   ├── Table4_modelm4n200seed1_30.csv
│   ├── Table4_modelm4n200seed211_240.csv
│   ├── Table4_modelm4n200seed241_270.csv
│   ├── Table4_modelm4n200seed271_300.csv
│   ├── Table4_modelm4n200seed31_60.csv
│   ├── Table4_modelm4n200seed61_90.csv
│   ├── Table4_modelm4n200seed91_120.csv
│   ├── Table4_modelm4n300seed121_150.csv
│   ├── Table4_modelm4n300seed151_180.csv
│   ├── Table4_modelm4n300seed181_210.csv
│   ├── Table4_modelm4n300seed1_30.csv
│   ├── Table4_modelm4n300seed211_240.csv
│   ├── Table4_modelm4n300seed241_270.csv
│   ├── Table4_modelm4n300seed271_300.csv
│   ├── Table4_modelm4n300seed31_60.csv
│   ├── Table4_modelm4n300seed61_90.csv
│   ├── Table4_modelm4n300seed91_120.csv
│   ├── Table4_modelm4n400seed121_150.csv
│   ├── Table4_modelm4n400seed151_180.csv
│   ├── Table4_modelm4n400seed181_210.csv
│   ├── Table4_modelm4n400seed1_30.csv
│   ├── Table4_modelm4n400seed211_240.csv
│   ├── Table4_modelm4n400seed241_270.csv
│   ├── Table4_modelm4n400seed271_300.csv
│   ├── Table4_modelm4n400seed31_60.csv
│   ├── Table4_modelm4n400seed61_90.csv
│   ├── Table4_modelm4n400seed91_120.csv
│   ├── Table4_modelm4n500seed121_150.csv
│   ├── Table4_modelm4n500seed151_180.csv
│   ├── Table4_modelm4n500seed181_210.csv
│   ├── Table4_modelm4n500seed1_30.csv
│   ├── Table4_modelm4n500seed211_240.csv
│   ├── Table4_modelm4n500seed241_270.csv
│   ├── Table4_modelm4n500seed271_300.csv
│   ├── Table4_modelm4n500seed31_60.csv
│   ├── Table4_modelm4n500seed61_90.csv
│   └── Table4_modelm4n500seed91_120.csv
├── Table4.py
├── Table4.sh
├── Table5 #Intermediate results
│   ├── SIMEX_bootstrap.csv
│   ├── Table5.out
│   └── Table5.R #R script to generate Table 5 using intermediate results
├── Table5.py
├── Table5.sh
├── Table6
│   ├── Table6.py #R script to generate Table 6
│   └── Table6.yml #Version information of codes for Table 6
└── wishreg.csv #Dietary data in Section 6.1, from "Design aspects of calibration studies in nutrition, with analysis of missing data in linear measurement error models", Carroll, R. J., Freedman, L., and Pee, D. (1997).
