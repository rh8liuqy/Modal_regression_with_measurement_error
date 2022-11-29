import sys
sys.path.append('C:\\Users\\Kevin_Liu\\OneDrive - University of South Carolina\\Research\\Beta Measurement Error\\paper\\simulation')
import pybetareg_data_analysis as pyb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(100)

df1 = pd.read_csv("wishreg.csv")
df1["ffq"] = df1["ffq"]/(8000)
df1["fr1"] = (df1["fr1"] - np.mean(df1["fr1"]))/np.sqrt(np.var(df1["fr1"]))
df1["fr2"] = (df1["fr2"] - np.mean(df1["fr2"]))/np.sqrt(np.var(df1["fr2"]))
df1["fr3"] = (df1["fr3"] - np.mean(df1["fr3"]))/np.sqrt(np.var(df1["fr3"]))
df1["fr4"] = (df1["fr4"] - np.mean(df1["fr4"]))/np.sqrt(np.var(df1["fr4"]))
df1["fr5"] = (df1["fr5"] - np.mean(df1["fr5"]))/np.sqrt(np.var(df1["fr5"]))
df1["fr6"] = (df1["fr6"] - np.mean(df1["fr6"]))/np.sqrt(np.var(df1["fr6"]))

y = df1["ffq"].to_numpy()
w = (df1["fr1"]+df1["fr2"]+df1["fr3"]+df1["fr4"]+df1["fr5"]+df1["fr6"])/6
w = w.to_numpy()
z = np.ones(w.shape[0])
sigmaw = np.zeros(w.shape[0])
for i in range(w.shape[0]):
    temp = df1.iloc[i,1:7].to_numpy()
    sigmaw[i] = np.std(temp,ddof=1)

x = np.column_stack((z,w))

model1 = pyb.reg(x=x, y=y, initial = [np.log(3),-1,1])
model1fit = model1.fit()
model1fit.summary()

model2 = pyb.reg_measurement_error(y=y,w=w,z=z,
                                   sigmaw=sigmaw,
                                   repeated_measure = 6,
                                   initial=[np.log(3.1),1.22,-1.61],
                                   CUDA = True,
                                   column_names = ['beta1','beta0'])
model2fit = model2.fit()
model2fit.summary()
#output = model2.hotelling_bootstrap(300)
#print(output)
#p-value = 0.61

df_out = pd.DataFrame({"y":y,"w":w})
df_out.to_csv("../figure3_update/data_plot_wishreg.csv",index=False)