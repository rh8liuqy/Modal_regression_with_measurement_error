df1 <- read.csv("01_Table2.csv")
sapply(df1[,5:8], function(x){round(mean(x),2)})[c(3,2,4,1)]
sapply(df1[,5:8], function(x){round(sd(x),2)})[c(3,2,4,1)]
sapply(df1[,1:4], function(x){round(sd(x),2)})[c(3,2,4,1)]

df2 <- read.csv("02_Table2.csv")
sapply(df2[,5:8], function(x){round(mean(x),2)})[c(3,2,4,1)]
sapply(df2[,5:8], function(x){round(sd(x),2)})[c(3,2,4,1)]
sapply(df2[,1:4], function(x){round(sd(x),2)})[c(3,2,4,1)]
