df1 <- read.csv("01_Table3.csv")
df1 <- df1[,c(3,2,4,1)]
df2 <- read.csv("02_Table3.csv")
df2 <- df2[,c(3,2,4,1)]
df3 <- read.csv("03_Table3.csv")
df3 <- df3[,c(3,2,4,1)]

sapply(df1, function(x){round(median(x),2)})
sapply(df1, function(x){round(IQR(x),2)})

sapply(df2, function(x){round(median(x),2)})
sapply(df2, function(x){round(IQR(x),2)})

sapply(df3, function(x){round(median(x),2)})
sapply(df3, function(x){round(IQR(x),2)})
