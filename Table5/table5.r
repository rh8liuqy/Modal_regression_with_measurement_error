df1 <- read.csv("SIMEX_bootstrap.csv")
sapply(df1, function(x){round(sd(x),3)})