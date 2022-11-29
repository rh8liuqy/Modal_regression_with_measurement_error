library(knitr)
library(kableExtra)
df_out <- data.frame(Model = c("(V1)","(V2)","(V3)","(V4)"),
                     n_200 = NA,
                     n_300 = NA,
                     n_400 = NA,
                     n_500 = NA)
df1 <- read.csv("36simulation_N200_bootstrap_unknown_variance_linear.csv")
df2 <- read.csv("37simulation_N300_bootstrap_unknown_variance_linear.csv")
df3 <- read.csv("38simulation_N400_bootstrap_unknown_variance_linear.csv")
df4 <- read.csv("39simulation_N500_bootstrap_unknown_variance_linear.csv")
df5 <- read.csv("44simulation_N200_bootstrap_unknown_variance_cloglog.csv")
df6 <- read.csv("45simulation_N300_bootstrap_unknown_variance_cloglog.csv")
df7 <- read.csv("46simulation_N400_bootstrap_unknown_variance_cloglog.csv")
df8 <- read.csv("47simulation_N500_bootstrap_unknown_variance_cloglog.csv")
df9 <- read.csv("54simulation_N200_bootstrap_unknown_variance_probit.csv")
df10 <- read.csv("55simulation_N300_bootstrap_unknown_variance_probit.csv")
df11 <- read.csv("56simulation_N400_bootstrap_unknown_variance_probit.csv")
df12 <- read.csv("57simulation_N500_bootstrap_unknown_variance_probit.csv")
df13 <- read.csv("74simulation_N200_bootstrap_unknown_variance_gumbel.csv")
df14 <- read.csv("75simulation_N300_bootstrap_unknown_variance_gumbel.csv")
df15 <- read.csv("76simulation_N400_bootstrap_unknown_variance_gumbel.csv")
df16 <- read.csv("77simulation_N500_bootstrap_unknown_variance_gumbel.csv")

df_out[1,2] <- sprintf("%.4f",mean(df1$p.value <= 0.05,na.rm = TRUE))
df_out[1,3] <- sprintf("%.4f",mean(df2$p.value <= 0.05,na.rm = TRUE))
df_out[1,4] <- sprintf("%.4f",mean(df3$p.value <= 0.05,na.rm = TRUE))
df_out[1,5] <- sprintf("%.4f",mean(df4$p.value <= 0.05,na.rm = TRUE))

df_out[2,2] <- sprintf("%.4f",mean(df5$p.value <= 0.05,na.rm = TRUE))
df_out[2,3] <- sprintf("%.4f",mean(df6$p.value <= 0.05,na.rm = TRUE))
df_out[2,4] <- sprintf("%.4f",mean(df7$p.value <= 0.05,na.rm = TRUE))
df_out[2,5] <- sprintf("%.4f",mean(df8$p.value <= 0.05,na.rm = TRUE))

df_out[3,2] <- sprintf("%.4f",mean(df9$p.value <= 0.05,na.rm = TRUE))
df_out[3,3] <- sprintf("%.4f",mean(df10$p.value <= 0.05,na.rm = TRUE))
df_out[3,4] <- sprintf("%.4f",mean(df11$p.value <= 0.05,na.rm = TRUE))
df_out[3,5] <- sprintf("%.4f",mean(df12$p.value <= 0.05,na.rm = TRUE))

df_out[4,2] <- sprintf("%.4f",mean(df13$p.value <= 0.05,na.rm = TRUE))
df_out[4,3] <- sprintf("%.4f",mean(df14$p.value <= 0.05,na.rm = TRUE))
df_out[4,4] <- sprintf("%.4f",mean(df15$p.value <= 0.05,na.rm = TRUE))
df_out[4,5] <- sprintf("%.4f",mean(df16$p.value <= 0.05,na.rm = TRUE))

saveRDS(df_out,"table4.rds")

kable(df_out,booktabs=TRUE,escape = FALSE,col.names = c("Model","$n=200$","$n=300$","$n=400$","$n=500$"),
      caption = "Power of parametric bootstrap based score test at four different cases. Case (V1) represents linear relationship violation. Case (V2) and (V3) represent link function misspecification. Case (V4) represents distributional assumption violation. For all cases, $300$ Monte Carlo replicates have been generated.") %>%
    kable_styling(position = "center")