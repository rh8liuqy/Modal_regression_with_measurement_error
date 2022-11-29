library(tidyverse)
library(latticeExtra)
library(gridExtra)
library(latex2exp)
library(kableExtra)
df_tab <- data.frame(Method = rep(c("Naive","MCCL"),each=3),
                     Effect = c("Intercept",
                                "Long Term Intake",
                                "log($m$)",
                                "Intercept",
                                "Long Term Intake",
                                "log($m$)"
                     ),
                     Estimate = c(-1.8243,0.6830,0.9760,
                                  -1.5248,1.4421,1.2243),
                     SE = c(0.210,0.217,0.137,
                            0.258,0.625,1.106),
                     p_value = c(sprintf("%.3f",0),
                                 sprintf("%.3f",0.002),
                                 sprintf("%.3f",0.000),
                                 sprintf("%.3f",0.000),
                                 sprintf("%.3f",0.021),
                                 sprintf("%.3f",0.268)),
                     lower = c(-2.236,0.258,0.707,
                               -2.031,0.216,-0.943),
                     upper = c(-1.412,1.108,1.245,
                               -1.019,2.668,3.392))
saveRDS(df_tab,"table5.rds")
coltxt <- colnames(df_tab)
coltxt[5] <- "$p$-value"
coltxt[6] <- "$95\\%$ Lower CL"
coltxt[7] <- "$95\\%$ Upper CL"
kable(df_tab,booktabs = TRUE,escape = FALSE,col.names = coltxt,
      caption = "Application of beta modal regression with logit link to dietary data. SE stands for standard error. CL stands for confidence limit.") %>%
    kable_styling(position = "center") %>%
    collapse_rows(columns = 1, latex_hline = "major", valign = "middle")
