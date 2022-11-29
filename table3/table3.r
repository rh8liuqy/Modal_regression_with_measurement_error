library(knitr)
library(kableExtra)
tab_out <- matrix(NA,ncol = 4,nrow = 3)
tab1 <- read.csv("19simulation_N200_corrected_variance1_robust.csv")
tab2 <- read.csv("20simulation_N200_corrected_variance_unknown_robust.csv")
tab3 <- read.csv("21simulation_N200_naive_variance1_robust.csv")

tab_out[1,1] <- paste0(sprintf("%.3f",median(tab1$b0)),"(",sprintf("%.3f",IQR(tab1$b0)),")")
tab_out[1,2] <- paste0(sprintf("%.3f",median(tab1$b1)),"(",sprintf("%.3f",IQR(tab1$b1)),")")
tab_out[1,3] <- paste0(sprintf("%.3f",median(tab1$b2)),"(",sprintf("%.3f",IQR(tab1$b2)),")")
tab_out[1,4] <- paste0(sprintf("%.3f",median(tab1$m)),"(",sprintf("%.3f",IQR(tab1$m)),")")
tab_out[2,1] <- paste0(sprintf("%.3f",median(tab2$b0)),"(",sprintf("%.3f",IQR(tab2$b0)),")")
tab_out[2,2] <- paste0(sprintf("%.3f",median(tab2$b1)),"(",sprintf("%.3f",IQR(tab2$b1)),")")
tab_out[2,3] <- paste0(sprintf("%.3f",median(tab2$b2)),"(",sprintf("%.3f",IQR(tab2$b2)),")")
tab_out[2,4] <- paste0(sprintf("%.3f",median(tab2$m)),"(",sprintf("%.3f",IQR(tab2$m)),")")
tab_out[3,1] <- paste0(sprintf("%.3f",median(tab3$b0)),"(",sprintf("%.3f",IQR(tab3$b0)),")")
tab_out[3,2] <- paste0(sprintf("%.3f",median(tab3$b1)),"(",sprintf("%.3f",IQR(tab3$b1)),")")
tab_out[3,3] <- paste0(sprintf("%.3f",median(tab3$b2)),"(",sprintf("%.3f",IQR(tab3$b2)),")")
tab_out[3,4] <- paste0(sprintf("%.3f",median(tab3$m)),"(",sprintf("%.3f",IQR(tab3$m)),")")
tab_out <- data.frame(tab_out)
#tab_out <- round(tab_out,3)
tab_rowname <- data.frame(col1 = rep("$n=200$",3),
                          col2 = c("MCCL",
                                   "MCCL",
                                   "Naive"),
                          col3 = c("Known",
                                   "Unknown",
                                   ""))
tab_out <- cbind(tab_rowname,tab_out)
saveRDS(tab_out,"table3.rds")
coltext <- c("",
             "",
             "Meas.error var.",
             "$\\beta_0$(IQR)",
             "$\\beta_1$(IQR)",
             "$\\beta_2$(IQR)",
             "$m$(IQR)")
tab <- kable(tab_out,booktabs=TRUE,escape = FALSE,col.names = coltext, caption = "Instead of generating $U_{r}$ from normal distribution, we generate $U_{r} \\sim \\text{Laplace}\\left(0, \\sqrt{3/2}\\right)$ and $\\bar{W}_{r}=X_{1, r} \\mid X_{2, r}+U_{r}$ in the case that measurement error variance is known. We generate $U_{r, j} \\sim \\text{Laplace}\\left(0, \\sqrt{1/2}\\right)$ and $W_{r, j}=X_{1, r} \\mid X_{2, r}+U_{r, j}$ for $j=1,2$, and $3$ in the case that measurement error variance is unknown. $1000$ Monte-Carlo replicates have been generated for exploring robustness of MCCL. The median of estimated values is used for each cell.",
      format = 'latex') %>%
    kable_styling(position = "center") %>%
    collapse_rows(columns = 1:3, latex_hline = "major", valign = "middle")

writeLines(tab, 'tab.tex')