library(knitr)
library(kableExtra)
tab_out <- matrix(NA,ncol = 8,nrow = 6)
tab1 <- read.csv("03simulation_N100_corrected.csv")
tab2 <- read.csv("12simulation_N100_corrected_unknown.csv")
tab3 <- read.csv("04simulation_N100_naive.csv")
tab4 <- read.csv("01simulation_N200_corrected.csv")
tab5 <- read.csv("13simulation_N200_corrected_unknown.csv")
tab6 <- read.csv("02simulation_N200_naive.csv")
tab_out[1,1] <- paste0(sprintf("%.2f",median(tab1$b0)),"(",sprintf("%.2f",IQR(tab1$b0)),")")
tab_out[1,2] <- paste0(sprintf("%.2f",median(tab1$b1)),"(",sprintf("%.2f",IQR(tab1$b1)),")")
tab_out[1,3] <- paste0(sprintf("%.2f",median(tab1$b2)),"(",sprintf("%.2f",IQR(tab1$b2)),")")
tab_out[1,4] <- median(tab1$m)
tab_out[2,1] <- paste0(sprintf("%.2f",median(tab2$b0)),"(",sprintf("%.2f",IQR(tab2$b0)),")")
tab_out[2,2] <- paste0(sprintf("%.2f",median(tab2$b1)),"(",sprintf("%.2f",IQR(tab2$b1)),")")
tab_out[2,3] <- paste0(sprintf("%.2f",median(tab2$b2)),"(",sprintf("%.2f",IQR(tab2$b2)),")")
tab_out[2,4] <- median(tab2$m)
tab_out[3,1] <- paste0(sprintf("%.2f",median(tab3$b0)),"(",sprintf("%.2f",IQR(tab3$b0)),")")
tab_out[3,2] <- paste0(sprintf("%.2f",median(tab3$b1)),"(",sprintf("%.2f",IQR(tab3$b1)),")")
tab_out[3,3] <- paste0(sprintf("%.2f",median(tab3$b2)),"(",sprintf("%.2f",IQR(tab3$b2)),")")
tab_out[3,4] <- median(tab3$m)
tab_out[4,1] <- paste0(sprintf("%.2f",median(tab4$b0)),"(",sprintf("%.2f",IQR(tab4$b0)),")")
tab_out[4,2] <- paste0(sprintf("%.2f",median(tab4$b1)),"(",sprintf("%.2f",IQR(tab4$b1)),")")
tab_out[4,3] <- paste0(sprintf("%.2f",median(tab4$b2)),"(",sprintf("%.2f",IQR(tab4$b2)),")")
tab_out[4,4] <- median(tab4$m)
tab_out[5,1] <- paste0(sprintf("%.2f",median(tab5$b0)),"(",sprintf("%.2f",IQR(tab5$b0)),")")
tab_out[5,2] <- paste0(sprintf("%.2f",median(tab5$b1)),"(",sprintf("%.2f",IQR(tab5$b1)),")")
tab_out[5,3] <- paste0(sprintf("%.2f",median(tab5$b2)),"(",sprintf("%.2f",IQR(tab5$b2)),")")
tab_out[5,4] <- median(tab5$m)
tab_out[6,1] <- paste0(sprintf("%.2f",median(tab6$b0)),"(",sprintf("%.2f",IQR(tab6$b0)),")")
tab_out[6,2] <- paste0(sprintf("%.2f",median(tab6$b1)),"(",sprintf("%.2f",IQR(tab6$b1)),")")
tab_out[6,3] <- paste0(sprintf("%.2f",median(tab6$b2)),"(",sprintf("%.2f",IQR(tab6$b2)),")")
tab_out[6,4] <- median(tab6$m)

tab7 <- read.csv("05simulation_N100_corrected_large_variance.csv")
tab8 <- read.csv("14simulation_N100_corrected_large_variance_unknown.csv")
tab9 <- read.csv("06simulation_N100_naive_large_variance.csv")
tab10 <- read.csv("07simulation_N200_corrected_large_variance.csv")
tab11 <- read.csv("15simulation_N200_corrected_large_variance_unknown.csv")
tab12 <- read.csv("08simulation_N200_naive_large_variance.csv")
tab_out[1,5] <- paste0(sprintf("%.2f",median(tab7$b0)),"(",sprintf("%.2f",IQR(tab7$b0)),")")
tab_out[1,6] <- paste0(sprintf("%.2f",median(tab7$b1)),"(",sprintf("%.2f",IQR(tab7$b1)),")")
tab_out[1,7] <- paste0(sprintf("%.2f",median(tab7$b2)),"(",sprintf("%.2f",IQR(tab7$b2)),")")
tab_out[1,8] <- median(tab7$m)
tab_out[2,5] <- paste0(sprintf("%.2f",median(tab8$b0)),"(",sprintf("%.2f",IQR(tab8$b0)),")")
tab_out[2,6] <- paste0(sprintf("%.2f",median(tab8$b1)),"(",sprintf("%.2f",IQR(tab8$b1)),")")
tab_out[2,7] <- paste0(sprintf("%.2f",median(tab8$b2)),"(",sprintf("%.2f",IQR(tab8$b2)),")")
tab_out[2,8] <- median(tab8$m)
tab_out[3,5] <- paste0(sprintf("%.2f",median(tab9$b0)),"(",sprintf("%.2f",IQR(tab9$b0)),")")
tab_out[3,6] <- paste0(sprintf("%.2f",median(tab9$b1)),"(",sprintf("%.2f",IQR(tab9$b1)),")")
tab_out[3,7] <- paste0(sprintf("%.2f",median(tab9$b2)),"(",sprintf("%.2f",IQR(tab9$b2)),")")
tab_out[3,8] <- median(tab9$m)
tab_out[4,5] <- paste0(sprintf("%.2f",median(tab10$b0)),"(",sprintf("%.2f",IQR(tab10$b0)),")")
tab_out[4,6] <- paste0(sprintf("%.2f",median(tab10$b1)),"(",sprintf("%.2f",IQR(tab10$b1)),")")
tab_out[4,7] <- paste0(sprintf("%.2f",median(tab10$b2)),"(",sprintf("%.2f",IQR(tab10$b2)),")")
tab_out[4,8] <- median(tab10$m)
tab_out[5,5] <- paste0(sprintf("%.2f",median(tab11$b0)),"(",sprintf("%.2f",IQR(tab11$b0)),")")
tab_out[5,6] <- paste0(sprintf("%.2f",median(tab11$b1)),"(",sprintf("%.2f",IQR(tab11$b1)),")")
tab_out[5,7] <- paste0(sprintf("%.2f",median(tab11$b2)),"(",sprintf("%.2f",IQR(tab11$b2)),")")
tab_out[5,8] <- median(tab11$m)
tab_out[6,5] <- paste0(sprintf("%.2f",median(tab12$b0)),"(",sprintf("%.2f",IQR(tab12$b0)),")")
tab_out[6,6] <- paste0(sprintf("%.2f",median(tab12$b1)),"(",sprintf("%.2f",IQR(tab12$b1)),")")
tab_out[6,7] <- paste0(sprintf("%.2f",median(tab12$b2)),"(",sprintf("%.2f",IQR(tab12$b2)),")")
tab_out[6,8] <- median(tab12$m)

#tab_out <- round(tab_out,2)
tab_out <- data.frame(tab_out)
tab_rowname <- data.frame(col1 = rep(c("$n=100$",
                                       "$n=200$"),each=3),
                          col2 = rep(c("MCCL",
                                       "MCCL",
                                       "Naive"),2),
                          col3 = rep(c("Known",
                                       "Unknown",
                                       ""),2))
tab_out_kable <- cbind(tab_rowname,tab_out)
coltext <- c("",
             "",
             "Meas.error var.",
             rep(c("$\\beta_0$(IQR)",
                   "$\\beta_1$(IQR)",
                   "$\\beta_2$(IQR)",
                   "$m$(IQR)"),2))
tab_out_kable[1,7] <- paste0(sprintf("%.2f",as.numeric(tab_out_kable[1,7])),"(",round(IQR(tab1$m),2),")")
tab_out_kable[2,7] <- paste0(sprintf("%.2f",as.numeric(tab_out_kable[2,7])),"(",round(IQR(tab2$m),2),")")
tab_out_kable[3,7] <- paste0(sprintf("%.2f",as.numeric(tab_out_kable[3,7])),"(",sprintf("%.2f",round(IQR(tab3$m),2)),")")
tab_out_kable[4,7] <- paste0(sprintf("%.2f",as.numeric(tab_out_kable[4,7])),"(",sprintf("%.2f",round(IQR(tab4$m),2)),")")
tab_out_kable[5,7] <- paste0(sprintf("%.2f",as.numeric(tab_out_kable[5,7])),"(",sprintf("%.2f",round(IQR(tab5$m),2)),")")
tab_out_kable[6,7] <- paste0(sprintf("%.2f",as.numeric(tab_out_kable[6,7])),"(",sprintf("%.2f",round(IQR(tab6$m),2)),")")
tab_out_kable[1,11] <- paste0(sprintf("%.2f",as.numeric(tab_out_kable[1,11])),"(",sprintf("%.2f",round(IQR(tab7$m),2)),")")
tab_out_kable[2,11] <- paste0(sprintf("%.2f",as.numeric(tab_out_kable[2,11])),"(",sprintf("%.2f",round(IQR(tab8$m),2)),")")
tab_out_kable[3,11] <- paste0(sprintf("%.2f",as.numeric(tab_out_kable[3,11])),"(",sprintf("%.2f",round(IQR(tab9$m),2)),")")
tab_out_kable[4,11] <- paste0(sprintf("%.2f",as.numeric(tab_out_kable[4,11])),"(",sprintf("%.2f",round(IQR(tab10$m),2)),")")
tab_out_kable[5,11] <- paste0(sprintf("%.2f",as.numeric(tab_out_kable[5,11])),"(",sprintf("%.2f",round(IQR(tab11$m),2)),")")
tab_out_kable[6,11] <- paste0(sprintf("%.2f",as.numeric(tab_out_kable[6,11])),"(",sprintf("%.2f",round(IQR(tab12$m),2)),")")
kbl(tab_out_kable,booktabs=TRUE,escape = FALSE,col.names = coltext, caption = "Comparison between MCCL and naive estimator across $1000$
Monte-Carlo replicates generated according to (B1) is shown. The median of estimated values is used for
each cell. IQR stands for interquartile range. The empirical interquartile range of each predictor is kept in parentheses. The true parameter values are $\\boldsymbol{\\beta}=\\left(\\beta_{0}, \\beta_{1}, \\beta_{2}\\right)^{\\mathrm{T}}=(0.25,0.25,0.25)^{\\mathrm{T}}$, and $m=3$.") %>%
    kable_styling(position = "center",
                  latex_options = c("HOLD_position"),
                  font_size = 7) %>%
    add_header_above(c(" "=3,
                       '$\\\\sigma^2_u=0.6$'=4,
                       '$\\\\sigma^2_u=1.2$'=4),
                     escape = FALSE) %>%
    collapse_rows(columns = 1:3, latex_hline = "major", valign = "middle")
saveRDS(tab_out_kable,"table1.rds")