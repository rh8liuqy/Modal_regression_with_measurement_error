library(knitr)
library(kableExtra)
tab_out <- matrix(NA,ncol = 8,nrow = 3)
tab1 <- read.csv("07simulation_N200_corrected_large_variance.csv")
tab2 <- read.csv("15simulation_N200_corrected_large_variance_unknown.csv")
tab3 <- read.csv("08simulation_N200_naive_large_variance.csv")


tab_out[1,1] <- paste0(sprintf("%.2f",mean(tab1$b0_cov)),
                       "(",
                       sprintf("%.2f",sd(tab1$b0_cov)),
                       ")")
tab_out[2,1] <- paste0(sprintf("%.2f",mean(tab2$b0_cov)),
                       "(",
                       sprintf("%.2f",sd(tab2$b0_cov)),
                       ")")
tab_out[3,1] <- paste0(sprintf("%.2f",mean(tab3$b0_cov)),
                       "(",
                       sprintf("%.2f",sd(tab3$b0_cov)),
                       ")")

tab_out[1,3] <- paste0(sprintf("%.2f",mean(tab1$b1_cov)),
                       "(",
                       sprintf("%.2f",sd(tab1$b1_cov)),
                       ")")
tab_out[2,3] <- paste0(sprintf("%.2f",mean(tab2$b1_cov)),
                       "(",
                       sprintf("%.2f",sd(tab2$b1_cov)),
                       ")")
tab_out[3,3] <- paste0(sprintf("%.2f",mean(tab3$b1_cov)),
                       "(",
                       sprintf("%.2f",sd(tab3$b1_cov)),
                       ")")

tab_out[1,5] <- paste0(sprintf("%.2f",mean(tab1$b2_cov)),
                       "(",
                       sprintf("%.2f",sd(tab1$b2_cov)),
                       ")")
tab_out[2,5] <- paste0(sprintf("%.2f",mean(tab2$b2_cov)),
                       "(",
                       sprintf("%.2f",sd(tab2$b2_cov)),
                       ")")
tab_out[3,5] <- paste0(sprintf("%.2f",mean(tab3$b2_cov)),
                       "(",
                       sprintf("%.2f",sd(tab3$b2_cov)),
                       ")")

tab_out[1,7] <- paste0(sprintf("%.2f",mean(tab1$m_cov)),
                       "(",
                       sprintf("%.2f",sd(tab1$m_cov)),
                       ")")
tab_out[2,7] <- paste0(sprintf("%.2f",mean(tab2$m_cov)),
                       "(",
                       sprintf("%.2f",sd(tab2$m_cov)),
                       ")")
tab_out[3,7] <- paste0(sprintf("%.2f",mean(tab3$m_cov)),
                       "(",
                       sprintf("%.2f",sd(tab3$m_cov)),
                       ")")

tab_out[1,2] <- round(sd(tab1$b0),2)
tab_out[2,2] <- round(sd(tab2$b0),2)
tab_out[3,2] <- round(sd(tab3$b0),2)

tab_out[1,4] <- round(sd(tab1$b1),2)
tab_out[2,4] <- round(sd(tab2$b1),2)
tab_out[3,4] <- round(sd(tab3$b1),2)


tab_out[1,6] <- round(sd(tab1$b2),2)
tab_out[2,6] <- round(sd(tab2$b2),2)
tab_out[3,6] <- round(sd(tab3$b2),2)

tab_out[1,8] <- round(sd(tab1$m),2)
tab_out[2,8] <- round(sd(tab2$m),2)
tab_out[3,8] <- round(sd(tab3$m),2)

#tab_out <- round(tab_out,2)
tab_out <- data.frame(tab_out)
tab_rowname <- data.frame(col1 = rep("$n=200$",3),
                          col2 = c("MCCL",
                                   "MCCL",
                                   "Naive"),
                          col3 = c("Known",
                                   "Unknown",
                                   ""))
tab_out_kable <- cbind(tab_rowname,tab_out)
saveRDS(tab_out_kable,"table2.rds")
coltext <- c("","","Meas.error var.",
             rep(c("$\\widehat{\\text { s.d. }}$",
                   "s.d."),
                 4))
tab <- kable(tab_out_kable,booktabs=TRUE,escape = FALSE,col.names = coltext, 
      caption = "Comparison between sandwich variance estimation, $\\widehat{\\text { s.d. }}$, and empirical variance estimation, s.d., across $1000$ Monte-Carlo replicates based on (B1) The mean of estimated values is used for cells about sandwich variance estimation. Numbers in parentheses are Monte-Carlo standard errors associated with the mean.",
      format = 'latex') %>%
    kable_styling(position = "center") %>%
    add_header_above(c(" "=3,
                       '$\\\\beta_0$'=2,
                       '$\\\\beta_1$'=2,
                       '$\\\\beta_2$'=2,
                       '$m$'=2),
                     escape = FALSE) %>%
    add_header_above(c(" "=3,
                       '$\\\\sigma^2_u=1.2$'=8),
                     escape = FALSE) %>%
    collapse_rows(columns = 1:3, latex_hline = "major", valign = "middle")

writeLines(tab, 'tab.tex')