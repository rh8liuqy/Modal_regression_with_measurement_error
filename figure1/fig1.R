library(tidyverse)
library(latex2exp)
df1 <- read.csv("09simulation_N2000_corrected_large_variance.csv")
df1 <- df1[2:4]
df2 <- read.csv("10simulation_N2000_naive_large_variance.csv")
df2 <- df2[2:4]
df1 <- gather(df1)
df2 <- gather(df2)
df3 <- read.csv("11simulation_N2000_corrected_large_variance_unknown.csv")
df3 <- df3[2:4]
df3 <- gather(df3)
df1$type <- "MCCL-known"
df2$type <- "Naive"
df3$type <- "MCCL-unknown"
df_plot1 <- rbind(df1,df2,df3)
df_plot1$setting <- as.character(TeX("$X_1$ and $X_2$ are dependent"))

df4 <- read.csv("16simulation_N2000_corrected_large_variance_ind.csv")
df4 <- df4[2:4]
df4 <- gather(df4)
df5 <- read.csv("17simulation_N2000_naive_large_variance_ind.csv")
df5 <- df5[2:4]
df5 <- gather(df5)
df6 <- read.csv("18simulation_N2000_corrected_large_variance_unknown.csv")
df6 <- df6[2:4]
df6 <- gather(df6)
df4$type <- "MCCL-known"
df5$type <- "Naive"
df6$type <- "MCCL-unknown"
df_plot2 <- rbind(df4,df5,df6)
df_plot2$setting <- as.character(TeX("$X_1$ and $X_2$ are independent"))

df_plot <- rbind(df_plot1,df_plot2)

p1 <- df_plot %>% ggplot(aes(x=key,y=value,color=type)) +
    geom_boxplot() +
    geom_hline(yintercept = 0.25, linetype = 2) +
    xlab(element_blank()) +
    ylab(element_blank()) +
    theme_bw() +
    scale_y_continuous(minor_breaks = NULL, breaks = seq(0,1.25,0.125)) +
    scale_x_discrete(labels = c(expression(paste(beta[0])),
                                expression(paste(beta[1])),
                                expression(paste(beta[2])))
    ) +
    facet_wrap(vars(setting),labeller = label_parsed) +
    theme(legend.position = "bottom",
          legend.title = element_blank())
p1
ggsave("fig1.pdf",p1,width = 5*1.2,height = 3*1.2)