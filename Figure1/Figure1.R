library(tidyverse)
library(latex2exp)
df1 <- read.csv("01_Figure1.csv")
df1 <- df1[,c(3,2,4)]
df1 <- gather(df1)
df1$type <- "MCCL"
df1$setting <- as.character(TeX("$X_1$ and $X_2$ are dependent"))

df2 <- read.csv("02_Figure1.csv")
df2 <- df2[,c(3,2,4)]
df2 <- gather(df2)
df2$type <- "MCCL"
df2$setting <- as.character(TeX("$X_1$ and $X_2$ are independent"))

df3 <- read.csv("03_Figure1.csv")
df3 <- df3[,c(3,2,4)]
df3 <- gather(df3)
df3$type <- "Naive"
df3$setting <- as.character(TeX("$X_1$ and $X_2$ are dependent"))

df4 <- read.csv("04_Figure1.csv")
df4 <- df4[,c(3,2,4)]
df4 <- gather(df4)
df4$type <- "Naive"
df4$setting <- as.character(TeX("$X_1$ and $X_2$ are independent"))

df_plot <- rbind(df1,df2,df3,df4)
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
ggsave("FFQ-1.pdf",p1,width = 5*1.2,height = 3*1.2)
