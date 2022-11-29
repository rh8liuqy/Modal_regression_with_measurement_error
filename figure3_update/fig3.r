library(tidyverse)
df1 <- read.csv("data_plot_wishreg.csv")
fit1 <- function(x) {
    etaX <- -1.5810 + 0.2700*x
    thetaX <- 1/(1+exp(-etaX))
    return(thetaX)
}
fit2 <- function(x) {
    etaX <- -1.5693 + 1.0563*x
    thetaX <- 1/(1+exp(-etaX))
    return(thetaX)
}
df2 <- data.frame(w = seq(min(df1$w),max(df1$w),0.001))
df2$y <- fit1(df2$w)
df2$Model <- "Navie Estimate"
df3 <- data.frame(w = seq(min(df1$w),max(df1$w),0.001))
df3$y <- fit2(df3$w)
df3$Model <- "MCCL"
df2 <- rbind(df2,df3)

plot_out <- df1 %>% ggplot(aes(x = w, y = y)) +
    geom_point(alpha=0.3) +
    geom_line(data = df2, aes(x = w, y = y, color = Model, linetype = Model)) +
    xlab("Long Term Intake") +
    ylab("Scaled FFQ Intake") +
    theme_bw() +
    theme(aspect.ratio=1)

saveRDS(plot_out,"figure3.rds")
ggsave("figure3.pdf",plot_out,width = 8,height = 4)
