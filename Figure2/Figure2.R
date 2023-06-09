library(tidyverse)
library(latticeExtra)
library(gridExtra)
library(latex2exp)
threashold <- 0.2

myfiles <- lapply(list.files(".",pattern="Figure2_n100seed*"), read.csv)
df1 <- bind_rows(myfiles)
myfiles <- lapply(list.files(".",pattern="Figure2_n200seed*"), read.csv)
df2 <- bind_rows(myfiles)
myfiles <- lapply(list.files(".",pattern="Figure2_n500seed*"), read.csv)
df3 <- bind_rows(myfiles)
myfiles <- lapply(list.files(".",pattern="Figure2_n1000seed*"), read.csv)
df4 <- bind_rows(myfiles)

rej_func <- function(data){
    output <- data.frame(x = seq(0,threashold,0.0001))
    output$y <- NA
    for (i in output$x) {
        output[output$x == i, 'y'] <- mean(data$`p.value` <= i)
    }
    return(output)
}

df_rej1 <- rej_func(df1)
df_rej1$group <- "n=100"
df_rej2 <- rej_func(df2)
df_rej2$group <- "n=200"
df_rej3 <- rej_func(df3)
df_rej3$group <- "n=500"
df_rej4 <- rej_func(df4)
df_rej4$group <- "n=1000"

df_ref <- data.frame(x = c(0,threashold),y=c(0,threashold),group = "Reference")
df_plot <- rbind(df_rej1,df_rej2,df_rej3,df_rej4,df_ref)
df_plot$group <- factor(df_plot$group,
                        levels = c("Reference",
                                   "n=100",
                                   "n=200",
                                   "n=500",
                                   "n=1000"))
df_plot$type <- "Under True Model"

df_temp <- df_plot %>% 
    filter(group == "n=100")
p1 <- df_temp %>% ggplot(aes(x=x,y=y)) +
    geom_line(color = rgb(0.8,0,0)) +
    geom_segment(aes(x = 0, xend = 0.2, y = 0, yend = 0.2),linetype = 2) +
    ylim(c(0,threashold)) +
    coord_fixed(ratio=1) +
    xlab(NULL) +
    ylab(NULL) +
    facet_wrap(vars(group)) +
    theme_bw() +
    theme(legend.position = "bottom",
          legend.title = element_blank())

df_temp <- df_plot %>% 
    filter(group == "n=200")
p2 <- df_temp %>% ggplot(aes(x=x,y=y)) +
    geom_line(color = rgb(0.8,0,0)) +
    geom_segment(aes(x = 0, xend = 0.2, y = 0, yend = 0.2),linetype = 2) +
    ylim(c(0,threashold)) +
    coord_fixed(ratio=1) +
    xlab(NULL) +
    ylab(NULL) +
    facet_wrap(vars(group)) +
    theme_bw() +
    theme(legend.position = "bottom",
          legend.title = element_blank())

df_temp <- df_plot %>% 
    filter(group == "n=500")
p3 <- df_temp %>% ggplot(aes(x=x,y=y)) +
    geom_line(color = rgb(0.8,0,0)) +
    geom_segment(aes(x = 0, xend = 0.2, y = 0, yend = 0.2),linetype = 2) +
    ylim(c(0,threashold)) +
    coord_fixed(ratio=1) +
    xlab(NULL) +
    ylab(NULL) +
    facet_wrap(vars(group)) +
    theme_bw() +
    theme(legend.position = "bottom",
          legend.title = element_blank())

df_temp <- df_plot %>% 
    filter(group == "n=1000")
p4 <- df_temp %>% ggplot(aes(x=x,y=y)) +
    geom_line(color = rgb(0.8,0,0)) +
    geom_segment(aes(x = 0, xend = 0.2, y = 0, yend = 0.2),linetype = 2) +
    ylim(c(0,threashold)) +
    coord_fixed(ratio=1) +
    xlab(NULL) +
    ylab(NULL) +
    facet_wrap(vars(group)) +
    theme_bw() +
    theme(legend.position = "bottom",
          legend.title = element_blank())

plotout <- grid.arrange(p1,p2,p3,p4,nrow=2,left = "Rejection Rate", bottom = "Significance Level")
plotout
ggsave("diagnosis-1.pdf",
       plotout,
       width = 5,
       height = 5)
