library(tidyverse)

reject_rate <- function(pattern){
    files_txt <- list.files(path = ".",pattern = pattern)
    output_list <- list()
    for (i in 1:length(files_txt)) {
        output_list[[i]] <- read.csv(files_txt[i])
    }
    output <- sprintf("%.3f",mean(bind_rows(output_list)$p.value <= 0.05))
    return(output)
}
reject_rate(pattern="*m2n200")
reject_rate(pattern="*m2n300")
reject_rate(pattern="*m2n400")
reject_rate(pattern="*m2n500")

reject_rate(pattern="*m3n200")
reject_rate(pattern="*m3n300")
reject_rate(pattern="*m3n400")
reject_rate(pattern="*m3n500")

reject_rate(pattern="*m4n200")
reject_rate(pattern="*m4n300")
reject_rate(pattern="*m4n400")
reject_rate(pattern="*m4n500")
