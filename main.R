devtools::install("NNPackage")
library("NNPackage")

N1 <- NN$new(4,c(1,50,50,50,50,1)) 
N1$calculate2(1)
