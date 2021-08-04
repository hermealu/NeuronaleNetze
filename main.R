devtools::install("NNPackage")
library("NNPackage")
#BP_reg(x[,(1+(i-1)*m):(i*m)],y[,(1+(i-1)*m):(i*m)])
N1 <- NN$new(3,c(1,30,30,30,1)) 
N1$calculate2(1)

x <- matrix(seq(-pi, pi, length.out = 1000), nrow = 1)
y <- sin(x)

Batch <- function(x,y,n){
  stopifnot(dim(x)[2] %% n == 0)
  c <- sample(1:(dim(x)[2]))
  for (i in 1:(dim(x)[1])){
    x[i,] <- x[i,][c]
    y[i,] <- y[i,][c]
  }
  m <- dim(x)[2]/n
  for (i in 1:(n)){
    print("Batch: ", i)
    N1$BP(x[,(1+(i-1)*m):(i*m)],y[,(1+(i-1)*m):(i*m)],gam=1e-2)
  }
  plot(x,N1$calculate2(x))
}

for(i in 1:100){Batch(x,y,10);print("Iteration: ", i)}
