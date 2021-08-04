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
    print(x[,(1+(i-1)*m):(i*m)])
    print(y[,(1+(i-1)*m):(i*m)])
  }
}



