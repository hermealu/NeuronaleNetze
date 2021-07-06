library(R6)

tanh <- function(x){
  return((exp(2*x)-1)/(exp(2*x)+1))
}

sigmoid <- function(x){
  return(1/(1+exp(-x)))
}

del_sigmoid <- function(x){
  return(sigmoid(x)(1-sigmoid(x)))
}

del_tanh <- function(x){
  return(4/((exp(-x)+exp(x))^2))
}

NN <- R6Class("NN", list(
  L = 1, #Anzahl der Layer
  B = c(1), #Breite der einzelnen Layer 
  W = c(1,1), #Liste soll Länge der dim haben und Einträge sind Matrizen W 
  d = c(1), #List der Affinen Vektoren
  J = 0,
  theta = c(1,1),
  f = sigmoid,
  
  initialize = function(L = 1, B = c(1,1,1), W = c(1,1,1),d=c(1,1,0), min_gewicht=-2, max_gewicht = 2 ) {
    stopifnot(length(B) == L+2)
    self$W <- vector(mode="list",length=L+1)
    self$L <- L
    self$B <- B
    
    #Erstellen der Matrizen W
    for (i in (0:L)){
      self$W[i+1] <- list(matrix(runif(B[i+1]*B[i+2],min=min_gewicht,max=max_gewicht),nrow=B[i+1],ncol=B[i+2]))
    }
    names(self$W) <- LETTERS[1:(L+1)] 
    
    #Erstellen der affinen Vektoren d
    for (i in (0:(L-1))){
      self$d[i+1] <- list(matrix(runif(B[i+2],min=min_gewicht,max=max_gewicht),nrow=1,ncol=B[i+2]))
    }
    self$d[L+1] <- 0
    names(self$d) <- LETTERS[1:(L+1)]
    
    #Erstellen des J -> braucht man hier aber wahrscheinlich nicht sondern erst in der Berechnung
    for (i in (0:L)){
      self$J <- self$J + norm(self$W[[i+1]],"F")
    }
    
    #Erstellen der theta
    
    
  },
  #calculate führt die funktion des NN aus 
  calculate = function(x=1){
    for (j in (1:length(x))){
      h <- x[j]
      if (self$L >= 1){
        for (i in LETTERS[1:(self$L)]){
          h <- self$f(self$d[[i]] + h %*% self$W[[i]]) #letzter Schritt ist ohne Aktivierungsfunktion
        }
      }  
      h <- self$d[[LETTERS[(self$L+1)]]] + h %*% self$W[[LETTERS[(self$L+1)]]]
      x[j] <- h
    }  
    return(x)
  },
  
  eval_till_layer = function(x=1,Layer=1){
    if (Layer == self$L +1){
      x <- self$calculate(x)
    }
    if (self$L >= 1){
      for (i in LETTERS[1:(Layer)]){
        x <- self$f(self$d[[i]] + x %*% self$W[[i]]) #letzter Schritt ist ohne Aktivierungsfunktion
      }
    }
    
      
    return(x)
  },
  
  #Durchführen eines Gradientdescends
  GD = function(x,y,lambda=1,stepsize=1e-4,iterations=100){
    n <- length(x)
    R1 <- 1/n * sum((y-self$calculate(x))^2)
    for (i in 1:iterations){ 
      self$W[["A"]] <- self$W[["A"]] + 1e-3 * x[i] * (y[i]-self$calculate(x[i])) 
    }
    R2 <- 1/n * sum((y-self$calculate(x))^2)
  },
  
  GD2 = function(x,y,lambda=1,stepsize=1e-4,iterations=100){
    n <- length(x)
    for (i in 1:n){
      self$W[[LETTERS[self$L+1]]] <- self$W[[LETTERS[self$L+1]]] + t(0.1*self$eval_till_layer(x[i],self$L)*(y[i]-self$calculate(x[i])))
      self$W[[LETTERS[self$L]]] <- self$W[[LETTERS[self$L]]] + 0.1*t(self$W[[LETTERS[self$L+1]]])*self$eval_till_layer(x[i],self$L)*(y[i]-self$calculate(x[i]))
    }
    
    
    
  }
  )
)



N1 <- NN$new(1,c(1,30,1))






#plot(1,1,xlim=c(-10,10),ylim=c(-10,10))
#lines(x,y,type="l",col="red")
#xs <- sample(x)
#N1$GD(xs,xs,iterations=25)
#y <- N1$calculate(x)
#lines(x,y,col="blue")
#N1$GD(xs,xs,iterations=50)
#y <- N1$calculate(x)
#lines(x,y,col="green")
#N1$GD(xs,xs,iterations=100)
#y <- N1$calculate(x)
#lines(x,y,col="black")
#N1$GD(xs,xs,iterations=200)
#y <- N1$calculate(x)
#lines(x,y,col="orange")
#lines(x,x,col="purple")
#legend(-10, 5, legend=c("start", "25 Schritte","50 Schritte","100 Schritte","200 Schritte","echte funktion"),
#       col=c("red", "blue","green","black","orange","purple"),lty=1,  cex=0.8)
#N1$W
#N1$J
