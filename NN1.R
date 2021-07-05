library(R6)

tanh <- function(x){
  return((exp(2*x)-1)/(exp(2*x)+1))
}

NN <- R6Class("NN", list(
  L = 1, #Anzahl der Layer
  B = c(1), #Breite der einzelnen Layer 
  W = c(1,1), #Liste soll Länge der dim haben und Einträge sind Matrizen W 
  d = c(1), #List der Affinen Vektoren
  J = 0,
  theta = c(1,1),
  f = tanh,
  
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
  
  #Durchführen eines Gradientdescends
  GD = function(x,y,lambda=1,stepsize=1e-4){
    n <- length(x)
    R1 <- 1/n * sum((y-self$calculate(x))^2)
    #for (i in 1:100){ #hier probiere ich nur rum
      
    #}
  }
  )
)



N1 <- NN$new(1,c(1,4,1))Funktion
N1$GD(x,x)
x <- seq(-10,10,0.001)
y <- N1$calculate(x)
plot(x,y,)
N1$d
N1$J
