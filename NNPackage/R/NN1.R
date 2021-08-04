library(R6)
library(readr)
library(stringr)
library(dplyr)


softmax <- function(x){
  return(1/sum(exp(x))*exp(x))
}

tanh <- function(x){
  return((exp(2*x)-1)/(exp(2*x)+1))
}

sigmoid <- function(x){
  return(1/(1+exp(-x)))
}

del_sigmoid <- function(x){
  return(sigmoid(x)*(1-sigmoid(x)))
}

del_tanh <- function(x){
  return(4/((exp(-x)+exp(x))^2))
}

id <- function(x){
  return(x)
}

NN <- R6Class("NN", list(
  L = 1, #Anzahl der Hidden Layer
  B = c(1), #Breite der einzelnen Layer 
  W = c(1,1), #Liste soll Länge der dim haben und Einträge sind Matrizen W 
  d = c(1), #List der Affinen Vektoren
  J = 0,
  theta = c(1,1),
  f = sigmoid,
  del_f = del_sigmoid,
  
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
    for(i in 1:L){
      self$theta[i] <- list(c(unlist(self$W[i]), unlist(self$d[i])))
    }
    self$theta[L+1] <- self$W[L+1]
    
  },
  #calculate führt die funktion des NN aus 
  # calculate = function(x=1){
  #   for (j in (1:length(x))){
  #     h <- x[j]
  #     if (self$L >= 1){
  #       for (i in LETTERS[1:(self$L)]){
  #         h <- self$f(self$d[[i]] + h %*% self$W[[i]]) #letzter Schritt ist ohne Aktivierungsfunktion
  #       }
  #     }  
  #     h <- self$d[[LETTERS[(self$L+1)]]] + h %*% self$W[[LETTERS[(self$L+1)]]]
  #     x[j] <- h
  #   }  
  #   return(x)
  # },
  
  calculate2 = function(x=1){
    if (is.array(x)){
      y <- matrix(1,nrow=dim(x)[2],ncol=self$B[self$L+2])
      for (j in (1:(dim(x)[2]))){    
        h <- x[,j]
        if (self$L >= 1){
          for (i in LETTERS[1:(self$L)]){
            h <- self$f(self$d[[i]] + h %*% self$W[[i]]) #letzter Schritt ist ohne Aktivierungsfunktion
          }
        }  
        h <- self$d[[LETTERS[(self$L+1)]]] + h %*% self$W[[LETTERS[(self$L+1)]]]
        
        y[j,] <- h
      }  
      return(y)
    }
    
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
  
  
  cal_clas = function(x=1){
    if (is.array(x)){
      y <- matrix(1,nrow=dim(x)[2],ncol=self$B[self$L+2])
      for (j in (1:(dim(x)[2]))){
        h <- x[,j]
        if (self$L >= 1){
          for (i in LETTERS[1:(self$L)]){
            h <- self$f(self$d[[i]] + h %*% self$W[[i]]) #letzter Schritt ist ohne Aktivierungsfunktion
          }
        }  
        h <- softmax(self$d[[LETTERS[(self$L+1)]]] + h %*% self$W[[LETTERS[(self$L+1)]]])
        
        y[j,] <- h
      }  
      return(y)
    }
    
    y <- matrix(1,nrow=length(x),ncol=self$B[self$L+2])
    for (j in (1:length(x))){
      h <- x[j]
      if (self$L >= 1){
        for (i in LETTERS[1:(self$L)]){
          h <- self$f(self$d[[i]] + h %*% self$W[[i]]) #letzter Schritt ist ohne Aktivierungsfunktion
        }
      }  
      h <- softmax(self$d[[LETTERS[(self$L+1)]]] + h %*% self$W[[LETTERS[(self$L+1)]]])
      y[j,] <- h
    }  
    return(y)
  },
  
  eval_till_layer = function(x=1,Layer=1){
    if (Layer == self$L +1){
      x <- self$calculate2(x)
    }
    if (self$L >= 1){
      h <- x
      for (i in LETTERS[1:(Layer)]){  # Kann das von 1 laufen? Was ist mit dem Input?
          h <- self$f(self$d[[i]] + h %*% self$W[[i]] ) #letzter Schritt ist ohne Aktivierungsfunktion
        }
        
      }
    
    return(h)
  },
  
  eval_till_layer_z = function(x=1,Layer=1){
    if(Layer > 1) evx <- self$eval_till_layer(x, Layer-1)
    if(Layer == 1) evx <- x
    h <- self$d[[LETTERS[Layer]]] + evx %*% self$W[[LETTERS[Layer]]] 
    
    return(h)
  },
  
  # Durchführen von Backwardpropagation 
  BP = function(x,y, gam = 1e-4, lambda = 0){
    if(!is.array(x)) {
      x <- matrix(x, ncol = 1)
      }
    
    if(is.array(x)){
      n <- length(dim(x)[2])
    # Definition con C_v und C_w
      C_w <- vector(mode="list",length=(self$L+1))
      names(C_w) <- LETTERS[1:(self$L+1)]
      
      C_v <- vector(mode="list",length=self$L)
      names(C_v) <- LETTERS[1:(self$L)]
      
      for(l in 1:self$L){
        C_v[[LETTERS[l]]] <- matrix(0, ncol = self$B[l+1], nrow = dim(x)[2])
      }
      
      for(l in 1:(self$L+1)){
        # C_w[[LETTERS[l]]] <- array(0, dim = c(length(self$W[[LETTERS[l]]][1,]), 
        #                                       length(self$W[[LETTERS[l]]][,1]), dim(x)[2] ))
        C_w[[LETTERS[l]]] <- array(0, dim = c(dim(self$W[[LETTERS[l]]])[2], 
                                              dim(self$W[[LETTERS[l]]])[1], dim(x)[2] ))
      }
      
      for (i in 1: dim(x)[2]){
        # Schritt 1 - Forwardpass und Definition von g_x, z und x_l
        g_x <- self$calculate2(matrix(x[,i],ncol=1))
        
        z <- vector(mode="list",length=self$L)
        names(z) <- LETTERS[1:(self$L)]
        x_l <- vector(mode="list",length=self$L)
        names(x_l) <- LETTERS[1:(self$L)]
        
        delta_l <- vector(mode="list",length=(self$L))
        names(delta_l) <- LETTERS[1:(self$L)]
        
        for(l in 1:(self$L)){
         z[[LETTERS[l]]] <- self$eval_till_layer_z(x[,i], l)
         x_l[[LETTERS[l]]] <- self$eval_till_layer(x[,i], l)
        }
        
        # Schritt 2a - Berechnung delta^(L) und der Ableitungen C_w[1,m]^(L+1)
        delta_l[[LETTERS[self$L]]] <- (-2*(y[i]-g_x) %*% t(self$W[[LETTERS[self$L+1]]])) *self$del_f(z[[LETTERS[self$L]]])
        
        # Berechnung von C_w[L+1]
        for(m in 1:length(self$W[[LETTERS[self$L+1]]][,1])){
          C_w[[LETTERS[self$L+1]]][1,m,i] <- -2*(y[i] - g_x)*x_l[[LETTERS[self$L]]][m]
          
        }
        
        # Schritt 2b
        for(l in (self$L-1):1){   
          # Berechne delta^(l)         
           delta_l[[LETTERS[l]]] <- (delta_l[[LETTERS[l+1]]] %*% t(self$W[[LETTERS[l+1]]])) *
                                       self$del_f(z[[LETTERS[l]]])
        }
        
        # Berechne C_v
        for(l in (self$L):1){
          C_v[[LETTERS[l]]][i,] <- delta_l[[LETTERS[l]]]
        }
         
        # Berechne restliches C_w
        for(l in (self$L):1){
          for(j in 1:dim(self$W[[LETTERS[l]]])[2]){
            for(m in 1:dim(self$W[[LETTERS[l]]])[1]){
              if(l>1)
              C_w[[LETTERS[l]]][j,m,i] <- delta_l[[LETTERS[l]]][j] * x_l[[LETTERS[l-1]]][m] else
                C_w[[LETTERS[l]]][j,m,i] <- delta_l[[LETTERS[l]]][j] * x[,i][m]
            }
          }
        }
      }
      
      # Berechne neue Verschiebungsvektoren
      for(l in 1:self$L){
        sum <- 0
        for(i in 1:n){
          sum <- sum + C_v[[LETTERS[l]]][i,]
        }
        self$d[[LETTERS[l]]] <- self$d[[LETTERS[l]]] - gam * (1/n) * sum
      }
      
      # Berechne neue Gewichtungsmatrizen
      for(l in 1:(self$L+1)){
        for(j in 1:dim(self$W[[LETTERS[l]]])[2]){
          for(m in 1:dim(self$W[[LETTERS[l]]])[1]){
            sum <- 0
            for(i in 1:n){
              sum <- sum + C_w[[LETTERS[l]]][j,m,i] +
                      2*lambda * self$W[[LETTERS[l]]][m,j]
            }
      self$W[[LETTERS[l]]][m,j] <- self$W[[LETTERS[l]]][m,j] -
                                  gam*(1/n)*sum
        }
      }
      }
      return(self$W)
    }
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
  },
  GD3 = function(x,y,iteration=10,delta=0.02){
      for (j in 1:iteration){
        W_tmp <- vector(mode="list",length=self$L+1)
        names(W_tmp) <- LETTERS[1:(self$L+1)]
        d_tmp <- vector(mode="list",length=self$L+1)
        names(d_tmp) <- LETTERS[1:(self$L+1)]
        R1 <- sum((y-self$calculate2(x))^2)
        for (k in 1:(self$L +1)){
          W_tmp[[LETTERS[k]]] <- self$W[[LETTERS[k]]]
          for (l in 1:length(self$W[[LETTERS[k]]])){
            self$W[[LETTERS[k]]][l] <- self$W[[LETTERS[k]]][l] + runif(1,min=-delta,max=delta)
          }
          d_tmp[[LETTERS[k]]] <- self$d[[LETTERS[k]]]
          for (l in 1:length(self$d[[LETTERS[k]]])){
            self$d[[LETTERS[k]]][l] <- self$d[[LETTERS[k]]][l] + runif(1,min=-delta,max=delta)
          }
        }
        R2 <- sum((y-self$calculate2(x))^2)
        if (R1 < R2){
          self$W <- W_tmp
          self$d <- d_tmp
          
        }
        
        print(R1)
        print(R2)
        if (j %% 10){
          print(j/10)
        }
      }

  },
  GD_clas = function(x,y,iteration=1000,delta=0.002){
    for (j in 1:iteration){
      W_tmp <- vector(mode="list",length=self$L+1)
      names(W_tmp) <- LETTERS[1:(self$L+1)]
      d_tmp <- vector(mode="list",length=self$L+1)
      names(d_tmp) <- LETTERS[1:(self$L+1)]
      R1 <- sum((y-self$cal_clas(x))^2)
      for (k in 1:(self$L +1)){
        W_tmp[[LETTERS[k]]] <- self$W[[LETTERS[k]]]
        for (l in 1:length(self$W[[LETTERS[k]]])){
          self$W[[LETTERS[k]]][l] <- self$W[[LETTERS[k]]][l] + runif(1,min=-delta,max=delta)
        }
        d_tmp[[LETTERS[k]]] <- self$d[[LETTERS[k]]]
        for (l in 1:length(self$d[[LETTERS[k]]])){
          self$d[[LETTERS[k]]][l] <- self$d[[LETTERS[k]]][l] + runif(1,min=-delta,max=delta)
        }
      }
      R2 <- sum((y-self$cal_clas(x))^2)
      if (R1 < R2){
        self$W <- W_tmp
        self$d <- d_tmp
        
      }
      
      print(R1)
      print(R2)
      if (j %% 10){
        print(j/10)
      }
    }
    
  }
  )
)

#lines <- read_lines("R_project/tic-tac-toe.data")
#lines2 <- read_lines("R_project/poker-hand-testing.data")
#lines2 %>% 
 # str_extract_all("[:digit:]+") ->
#  poker1
#poker2 <- as.numeric(unlist(poker1))

#poker <- matrix(poker2, nrow = 11)
#lines %>% 
  #str_replace_all("negative","0") %>% 
  #str_replace_all("positive","1") %>% 
  #str_replace_all("x","1") %>% 
  #str_replace_all("o","3") %>% 
  #str_replace_all("b","2") %>% 
  #str_extract_all("[:digit:]") ->
  #tic_tac1


#tic_tac2 <- as.numeric(unlist(tic_tac1))
#tic_tac <- matrix(tic_tac2, nrow = 10)

#x_poker <- poker[-11,]

#x <- tic_tac[-10,1:950] 
#x_all <- tic_tac[-10,]


#y <- tic_tac[10,1:950]
#y_poker <- poker[11,]


#y1 <- cbind(y,integer(length(y)))
#for (i in 1:length(y)){
 # if(y[i]==0) y1[i,2] <- 1
#}
#y_poker1 <- cbind(y_poker,integer(length(y_poker)))
#for (i in 1:length(y_poker)){
 # if(y_poker[i]==0) y_poker1[i,2] <- 1
  #if(y_poker[i]!=0) y_poker1[i,1] <- 1
#}


#y_poker1 

#N1 <- NN$new(4,c(10,50,50,50,50,2)) 

#N1$GD_clas(x_poker,y_poker1,iteration = 1000,delta=0.02)



#N1$cal_clas(x_poker[,1:2])

N1 <- NN$new(4,c(2,10,7,8,10,1))
x <- matrix(c(1,1,2,2), ncol = 2)
y <- c(1,1)


N1$calculate2(x)
N1$eval_till_layer_z(1:2,3)
N1$eval_till_layer(1:2,3)
N1$BP(x,y)
N1$W







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
