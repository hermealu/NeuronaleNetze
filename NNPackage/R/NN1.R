#' Softmax function
#'
#' Calculating the softmax function of a vector
#'
#' The softmax function maps the entries a vector onto the intervall (0,1]
#'
#' @param x A vector.
#' @return A vector of solutions of \code{1/sum(exp(x))*exp(x)}
#' @examples
#' softmax(1)
#' softmax(1:100)
#' @export
softmax <- function(x){
  return(1/sum(exp(x))*exp(x))
}


#' Tangens hyperbolicus
#'
#' Calculating the tangens hyperbolicus of a vector
#'
#' @param x A vector.
#' @return A vector of solutions of \code{(exp(2*x)-1)/(exp(2*x)+1)}
#' @examples
#' tanh(1)
#' tanh(1:100)
#' @export
tanh <- function(x){
  return((exp(2*x)-1)/(exp(2*x)+1))
}


#' Sigmoid function
#'
#' Calculating the sigmoid of a vector
#'
#' @param x A vector.
#' @return A vector of solutions of \code{1/(1+exp(-x))}
#' @examples
#' sigmoid(1)
#' sigmoid(1:100)
#' @export
sigmoid <- function(x){
  return(1/(1+exp(-x)))
}


#' Derivative of sigmoid function
#'
#' Calculating the sigmoid's first derivative of a vector
#'
#' @param x A vector.
#' @return A vector of solutions of \code{sigmoid(x)*(1-sigmoid(x))}
#' @examples
#' del_sigmoid(1)
#' del_sigmoid(1:100)
#' @export
del_sigmoid <- function(x){
  return(sigmoid(x)*(1-sigmoid(x)))
}


#' Derivative of tangens hyperbolicus
#'
#' Calculating the tangens hyperbolicus' first derivative of a vector
#'
#' @param x A vector.
#' @return A vector of solutions of \code{4/((exp(-x)+exp(x))^2)}
#' @examples
#' del_tanh(1)
#' del_tanh(1:100)
#' @export
del_tanh <- function(x){
  return(4/((exp(-x)+exp(x))^2))
}

#' S6 class that can generate Neural Networks
#'
#' In this class Neural Networks can be generated and optimized via different
#' Gradient Descends Methods
#'
#' \describe{ \item{L}{number of hidden layers}
#'
#' \item{B}{width of each layer}
#'
#' \item{W}{weight matrices}
#'
#' \item{d}{width vector} }
#' @name NN
#' @rdname NN
#' @exportClass NN
NN <- R6Class("NN", list(
  L = 1, #Anzahl der Hidden Layer
  B = c(1), #Breite der einzelnen Layer
  W = c(1,1), #Liste soll Länge der dim haben und Einträge sind Matrizen W
  d = c(1), #List der Affinen Vektoren
  J = 0,
  theta = c(1,1),
  f = sigmoid,
  del_f = del_sigmoid,

  #' @description
  #'
  #' Initializing a neural network
  #'
  #' @param L a scalar, which refers to the number of hidden layers
  #' @param B a atomic vector, which contains the width of the input, the hidden layers and the output
  #' @param W a list with length L+1, which contains matrix-inputs of the weight matrices of the neural network
  #' @param d a list with length L+1, which contains atomic vectors. They represent the affine vectors of the neural network
  #' @param min_gewicht a scalar, which sets the minimal number that can appear in a weight matrix
  #' @param max_gewicht a scalar, which sets the maximal number that can appear in a weight matrix
  #' @return A R6 object, which contains the parameters of a neural network and methods for its optimization
  #' @export
  initialize = function(L = 1, B = c(1,1,1), W = c(1,1,1), d=c(1,1,0), min_gewicht=-2, max_gewicht = 2 ) {
    L <- length(B)-2
    stopifnot(L >= 1)
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
  },

  #' @description
  #' Calculate feedforward propagation for regression of a neural network
  #'
  #' @param x A atomic vector or an array, which represents the input of the network
  #' @export
  # Feedforward propagation - regression
  ffprop = function(x=1){
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

  #' @description
  #' Calculating feed forward propagation for classification with softmax
  #'
  #' @param x An array, which represents the input of the network
  #' @export
  # Feedwordward propagation - classification
  ffprop_clas = function(x=1){
    stopifnot("x has to be an array!" = is.array(x))
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

  },

  #' @description
  #' Calculating feedworward propagation up to a certain layer Layer
  #'
  #' @param x A vector or array
  #' @param Layer A scalar smaller than or equal as L+1 and bigger than 0.

  eval_till_layer = function(x=1,Layer=1){
    stopifnot("Layer has to be smaller than L+2!" = Layer <= self$L+1)
    stopifnot("Layer has to be bigger than 0!" = Layer >= 1)
    if (Layer == self$L +1){
      x <- self$ffprop(x)
    }
    if (self$L >= 1){
      h <- x
      for (i in LETTERS[1:(Layer)]){  # Kann das von 1 laufen? Was ist mit dem Input?
        h <- self$f(self$d[[i]] + h %*% self$W[[i]] )
      }

    }

    return(h)
  },

  #' @description
  #' Calculating feedforward propagation up to a certain layer without activation function in last step
  #'
  #' @param x A vector or array
  #' @param Layer A scalar smaller than or equal as L+1 and bigger than 0.
  eval_till_layer_z = function(x=1,Layer=1){
    stopifnot("Layer has to be smaller than L+2!" = Layer <= self$L+1)
    stopifnot("Layer has to be bigger than 0!" = Layer >= 1)
    if(Layer > 1) evx <- self$eval_till_layer(x, Layer-1)
    if(Layer == 1) evx <- x
    h <- self$d[[LETTERS[Layer]]] + evx %*% self$W[[LETTERS[Layer]]]

    return(h)
  },

  #' @description
  #' Method, which calculates a random descent for a neural network
  #' Calculating gradient descent
  #'
  #' @param x A vector or array with rows representing the input of the training
  #'   data and columns representing the number of different training data
  #' @param y A vector or array with rows representing the input of the training
  #'   data and columns representing the number of different training data
  #' @param delta A scalar between 0 and 1 that determines how much the weights change
  #' @param iteration a scalar that represents the times the full data is applied in the algorithm
  #' @export
  GD = function(x,y,iteration=10,delta=0.02){
    for (j in 1:iteration){
      W_tmp <- vector(mode="list",length=self$L+1)
      names(W_tmp) <- LETTERS[1:(self$L+1)]
      d_tmp <- vector(mode="list",length=self$L+1)
      names(d_tmp) <- LETTERS[1:(self$L+1)]
      R1 <- sum((y-self$ffprop(x))^2)
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
      R2 <- sum((y-self$ffprop(x))^2)
      if (R1 < R2){
        self$W <- W_tmp
        self$d <- d_tmp

      }
    }

  },

  #' @description
  #' A Method, which calculates the Gradient Decent for regression for a neural network
  #'
  #' @param x A vector or array with rows representing the input of the training
  #'   data and columns representing the number of different training data
  #' @param y A vector or array with rows representing the input of the training
  #'   data and columns representing the number of different training data
  #' @param gam A scalar between zero and 1
  #' @export
  # Realization of backwardpropagation - regression
  BP_reg = function(x,y, gam = 1e-4){
    if(!is.array(x)) {
      x <- matrix(x, nrow = 1)
    }
    if(is.array(x)){
      n <- dim(x)[2]

      # Definition von C_v und C_w
      C_w <- vector(mode="list",length=(self$L+1))
      names(C_w) <- LETTERS[1:(self$L+1)]

      C_v <- vector(mode="list",length=self$L)
      names(C_v) <- LETTERS[1:(self$L)]

      for(l in 1:self$L){
        C_v[[LETTERS[l]]] <- matrix(0, ncol = self$B[l+1], nrow = dim(x)[2])
      }

      for(l in 1:(self$L+1)){
        C_w[[LETTERS[l]]] <- array(0, dim = c(dim(self$W[[LETTERS[l]]])[2],
                                              dim(self$W[[LETTERS[l]]])[1], dim(x)[2] ))
      }

      for (i in 1: dim(x)[2]){

        # Schritt 1 - Forwardpass und Definition von g_x, z und x_l
        g_x <- self$ffprop(matrix(x[,i],ncol=1))

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
        # Berechne delta^(l)
        if(self$L > 1){
          for(l in (self$L-1):1){
            delta_l[[LETTERS[l]]] <- (delta_l[[LETTERS[l+1]]] %*% t(self$W[[LETTERS[l+1]]])) *
              self$del_f(z[[LETTERS[l]]])
          }
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
              sum <- sum + C_w[[LETTERS[l]]][j,m,i]
            }
            self$W[[LETTERS[l]]][m,j] <- self$W[[LETTERS[l]]][m,j] -
              gam*(1/n)*sum
          }
        }
      }
    }
  },

  #' @description
  #' Calculating stochastic gradient descent
  #'
  #' @param x A vector or array with rows representing the input of the training
  #'   data and columns representing the number of different training data
  #' @param y A vector or array with rows representing the input of the training
  #'   data and columns representing the number of different training data
  #' @param n a vector
  #' @param delta A scalar between 0 and 1 that determines how much the weights change
  #' @param iteration a scalar that represents the times the full data is applied in the algorithm
  #' @export
  SGD = function(x,y,n,delta=1e-2,iteration=10){
    if(!is.array(x)) {x <- matrix(x,nrow=1); y <- matrix(y,nrow=1)}
    for (j in 1:iteration){
      stopifnot(dim(x)[2] %% n == 0)
      c <- sample(1:(dim(x)[2]))
      for (i in 1:(dim(x)[1])){
        x[i,] <- x[i,][c]
        y[i,] <- y[i,][c]
      }
      m <- dim(x)[2]/n
      for (i in 1:(n)){
        self$BP_reg(x[,(1+(i-1)*m):(i*m)],y[,(1+(i-1)*m):(i*m)],gam=delta)
      }
    }
  },

  #' @description
  #' Calculating stochastic gradient descent for classification
  #'
  #' @param x A vector or array with rows representing the input of the training
  #'   data and columns representing the number of different training data
  #' @param y A vector or array with rows representing the input of the training
  #'   data and columns representing the number of different training data
  #' @param n a vector
  #' @param delta A scalar between 0 and 1 that determines how much the weights change
  #' @param iteration a scalar that represents the times the full data is applied in the algorithm
  #' @export
  SGD_clas = function(x,y,n,delta=1e-2,iteration=10){
    if(!is.array(x)) {x <- matrix(x,nrow=1); y <- matrix(y,nrow=1)}
    for (j in 1:iteration){
      stopifnot(dim(x)[2] %% n == 0)
      c <- sample(1:(dim(x)[2]))
      for (i in 1:(dim(x)[1])){
        x[i,] <- x[i,][c]
      }
      for (i in 1:(dim(y)[1])){
        y[i,] <- y[i,][c]
      }

      m <- dim(x)[2]/n
      for (i in 1:(n)){
        self$BP_clas(x[,(1+(i-1)*m):(i*m)],y[,(1+(i-1)*m):(i*m)],gam=delta)
      }
    }
  },

  # Realization of backwardpropagation - classifikation
  #' @description
  #' Calculating gradient descent for classification
  #'
  #' @param x A vector or array with rows representing the input of the training
  #'   data and columns representing the number of different training data
  #' @param y A vector or array with rows representing the input of the training
  #'   data and columns representing the number of different training data
  #' @param gam A scalar between 0 and 1 that determines how much the weights change
  #' @export
  BP_clas = function(x,y, gam = 1e-4){
    if(!is.array(x)) {
      x <- matrix(x, nrow = 1)
    }

    if(is.array(x)){
      n <- dim(x)[2]
      # Definition con C_v und C_w
      C_w <- vector(mode="list",length=(self$L+1))
      names(C_w) <- LETTERS[1:(self$L+1)]

      C_v <- vector(mode="list",length=self$L)
      names(C_v) <- LETTERS[1:(self$L)]

      for(l in 1:self$L){
        C_v[[LETTERS[l]]] <- matrix(0, ncol = self$B[l+1], nrow = dim(x)[2])
      }

      for(l in 1:(self$L+1)){
        C_w[[LETTERS[l]]] <- array(0, dim = c(dim(self$W[[LETTERS[l]]])[2],
                                              dim(self$W[[LETTERS[l]]])[1], dim(x)[2] ))
      }

      for (i in 1: dim(x)[2]){
        # Schritt 1 - Forwardpass und Definition von g_x, z und x_l
        g_x <- self$ffprop_clas(matrix(x[,i],ncol=1))

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

        secondterm <- 0
        for(k in 1:(dim(self$W[[LETTERS[self$L+1]]])[2])){
          secondterm <- secondterm + g_x[k]*self$W[[LETTERS[self$L+1]]][k,]
        }

        delta_l[[LETTERS[self$L]]] <- -(self$W[[LETTERS[self$L+1]]][y[,i],] - secondterm)*
          self$del_f(z[[LETTERS[self$L]]])

        # Berechnung von C_w[L+1]
        for(m in 1:length(self$W[[LETTERS[self$L+1]]][,1])){
          C_w[[LETTERS[self$L+1]]][,m,i] <- -(y[,i] - g_x) * x_l[[LETTERS[self$L]]][m]
        }
      }
      # Schritt 2b
      for(i in 1:(dim(x)[2])){
        if(self$L >1){
          for(l in (self$L-1):1){
            # Berechne delta^(l)
            delta_l[[LETTERS[l]]] <- (delta_l[[LETTERS[l+1]]] %*% t(self$W[[LETTERS[l+1]]])) *
              self$del_f(z[[LETTERS[l]]])
          }
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
              sum <- sum + C_w[[LETTERS[l]]][j,m,i]
            }
            self$W[[LETTERS[l]]][m,j] <- self$W[[LETTERS[l]]][m,j] -
              gam*(1/n)*sum
          }
        }
      }
    }
  }
)
)
