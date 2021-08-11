#' Save the neural network
#'
#' Saving the neural network including it's weights
#'
#' @param NN A NN-Class Object
#' @examples
#' save_NN(NN$new(B=c(1,10,1)))
#' @export
save_NN <- function(NN){
  W_saved_neuralnetwork <-NN$W
  d_saved_neuralnetwork <- NN$d
  B_saved_neuralnetwork <- NN$B

  name <- readline(prompt = "Do you really want to overwrite the existing neural network?[y/n] ")
  if (name == "y"){
    usethis::use_data(W_saved_neuralnetwork,B_saved_neuralnetwork,d_saved_neuralnetwork, internal = TRUE, overwrite = TRUE)
  }
}

#' Load the neural network
#'
#' Loading in the saved neural network including it's weights
#' @param path path from working directory to the saved file which is in NNPackage/R/dateiname
#' @export
load_NN <- function(path = "R/sysdata.rda"){
  #devtools::load_all(".")
  load(path)
  N_loc <- NN$new(B=B_saved_neuralnetwork)
  N_loc$W <- W_saved_neuralnetwork
  N_loc$d <- d_saved_neuralnetwork
  return(N_loc)
}

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
