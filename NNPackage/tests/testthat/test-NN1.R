testthat::test_that("initialize", {
  N_test <- NN$new(B=c(1,10,10,10,1))
  testthat::expect_true(is.R6(NN$new(B=c(1,30,1))))
  testthat::expect_equal(N_test$L, 3)
  testthat::expect_length(N_test$W, 4)
  testthat::expect_equal(dim(N_test$W[["A"]]), c(1,10))
  testthat::expect_length(N_test$d, 4)
  testthat::expect_length(N_test$B, 5)
  testthat::expect_length(N_test$f(1:5), 5)
  testthat::expect_length(N_test$del_f(1:5), 5)
})

testthat::test_that("Class Methods Regression", {
  N_test <- NN$new(B=c(1,10,10,10,1))
  testthat::expect_length(N_test$ffprop(1), 1)
  testthat::expect_length(N_test$ffprop(1:5), 5)
  testthat::expect_length(N_test$eval_till_layer(x=1,Layer=3), 10)
  testthat::expect_length(N_test$eval_till_layer_z(x=1,Layer=3), 10)
})

testthat::test_that("Class Methods Classification", {
  N_test <- NN$new(B=c(2,10,10,10,2))
  testthat::expect_length(N_test$ffprop_clas(matrix(1:4, nrow=2)), 4)
  testthat::expect_length(N_test$ffprop_clas(matrix(1:6, nrow=2)), 6)
})


