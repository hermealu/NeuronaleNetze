testthat::test_that("sigmoid", {
  testthat::expect_equal(sigmoid(1), 0.73105858)
  testthat::expect_equal(length(sigmoid(1:5)), 5)
})

testthat::test_that("del_sigmoid", {
  testthat::expect_equal(del_sigmoid(0), 0.25)
  testthat::expect_equal(length(del_sigmoid(1:5)), 5)
})

testthat::test_that("softmax", {
  testthat::expect_equal(softmax(1), 1)
  testthat::expect_equal(length(softmax(1:5)), 5)
})

testthat::test_that("tanh", {
  testthat::expect_equal(tanh(0), 0)
  testthat::expect_equal(length(tanh(1:5)), 5)
})

testthat::test_that("del_tanh", {
  testthat::expect_equal(del_tanh(0), 1)
  testthat::expect_equal(length(del_tanh(1:5)), 5)
})

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


