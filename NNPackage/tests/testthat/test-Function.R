test_that("saving a NN", {
  N_test <- NN$new(B=c(1,10,10,10,1))
})

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
