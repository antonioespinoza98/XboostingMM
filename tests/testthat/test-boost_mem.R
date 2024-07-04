library(testthat)
library(xgboost)
library(Matrix)

# Create a simple dataset for testing
set.seed(123)
data <- data.frame(
  x1 = rnorm(100),
  x2 = rnorm(100),
  group = rep(1:10, each = 10),
  y = rnorm(100)
)

# Define the formula and random effect
formula <- y ~ x1 + x2
random <- ~ 1 | group

# Run the boost_mem function and store the result
result <- boost_mem(
  formula = formula,
  data = data,
  random = random,
  shrinkage = 0.3,
  loss = "reg:squarederror",
  interaction.depth = 5,
  n.trees = 10,
  minsplit = 10,
  subsample = 0.5,
  lambda = 1,
  alpha = 0,
  conv_memboost = 0.001,
  maxIter_memboost = 10,
  minIter_memboost = 0,
  verbose_memboost = FALSE
)

# Define the test case
test_that("boost_mem function works", {
  expect_s3_class(result, "XtremeRMM")
  expect_true(all(!is.na(result$fhat)))
  expect_equal(result$noIterations <= 10, TRUE)
  expect_true("boosting_ensemble" %in% names(result))
  expect_true("var_random_effects" %in% names(result))
  expect_true("errorVar" %in% names(result))
  expect_true("logLik" %in% names(result))
  expect_true("raneffs" %in% names(result))
  expect_true("errorTerms" %in% names(result))
  expect_true("fhat" %in% names(result))
  expect_true("noIterations" %in% names(result))
  expect_true("convWarning" %in% names(result))
  expect_true("means.Ystar" %in% names(result))
  expect_true("means.fhat" %in% names(result))
  expect_true("means.ranint" %in% names(result))
  expect_true("DhatList" %in% names(result))
  expect_true("errorVarList" %in% names(result))
})
