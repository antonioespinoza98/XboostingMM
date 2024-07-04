library(testthat)
library(xgboost)
library(Matrix)

# Create a simple dataset for testing
set.seed(123)
data <- data.frame(
  x1 = rnorm(100),
  x2 = rnorm(100),
  y = rnorm(100)
)

# Define the formula
formula <- y ~ x1 + x2

# Run the xboosting function and store the result
result <- xboosting(
  formula = formula,
  loss = "reg:squarederror",
  data = data,
  n.trees = 10,
  shrinkage = 0.1,
  interaction.depth = 1,
  minsplit = 20,
  lambda = 1,
  alpha = 0,
  subsample = 0.5
)

# Define the test case
test_that("xboosting function works", {
  expect_s3_class(result, "xtremeBoost")
  expect_equal(length(result$models), 10)
  expect_true(all(!is.na(result$fhat)))
  expect_equal(result$n.trees, 10)
  expect_equal(result$shrinkage, 0.1)
  expect_equal(result$formula, formula)
  expect_true("formula" %in% names(result))
  expect_true("fhat" %in% names(result))
  expect_true("models" %in% names(result))
  expect_true("pred_matrix" %in% names(result))
  expect_true("finit" %in% names(result))
})
