# Extreme Gradient Boosting -----------------------------------------------
# =========================================================================
# - Author: Marco Espinoza
# - espinozamarco70@gmail.com
# - GitHub: antonioespinoza98
# - References:
# -   Chen T, He T, Benesty M, Khotilovich V, Tang Y, Cho H, Chen K, Mitchell R, Cano I,
# Zhou T, Li M, Xie J, Lin M, Geng Y, Li Y, Yuan J (2024). _xgboost: Extreme
# Gradient Boosting_. R package version 1.7.7.1,
# <https://CRAN.R-project.org/package=xgboost>.

# - Marie Salditt, Sarah Humberg & Steffen Nestler (2023) Gradient Tree Boosting for
# Hierarchical Data, Multivariate Behavioral Research, 58:5, 911-937, DOI:
# 10.1080/00273171.2022.2146638
# =========================================================================

#' Computes Extreme Gradient Boosting Trees
#'
#' @description
#' Relies on the Extreme Gradient Boosting model
#' to generate the decision trees and subsequently an assembly of m trees.
#'
#' Initialize model with a constant value:
#' \deqn{\hat{f}_0(x) = \arg \min \sum_{i=1}^N L(y_i, \theta)}
#'
#' Compute gradients and hessians.
#'
#' fit a weak learner:
#'
#' \deqn{\hat{\phi}_m = \sum_{i =1}^{Im} \phi_{im}I(x \in R_{im})}
#'
#' Update model:
#'
#' \deqn{\hat{f}_{m-1}(x) + \eta \cdot \hat{\phi}_m}
#'
#' @param formula an object of class formula
#' @param data list or environment (or object coercible by \code{as.data.frame} to a data frame) containing the variables in the model.
#' @param loss by default it uses "reg:squarederror" more functions available in the \link[xgboost]{xgboost} documentation. Users, can pass a self-defined function to it.
#' @param n.trees number of trees to be generated. Default: 100
#' @param shrinkage learning rate. Default: 0.1
#' @param interaction.depth maximum depth of the trees. Default: 1
#' @param minsplit minimum observations for a tree to split. Default: 20
#' @param lambda regularization term on weights. Default: 1
#' @param alpha regularization term on weights. Default: 0
#' @param subsample sub-sample size for the tree training: Default: 0.5
#'
#' @importFrom stats model.frame terms reformulate predict
#' @importFrom utils txtProgressBar setTxtProgressBar
#' @references
#'
#' Chen T, He T, Benesty M, Khotilovich V, Tang Y, Cho H, Chen K, Mitchell R,
#' Cano I,Zhou T, Li M, Xie J, Lin M, Geng Y, Li Y, Yuan J (2024).
#'  _xgboost: Extreme Gradient Boosting_. R package version 1.7.7.1,<https://CRAN.R-project.org/package=xgboost>.
#'
#'
#' Marie Salditt, Sarah Humberg & Steffen Nestler (2023)
#' Gradient Tree Boosting for Hierarchical Data, Multivariate Behavioral Research,
#' 58:5, 911-937, DOI: 10.1080/00273171.2022.2146638
#'
#' Corral, P., Henderson, H., & Segovia, S. (2023).
#' Poverty Mapping in the Age of Machine Learning.
#' World Bank policy research working paper. https://doi.org/10.1596/1813-9450-10429
#'
#' @returns an xtremeBoost object
#' @export
#'
#'
xboosting <- function(formula,
                      data = NULL,
                      loss = "reg:squarederror",
                      n.trees = 100,
                      shrinkage = 0.1,
                      interaction.depth = 1,
                      minsplit = 20,
                      lambda = 1,
                      alpha = 0,
                      subsample = 0.5) {
  # -- CHECKS

  if (missing(subsample)) {
    warning("Subsample using default value of 0.5")
  }
  if (missing(loss)) {
    warning("Loss function using default value.")
  }
  if (missing(data)) {
    stop("Must provide a dataset.")
  }
  if (missing(n.trees) |
      missing(shrinkage) |
      missing(interaction.depth) | missing(minsplit)) {
    warning("Using default values trees, learning rate, etc. Please refer to documentation.")
  }

  # Check if data frame is data.frame()
  if (!is.data.frame(data)) {
    stop("Data must be a data.frame class object.")
  }
  # arrangements ------------------------------------------------------------
  #- get predictive variable names
  PredNames <- attr(stats::terms(formula), "term.labels")
  # - filter data by keeping only predictive values
  X <- model.frame(terms(reformulate(PredNames)), data = data)
  # Since XGBoost only takes matrix-wise objects, we need to convert it to matrix
  # Also, categorical variables must be passed as numerical (independent variables)
  X_matrix <- Matrix::sparse.model.matrix(~ ., data = data[, PredNames])[, -1]

  #- get y
  OutcomeName <- formula[[2]]
  y <- data[, toString(OutcomeName)]

  # Set XGBoost parameters --------------------------------------------------
  params <- list(
    objective = loss,
    eta = shrinkage,
    max_depth = interaction.depth,
    min_child_weight = minsplit,
    lambda = lambda,
    alpha = alpha
  )

  # Train XGBoost model -----------------------------------------------------

  models <- list()
  preds_matrix <- matrix(NA, nrow = nrow(X), ncol = n.trees)
  n <- nrow(data)
  # - initial values for f, in this case. We've modified the XGBoost
  # - algorithm so we start off with the mean as initial value, instead of 0.
  f <- rep(mean(y), n)
  f0 <- unique(f)
  cat("Estimating model", "\n")
  progress_est <- txtProgressBar(min = 0, max = n.trees, style = 3)
  for (i in 1:n.trees) {
    # subsampling at each iteration to reduce overfitting
    inbag.idx <- sample(n, replace = F, size = subsample * n)
    ysub <- y[inbag.idx]
    Xsub <- data.frame(X[inbag.idx, ])
    colnames(Xsub) <- colnames(X)
    fsub <- f[inbag.idx]
    # Get residuals
    dsub <- ysub - fsub
    r <- dsub
    Xsub_matrix <- Matrix::sparse.model.matrix(~ ., data = Xsub[, PredNames])[, -1]
    # Get the i trees
    model <- xgboost::xgboost(
      params = params,
      data = Xsub_matrix,
      label = r,
      nrounds = 1,
      verbose = 0
    )

    setTxtProgressBar(progress_est, i)
    # - Save models for future prediction
    models[[i]] <- model
    # - get prediction for f residuals
    pred <- predict(model, newdata = X_matrix)
    # - update overall model
    f <- f + shrinkage * pred
    preds_matrix[, i] <- pred
  }
  close(progress_est)

  # Return the model and predictions ----------------------------------------

  # preds <- predict(model, newdata = X)
  out <- list(
    models = models,
    pred_matrix = preds_matrix,
    fhat = f,
    formula = formula,
    finit = f0,
    n.trees = n.trees,
    shrinkage = shrinkage
  )
  class(out) <- "xtremeBoost"
  return(out)
}
