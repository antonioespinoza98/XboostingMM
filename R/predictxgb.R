#' Predict Method for XtremeBoost Objects
#'
#' @description
#' It takes an XtremeBoost model from the \code{xboosting()} and performs a prediction
#'
#' @param object an XtremeBoost object
#' @param newdata data required to perform
#' @param n.trees number of trees to use from the object.
#' @param ... additional arguments affecting the predictions produced.
#' @importFrom stats model.frame terms reformulate predict formula
#' @references Marie Salditt, Sarah Humberg & Steffen Nestler (2023) Gradient Tree Boosting for Hierarchical Data, Multivariate Behavioral Research, 58:5, 911-937, DOI: 10.1080/00273171.2022.2146638
#'
#' @returns a vector with predictions
#' @export
#'
#'
predict.xgb <- function(object, newdata, n.trees, ...) {
  # Ensure newdata is in the correct format
  PredNames <- attr(stats::terms(formula), "term.labels")
  newdata <- model.frame(terms(reformulate(PredNames)), data = newdata)
  # Get shrinkage value
  shrinkage <- object$shrinkage
  # Create DMatrix for newdata
  dnew <- Matrix::sparse.model.matrix( ~ ., data = newdata[, PredNames])[, -1]

  # Generate predictions
  preds <- sapply(1:n.trees, function(i) {
    shrinkage * predict(object$models[[i]], newdata = dnew)
  })

  # Get initial F_0
  f0 <- object$finit
  # - Get a prediction value
  if (is.null(dim(preds))) {
    fit_pred <- f0 + sum(preds)
  } else {
    fit_pred <- f0 + apply(preds, 1, sum)
  }


  return(fit_pred)
}
