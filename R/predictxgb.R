#' Predict Method for XtremeBoost Objects
#'
#' @description
#' Generates predictions from an \code{xtremeBoost} object returned by \code{xboosting()}
#' or accessible via \code{result$boosting_ensemble} from \code{boost_mem()}.
#'
#' The prediction accumulates contributions from all trees:
#' \deqn{\hat{f}(x) = \hat{f}_0 + \sum_{m=1}^{M} \hat{\phi}_m(x)}
#' where \eqn{\hat{f}_0 = \bar{y}} is the mean-based initialisation and each
#' \eqn{\hat{\phi}_m} is the output of the \eqn{m}-th fitted tree (with the learning
#' rate already absorbed by XGBoost internally).
#'
#' This function predicts the \strong{boosting component only} — it does not add
#' area-level random effects. To obtain small area estimates, add the relevant
#' random effect \eqn{\hat{b}_i} from \code{boost_mem()}'s \code{raneffs} output.
#'
#' @param object an \code{xtremeBoost} object (from \code{xboosting()} or
#'   \code{boost_mem()$boosting_ensemble}).
#' @param newdata a \code{data.frame} containing the same predictor columns used
#'   during training.
#' @param n.trees number of trees to use for prediction. Must be \eqn{\leq} the
#'   number of trees in \code{object}. Use the same value as \code{n.trees} passed
#'   to \code{boost_mem()} for in-sample consistency.
#' @param ... currently unused.
#'
#' @returns a numeric vector of predicted values, one per row of \code{newdata}.
#'
#' @examples
#' \donttest{
#' set.seed(42)
#' df <- data.frame(y = rnorm(100), x1 = rnorm(100), x2 = rnorm(100),
#'                  area = rep(1:10, each = 10))
#'
#' result <- boost_mem(y ~ x1 + x2, data = df, random = ~ 1 | area,
#'                     n.trees = 50, maxIter_memboost = 10)
#'
#' # Predict on new data (boosting component only)
#' new_df <- data.frame(x1 = rnorm(5), x2 = rnorm(5))
#' preds  <- predict.xgb(result$boosting_ensemble, newdata = new_df, n.trees = 50)
#' }
#'
#' @importFrom stats model.frame terms reformulate predict formula
#' @references Marie Salditt, Sarah Humberg & Steffen Nestler (2023) Gradient Tree Boosting for Hierarchical Data, Multivariate Behavioral Research, 58:5, 911-937, DOI: 10.1080/00273171.2022.2146638
#' @export
#'
#'
predict.xgb <- function(object, newdata, n.trees, ...) {
  # Ensure newdata is in the correct format
  PredNames <- attr(stats::terms(object$formula), "term.labels")
  newdata <- model.frame(terms(reformulate(PredNames)), data = newdata)
  # Get shrinkage value
  shrinkage <- object$shrinkage
  # Create DMatrix for newdata
  dnew <- Matrix::sparse.model.matrix( ~ ., data = newdata[, PredNames])[, -1]

  # Generate predictions
  preds <- sapply(1:n.trees, function(i) {
    predict(object$models[[i]], newdata = dnew)
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
