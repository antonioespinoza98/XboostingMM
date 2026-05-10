#' Iteration function for Tree-based MM
#'
#' @description
#' Main entry point for the Extreme Boosting Mixed-Effects Model (XboostingMM).
#' Combines XGBoost gradient tree boosting with a linear mixed-effects model via
#' an EM-like iterative algorithm, making it suitable for Small Area Estimation (SAE)
#' and other hierarchical data settings.
#'
#' The unit-level model is:
#' \deqn{y_i = f(X_i) + Z_i b_i + e_i}
#' where \eqn{f(X_i)} is estimated by XGBoost, \eqn{b_i \sim \mathcal{N}(0, D)} are
#' area-level random effects, and \eqn{e_i \sim \mathcal{N}(0, \sigma^2 I)}.
#'
#' At each iteration the algorithm:
#' \enumerate{
#'   \item Subtracts current random-effect estimates to form a transformed outcome
#'         \eqn{\bar{y}_i = y_i - Z_i \hat{b}_i}, then fits XGBoost to \eqn{\bar{y}_i}.
#'   \item Updates random effects via the BLUP formula
#'         \eqn{\hat{b}_i = \hat{D} Z_i^T \hat{V}_i^{-1} [y_i - \hat{f}(X_i)]}.
#'   \item Updates variance components \eqn{\hat{D}} and \eqn{\hat{\sigma}^2} via
#'         closed-form EM M-step equations.
#'   \item Checks convergence using the Generalised Log-Likelihood (see \link{mem_boost_gll}).
#' }
#'
#' @param formula a standard R formula of the form \code{outcome ~ predictor1 + predictor2 + ...}
#' @param data a \code{data.frame} containing all variables referenced in \code{formula} and \code{random}.
#' @param random a one-sided formula specifying the random intercept grouping variable,
#'   written as \code{~ 1 | group_variable}. Only random intercepts are currently supported.
#' @param shrinkage XGBoost learning rate (\code{eta}). Smaller values require more trees
#'   but reduce overfitting. Default: \code{0.3}.
#' @param loss XGBoost objective function. Default: \code{"reg:squarederror"}.
#'   See the \code{xgboost} documentation for alternatives.
#' @param interaction.depth maximum depth of each tree. \code{1} (stumps) is recommended
#'   for SAE where small area sample sizes are small. Default: \code{1}.
#' @param n.trees number of boosting trees fitted per EM iteration. Default: \code{100}.
#' @param minsplit minimum number of observations required to split a node
#'   (\code{min_child_weight} in XGBoost). Default: \code{20}.
#' @param subsample fraction of training rows sampled at each tree. Values below 1
#'   introduce stochastic gradient boosting and reduce overfitting. Default: \code{0.5}.
#' @param lambda L2 regularization on leaf weights. Default: \code{1}.
#' @param alpha L1 regularization on leaf weights. Default: \code{0}.
#' @param weight optional numeric vector of per-row weights passed to XGBoost.
#' @param conv_memboost convergence threshold on the relative change in GLL between
#'   successive EM iterations: \eqn{|\Delta\text{GLL}| / |\text{GLL}|} < \code{conv_memboost}.
#'   Default: \code{0.001}.
#' @param maxIter_memboost maximum number of EM iterations before stopping regardless
#'   of convergence. Default: \code{100}.
#' @param minIter_memboost minimum number of EM iterations before convergence is checked.
#'   Default: \code{0}.
#' @param verbose_memboost if \code{TRUE}, prints the GLL and iteration number at each
#'   EM step. Default: \code{FALSE}.
#'
#' @returns An object of class \code{XtremeRMM} (a named list) with components:
#' \describe{
#'   \item{\code{fhat}}{numeric vector of fitted values \eqn{\hat{f}(X)} for every row
#'     in \code{data} (boosting component only, without random effects).}
#'   \item{\code{raneffs}}{matrix of estimated random effects \eqn{\hat{b}_i},
#'     with one row per area and one column per random effect.}
#'   \item{\code{var_random_effects}}{estimated covariance matrix \eqn{\hat{D}} of
#'     the random effects.}
#'   \item{\code{errorVar}}{estimated error variance \eqn{\hat{\sigma}^2}.}
#'   \item{\code{mse_approx}}{named numeric vector of approximate MSE per area,
#'     computed as the g1 BLUP variance term
#'     \eqn{g_{1i} = \mathrm{tr}(\hat{D} - \hat{D} Z_i^T \hat{V}_i^{-1} Z_i \hat{D})}.
#'     This is a lower bound (captures uncertainty in \eqn{\hat{b}_i} only).}
#'   \item{\code{logLik}}{final GLL value at convergence.}
#'   \item{\code{errorTerms}}{residuals \eqn{y_i - \hat{f}(X_i) - Z_i \hat{b}_i}.}
#'   \item{\code{boosting_ensemble}}{the \code{xtremeBoost} object from the final
#'     boosting step, used for out-of-sample prediction via \code{predict.xgb()}.}
#'   \item{\code{noIterations}}{number of EM iterations performed.}
#'   \item{\code{convWarning}}{logical; \code{TRUE} if \code{maxIter_memboost} was
#'     reached before convergence.}
#'   \item{\code{means.Ystar}, \code{means.fhat}, \code{means.ranint}}{vectors tracking
#'     the mean of the transformed outcome, boosting predictions, and random intercept
#'     across EM iterations — useful for diagnosing convergence.}
#'   \item{\code{DhatList}, \code{errorVarList}}{lists of \eqn{\hat{D}} and
#'     \eqn{\hat{\sigma}^2} estimates at each iteration.}
#' }
#'
#' @examples
#' \donttest{
#' set.seed(42)
#' df <- data.frame(
#'   y     = rnorm(200),
#'   x1    = rnorm(200),
#'   x2    = rnorm(200),
#'   area  = rep(1:20, each = 10)
#' )
#'
#' result <- boost_mem(
#'   formula           = y ~ x1 + x2,
#'   data              = df,
#'   random            = ~ 1 | area,
#'   shrinkage         = 0.1,
#'   interaction.depth = 1,
#'   n.trees           = 50,
#'   conv_memboost     = 0.001,
#'   maxIter_memboost  = 30
#' )
#'
#' # Boosting fit + random effects per area
#' area_est <- tapply(result$fhat, df$area, mean) + result$raneffs[, 1]
#'
#' # Approximate standard error per area
#' area_se <- sqrt(result$mse_approx)
#'
#' # Out-of-sample prediction (boosting component only)
#' new_df <- data.frame(x1 = rnorm(5), x2 = rnorm(5))
#' preds  <- predict.xgb(result$boosting_ensemble, newdata = new_df, n.trees = 50)
#' }
#'
#' @importFrom stats model.frame terms reformulate model.matrix update.formula as.formula
#' @references Marie Salditt, Sarah Humberg & Steffen Nestler (2023) Gradient Tree Boosting for Hierarchical Data, Multivariate Behavioral Research, 58:5, 911-937, DOI: 10.1080/00273171.2022.2146638
#' @export
#'
#'
boost_mem <- function(formula,
                      data = NULL,
                      random = NULL,
                      shrinkage = 0.3,
                      loss = "reg:squarederror",
                      interaction.depth = 1,
                      n.trees = 100,
                      minsplit = 20,
                      subsample = 0.5,
                      lambda = 1,
                      alpha = 0,
                      weight = NULL,
                      conv_memboost = 0.001,
                      maxIter_memboost = 100,
                      minIter_memboost = 0,
                      verbose_memboost = FALSE) {
  # STEP 0: PREPARATION

  #- Get X
  # - Get predictive variable
  PredNames <- attr(stats::terms(formula), "term.labels")
  # - filter data to keep only a frame with predictive variables
  X <- model.frame(terms(reformulate(PredNames)), data = data)

  #- Get Y
  OutcomeName <- formula[[2]]
  if (length(OutcomeName) > 1) {
    OutcomeName <- OutcomeName[3]
  }
  Y <- data[, toString(OutcomeName)]

  #- Get ID and Z
  FormulaRandom <- random
  NamesRandom <- attr(stats::terms(FormulaRandom), "term.labels")
  NamesRandom <- gsub("\\s", "", NamesRandom) # delete all spaces
  HasBar <- grepl("|", NamesRandom, fixed = TRUE)
  if (any(!HasBar)) {
    stop("'random' must contain a grouping variable after the | symbol.")
  }
  FormulaRandomSplit <- strsplit(NamesRandom, "\\|", perl = FALSE)[[1]]
  IdVar <- FormulaRandomSplit[2]
  if (!(IdVar %in% colnames(data))) {
    stop("Level-2 identifier not found.")
  }
  # Get random variable vector
  ID <- data[, toString(IdVar)]
  FormulaRandom <- stats::formula(paste0("~", FormulaRandomSplit[1], collapse =
                                           ""))
  # Matriz de diseño
  Z <- model.matrix(FormulaRandom, data)

  #- Some initial specifications
  TotalObs <- dim(data)[1]
  UniqueID <- unique(ID)
  NID <- length(UniqueID)
  p <- dim(Z)[2]
  # Covariance matrix
  Dhat <- diag(1, p)
  Sigma2hat <- 1
  bhat <- matrix(0, nrow = NID , ncol = p)
  ehat <- rep(0, TotalObs)

  # Some preparations for saving the means of the transformed outcome,
  # of the random intercept and of the boosting ensemble predictions
  # as well as the estimated covariance matrix of the random effects
  # and the estimated error variance per iteration
  means.Ystar <- NULL
  means.fhat <- NULL
  means.ranint <- NULL
  DhatList <- list()
  errorVarList <- list()

  # Dataframe and initializations for the while-loop:
  newdata <- data
  toIterate <- TRUE
  convWarning <- FALSE
  noIterations <- 0
  llnew <- 0
  absDiffLogLik <- Inf

  #- step 5: Start the while loop
  while (toIterate) {
    #- Count number of iterations
    noIterations <- noIterations + 1
    llold <- llnew

    # STEP 1a: Get an estimate f
    #- (i): compute the transformed outcome Ystar
    Ystar <- rep(0, TotalObs)
    for (ii in 1:NID) {
      #- get index variable:
      idx <- which(ID == UniqueID[ii])
      #- get relevant matrices and vectors:
      Yi <- Y[idx]
      Zi <- Z[idx, , drop = FALSE]
      bi <- bhat[ii, ]
      Ystar[idx] <- Yi - Zi %*% bi
    }

    meanYstar <- mean(Ystar)
    means.Ystar <- rbind(means.Ystar, meanYstar)

    #- (ii): estimate f via gradient tree boosting
    newdata[, "Ystar"] <- Ystar
    working_formula <- update.formula(formula, as.formula('Ystar ~ .'))

    tmpGTB <- xboosting(
      formula = working_formula,
      data = newdata,
      loss = loss,
      n.trees = n.trees,
      shrinkage = shrinkage,
      interaction.depth = interaction.depth,
      minsplit = minsplit,
      alpha = alpha,
      lambda = lambda,
      weight = weight,
      subsample = subsample
    )


    #- get the boosting ensemble predictions
    fhat <- predict.xgb(tmpGTB, newdata = data, n.trees = n.trees)
    means.fhat <- rbind(means.fhat, mean(fhat))


    # STEP 1b and STEP 2: Update the random effects and variance components
    #- (iii): compute new bs and new epsilons
    for (ii in 1:NID) {
      #- get index variable:
      idx <- which(ID == UniqueID[ii])
      #- get relevant matrices and vectors:
      Yi <- Y[idx]
      fhati <- fhat[idx]
      Zi <- Z[idx, , drop = FALSE]
      ni <- dim(Zi)[1]
      Ri <- diag(as.numeric(Sigma2hat), ni)
      Vi <- Zi %*% Dhat %*% t(Zi) + Ri
      InvVi <- solve(Vi)
      #- compute new bhati and new epsilons:
      bhat[ii, ] <- Dhat %*% t(Zi) %*% (InvVi %*% (Yi - fhati))
      ehat[idx] <- Yi - fhati - Zi %*% bhat[ii, ]
      }

      #- (iv): update Dhat and Sigma2hat
      DhatNew <- diag(0, p)
      Sigma2hatNew <- 0
      for (ii in 1:NID) {
        #- get index variable:
        idx <- which(ID == UniqueID[ii])
        #- get relevant matrices and vectors:
        Yi <- Y[idx]
        Zi <- Z[idx, , drop = FALSE]
        ni <- dim(Zi)[1]
        Ri <- diag(as.numeric(Sigma2hat), ni)
        Vi <- Zi %*% Dhat %*% t(Zi) + Ri
        InvVi <- solve(Vi)
        #- compute new variance components:
        DhatNew <- DhatNew + (bhat[ii, ] %*% t(bhat[ii, ]) + (Dhat - Dhat %*%
                                                                t(Zi) %*% InvVi %*% Zi %*% Dhat))
        tmpSigma <- as.numeric(Sigma2hat) * (ni - as.numeric(Sigma2hat) *
                                               sum(diag(InvVi)))
        Sigma2hatNew <- Sigma2hatNew + (t(ehat[idx]) %*% ehat[idx] + tmpSigma)
      }
      #- update matrices:
      Dhat <- DhatNew / NID
      Sigma2hat <- Sigma2hatNew / TotalObs


    means.ranint <- rbind(means.ranint, mean(bhat[, 1]))
    errorVarList[[noIterations]] <- Sigma2hat
    DhatList[[noIterations]] <- Dhat

    #- Compute GLLnew
    llnew <- mem_boost_gll(
      Z = Z,
      ID = ID,
      bhat = bhat,
      ehat = ehat,
      UniqueID = UniqueID,
      NID = NID,
      D = Dhat,
      Sigma2 = Sigma2hat
    )

    cat("Checking for convergence...", "\n")
    #- Verbose output:
    if (verbose_memboost) {
      h1 <- paste0("Loglikelihood: ",
                   round(llnew, 2),
                   " | No. iteration: ",
                   noIterations)
      cat(h1, "\n")
      utils::flush.console()
    }
    #- Leaving the while loop?
    if (noIterations > 1) {
      absDiffLogLik <- abs((llold - llnew) / llold)
    }

    if (noIterations > minIter_memboost &
        (absDiffLogLik < conv_memboost |
         noIterations >= maxIter_memboost)) {
      cat("algorithm converged after: ", noIterations, " iterations")
      toIterate <- FALSE
    }
  } # while

  if (absDiffLogLik >= conv_memboost) {
    warning("EM algorithm did not converge")
    convWarning <- TRUE
  }

  # Approximate MSE per area: g1 term (BLUP prediction variance).
  # This is a lower bound — it captures variance from estimating b_i but not
  # from estimating f or the variance components.
  mse_approx <- numeric(NID)
  for (ii in 1:NID) {
    idx <- which(ID == UniqueID[ii])
    Zi <- Z[idx, , drop = FALSE]
    ni <- dim(Zi)[1]
    Ri <- diag(as.numeric(Sigma2hat), ni)
    Vi <- Zi %*% Dhat %*% t(Zi) + Ri
    InvVi <- solve(Vi)
    g1 <- Dhat - Dhat %*% t(Zi) %*% InvVi %*% Zi %*% Dhat
    mse_approx[ii] <- if (p == 1) as.numeric(g1) else sum(diag(g1))
  }
  names(mse_approx) <- UniqueID

  #- output:
  out <- list(
    boosting_ensemble = tmpGTB,
    # Covariance matrix
    var_random_effects = Dhat,
    errorVar = Sigma2hat,
    logLik = llnew,
    raneffs = bhat,
    # error terms (residuals terms)
    errorTerms = ehat,
    fhat = fhat,
    mse_approx = mse_approx,
    noIterations = noIterations,
    convWarning = convWarning,
    means.Ystar = means.Ystar,
    means.fhat = means.fhat,
    means.ranint = means.ranint,
    DhatList = DhatList,
    errorVarList = errorVarList
  )
  class(out) <- "XtremeRMM"
  return(out)
}
