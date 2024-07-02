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
#' `xboosting()` relies on the Extreme Gradient Boosting model
#' to generate the decision trees and subsequently an assembly of m trees.
#'
#' @param formula an object of class formula
#' @param data list or environment (or object coercible by `as.data.frame` to a data frame) containing the variables in the model.
#' @param loss by default it uses "reg:squarederror" more functions available in the `xgboost` documentation. Users, can pass a self-defined function to it.
#' @param n.trees number of trees to be generated. Default: 100
#' @param shrinkage learning rate. Default: 0.1
#' @param interaction.depth maximum depth of the trees. Default: 1
#' @param minsplit minimum observations for a tree to split. Default: 20
#' @param subsample sub-sample size for the tree training: Default: 0.5
#'
#' @references Chen T, He T, Benesty M, Khotilovich V, Tang Y, Cho H, Chen K, Mitchell R, Cano I,Zhou T, Li M, Xie J, Lin M, Geng Y, Li Y, Yuan J (2024). _xgboost: Extreme Gradient Boosting_. R package version 1.7.7.1,<https://CRAN.R-project.org/package=xgboost>.
#' Marie Salditt, Sarah Humberg & Steffen Nestler (2023) Gradient Tree Boosting for Hierarchical Data, Multivariate Behavioral Research, 58:5, 911-937, DOI: 10.1080/00273171.2022.2146638
#'
#' @returns an xtremeBoost object
#' @export
#'
#' @examples
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


#' Performs a prediction from an XtremeBoost object
#'
#' @description
#' It takes an XtremeBoost model from the `xgboosting()` and performs a prediction
#'
#' @param object an XtremeBoost object
#' @param newdata data required to perform
#' @param n.trees number of trees to use from the object.
#'
#' @references Marie Salditt, Sarah Humberg & Steffen Nestler (2023) Gradient Tree Boosting for Hierarchical Data, Multivariate Behavioral Research, 58:5, 911-937, DOI: 10.1080/00273171.2022.2146638
#' @returns a vector with predictions
#' @export
#'
#' @examples
predict.xgb <- function(object, newdata, n.trees) {
  # Ensure newdata is in the correct format
  PredNames <- attr(stats::terms(formula), "term.labels")
  newdata <- model.frame(terms(reformulate(PredNames)), data = newdata)
  # Get shrinkage value
  shrinkage <- object$shrinkage
  # Create DMatrix for newdata
  dnew <- Matrix::sparse.model.matrix(~ ., data = newdata[, PredNames])[, -1]

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

#' Log-Likelihood convergence function
#'
#' @description
#' Function that takes the estimates of the `boost_mem()` and checks for convergence
#'
#' @param Z
#' @param ID
#' @param bhat
#' @param ehat
#' @param UniqueID
#' @param NID
#' @param D
#' @param Sigma2
#'
#' @references Marie Salditt, Sarah Humberg & Steffen Nestler (2023) Gradient Tree Boosting for Hierarchical Data, Multivariate Behavioral Research, 58:5, 911-937, DOI: 10.1080/00273171.2022.2146638
#' @return
#' @export
#'
#' @examples
mem_boost_gll <- function(Z = NULL,
                          ID = NULL,
                          bhat = NULL,
                          ehat = NULL,
                          UniqueID = NULL,
                          NID = NULL,
                          D = NULL,
                          Sigma2 = NULL)
{
  #- define ll:
  ll <- 0
  #- compute inverse and determinant of D:
  invD <- solve(D)
  logD <- as.numeric(determinant(D, log = TRUE)$modulus)
  #- let's go:
  for (ii in 1:NID) {
    #- get index variable:
    idx <- which(ID == UniqueID[ii])
    #- get person data:
    ei <- ehat[idx]
    Zi <- Z[idx, , drop = FALSE]
    ni <- dim(Zi)[1]
    Ri <- diag(as.numeric(Sigma2), ni)
    #- compute inverse matrix and determinant:
    InvRi <- solve(Ri)
    logR <- as.numeric(determinant(Ri, log = TRUE)$modulus)
    #- compute ll:
    ll <- ll + logD + logR + t(ei) %*% InvRi %*% (ei) + t(bhat[ii, ]) %*%
      invD %*% bhat[ii, ]
  }
  return(-ll)
}

#' Iteration function for Tree-based MM
#'
#' @description
#' Function that leverages `xboosting()` to estimate the trees.
#' `predict.xgb()` for prediction, and `mem_boost_gll()` for convergence.
#'
#' @param formula an object of class formula
#' @param data list or environment (or object coercible by `as.data.frame` to a data frame) containing the variables in the model.
#' @param random
#' @param shrinkage learning rate. Default: 0.1
#' @param loss by default it uses "reg:squarederror" more functions available in the `xgboost` documentation. Users, can pass a self-defined function to it.
#' @param interaction.depth maximum depth of the trees. Default: 1
#' @param n.trees number of trees to be generated. Default: 100
#' @param minsplit minimum observations for a tree to split. Default: 20
#' @param subsample sub-sample size for the tree training: Default: 0.5
#' @param conv_memboost Convergence threshold. Default: 0.001
#' @param maxIter_memboost maximum iterations. Default: 100
#' @param minIter_memboost minimum iterations. Default: 0
#' @param verbose_memboost Print information
#'
#' @references Marie Salditt, Sarah Humberg & Steffen Nestler (2023) Gradient Tree Boosting for Hierarchical Data, Multivariate Behavioral Research, 58:5, 911-937, DOI: 10.1080/00273171.2022.2146638
#' @returns list of values
#' @export
#'
#' @examples
boost_mem <- function(formula,
                      data = NULL,
                      random = NULL,
                      shrinkage = 0.3,
                      loss = "reg:squarederror",
                      interaction.depth = 20,
                      n.trees = 100,
                      minsplit = 20,
                      subsample = 0.5,
                      lambda = 1,
                      alpha = 0,
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
    formula <- update.formula(formula, as.formula('Ystar ~ .'))

    tmpGTB <- xboosting(
      formula = formula,
      data = newdata,
      loss = loss,
      n.trees = n.trees,
      shrinkage = shrinkage,
      interaction.depth = interaction.depth,
      minsplit = minsplit,
      alpha = alpha,
      lambda = lambda,
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
      # }

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

    }

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
    absDiffLogLik <- abs((llold - llnew) / llold)

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


#' Direct Estimate Validation
#'
#' @description
#' A function that fits the Direct Estimates in Small Area Estimation
#' and plots it against the estimates of the model.
#'
#'
#' @param test Test set used in the prediction of the fitted values (usually administrative records)
#' @param prediction Vector type object with the fitted values
#' @param validation_set Data set to fit Direct Estimates
#' @param weights Name of column with the weights for direct estimates
#' @param label response variable
#' @param region domain variable
#' @param model Model name for the plot
#'
#'
#' @return
#' @export
#'
#' @examples
validation <- function(test = NULL,
                       prediction = NULL,
                       validation_set = NULL,
                       weights = NULL,
                       label = NULL,
                       region = NULL,
                       model = NULL) {
  # --- REQUIREMENTS ---
  # Data set for predictions
  if (missing(test)) {
    stop("Must provide a test data set.")
  }
  # Check dimensions of vector and administrative record.
  if (dim(test)[1] != length(prediction)) {
    stop("Data set must have same length as prediction vector")
  }
  if (missing(prediction)) {
    stop("Must provide a prediction vector.")
  }
  if (!is.vector(prediction)) {
    stop("Prediction must be a vector. Check class()")
  }
  if (missing(weights)) {
    stop("Must provide a value for the weights.")
  }
  if (missing(label)) {
    stop("Must provide a value to be estimated.")
  }
  if (missing(region)) {
    stop("Must provide a value for the regions.")
  }

  # --- PROCESS ---
  # We add the prediction to the data set as a new column.
  test$pred <- prediction

  # --- DIRECT ESTIMATES ---
  # survey design
  diseno <- as_survey_design(.data = validation_set, weights = weights)
  # Mean
  formula_label <- as.formula(paste("~", label))
  formula_region <- as.formula(paste("~", region))

  media_est_srvyr <- svyby(formula_label, by = formula_region, design = diseno, svymean)

  # Confidence intervals
  confint_int <- confint(svyby(
    formula = formula_label,
    by = formula_region,
    design = diseno,
    svymean
  ))

  # --- CONFIDENCE INTERVALS ---
  confint_int <- data.table(confint_int, keep.rownames = TRUE)
  colnames(confint_int) <- c("region", "min_inter", "max_inter")


  Y <- media_est_srvyr[, toString(label)]

  confint_int[[label]] <- Y

  # --- PREDICTION ARRANGEMENTS ---
  prediction_data <- test |> group_by(!!sym(region)) |>
    summarise(!!sym(label) := mean(pred))

  # --- MAPPING DATA FRAME ---
  mapping <- rbind(media_est_srvyr[, -3], prediction_data)

  count_rep <- length(unique(validation_set$region))
  estim <- c(rep("Directo", times = count_rep),
             rep(model, times = count_rep))
  mapping <- cbind(mapping, estim)

  # --- PLOT ---
  plt1 <- mapping |>
    ggplot(aes(x = !!sym(region), y = !!sym(label))) + geom_point(aes(color = estim), size = 2, position = "jitter") +
    scale_color_manual(values = c("red", "green")) +
    labs(x = "Region", y = label, col = "Modelo") +
    geom_errorbar(data = confint_int, aes(x = region, ymin = min_inter, ymax = max_inter)) +
    theme_minimal()

  out <- list(direct_estimate = media_est_srvyr,
              conf_int = confint_int,
              plot = plt1)

  return(out)
}
