#' Iteration function for Tree-based MM
#'
#' @description
#' Function that leverages `xboosting()` to estimate the trees.
#' `predict.xgb()` for prediction, and `mem_boost_gll()` for convergence.
#'
#' @param formula an object of class formula
#' @param data list or environment (or object coercible by `as.data.frame` to a data frame) containing the variables in the model.
#' @param random an object of class formula with the random intercept
#' @param shrinkage learning rate. Default: 0.1
#' @param loss by default it uses "reg:squarederror" more functions available in the `xgboost` documentation. Users, can pass a self-defined function to it.
#' @param interaction.depth maximum depth of the trees. Default: 1
#' @param n.trees number of trees to be generated. Default: 100
#' @param minsplit minimum observations for a tree to split. Default: 20
#' @param subsample sub-sample size for the tree training. Default: 0.5
#' @param lambda regularization term on weights. Default: 1
#' @param alpha regularization term on weights. Default: 0
#' @param conv_memboost Convergence threshold. Default: 0.001
#' @param maxIter_memboost maximum iterations. Default: 100
#' @param minIter_memboost minimum iterations. Default: 0
#' @param verbose_memboost Print information
#'
#' @importFrom stats model.frame terms reformulate model.matrix update.formula as.formula
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
