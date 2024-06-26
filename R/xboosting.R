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

#' Xboosting
#'
#' @param formula
#' @param data
#' @param loss
#' @param n.trees
#' @param shrinkage
#' @param interaction.depth
#' @param minsplit
#' @param subsample
#'
#' @return
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
                      subsample = 0.5){


  # arrangements ------------------------------------------------------------
  #- get predictive variable names
  PredNames <- attr(stats::terms(formula), "term.labels")
  # - filter data by keeping only predictive values
  X <- model.frame(terms(reformulate(PredNames)), data = data)
  # Since XGBoost only takes matrix-wise objects, we need to convert it to matrix
  # Also, categorical variables must be passed as numerical (independent variables)
  X_matrix <- Matrix::sparse.model.matrix( ~., data = data[,PredNames])[,-1]

  #- get y
  OutcomeName <- formula[[2]]
  y <- data[, toString(OutcomeName)]

  # Set XGBoost parameters --------------------------------------------------
  params <- list(
    objective = loss,
    eta = shrinkage,
    max_depth = interaction.depth,
    min_child_weight = minsplit
    # subsample = subsample,
    # colsample_bytree = colsample_bytree
  )

  # Train XGBoost model -----------------------------------------------------

  models <- list()
  preds_matrix <- matrix(NA, nrow = nrow(X), ncol = n.trees)
  n <- nrow(data)
  # - initial values for f, in this case. We've modified the XGBoost
  # - algorithm so we start off with the mean as initial value, instead of 0.
  f <- rep(mean(y), n)
  f0 <- unique(f)
  for(i in 1:n.trees){
    # subsampling at each iteration to reduce overfitting
    inbag.idx <- sample(n, replace = F, size = subsample*n)
    ysub <- y[inbag.idx]
    Xsub <- data.frame(X[inbag.idx, ])
    colnames(Xsub) <- colnames(X)
    fsub <- f[inbag.idx]
    # Get residuals
    dsub <- ysub - fsub
    r <- dsub
    Xsub_matrix <- Matrix::sparse.model.matrix( ~., data = Xsub[,PredNames])[,-1]
    # Get the i trees
    model <- xgboost::xgboost(params = params,
                              data = Xsub_matrix,
                              label = r,
                              nrounds = 1,
                              verbose = 0)

    cat("Estimating model:", i, "\n")
    # - Save models for future prediction
    models[[i]] <- model
    # - get prediction for f residuals
    pred <- predict(model, newdata = X_matrix)
    # - update overall model
    f <- f + shrinkage * pred
    preds_matrix[,i] <- pred
  }

  # Return the model and predictions ----------------------------------------

  # preds <- predict(model, newdata = X)
  out <- list(models = models, pred_matrix = preds_matrix,
              fhat = f, formula = formula, finit = f0, n.trees = n.trees,
              shrinkage = shrinkage)
  class(out) <- "xtremeBoost"
  return(out)
}


#' XGBoost prediction
#'
#' @param object
#' @param newdata
#' @param n.trees
#'
#' @return
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
  dnew <- Matrix::sparse.model.matrix( ~., data = newdata[,PredNames])[,-1]

  # Generate predictions
  preds <- sapply(1:n.trees, function(i){
    shrinkage * predict(object$models[[i]],
                        newdata = dnew
    )
  })

  # Get initial F_0
  f0 <- object$finit
  # - Get a prediction value
  if ( is.null( dim(preds) ) ){
    fit_pred <- f0 + sum(preds)
  } else {
    fit_pred <- f0 + apply( preds, 1, sum )
  }


  return(fit_pred)
}

#' GLL function
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
#' @return
#' @export
#'
#' @examples
mem_boost_gll <- function( Z = NULL, ID = NULL, bhat = NULL,
                           ehat = NULL, UniqueID = NULL, NID = NULL,
                           D = NULL, Sigma2 = NULL )
{
  #- define ll:
  ll <- 0
  #- compute inverse and determinant of D:
  invD <- solve( D )
  logD <- as.numeric( determinant( D, log = TRUE )$modulus )
  #- let's go:
  for ( ii in 1:NID ) {
    #- get index variable:
    idx <- which( ID == UniqueID[ii] )
    #- get person data:
    ei <- ehat[idx]
    Zi <- Z[idx,,drop = FALSE]
    ni <- dim( Zi )[1]
    Ri <- diag( as.numeric( Sigma2 ), ni )
    #- compute inverse matrix and determinant:
    InvRi <- solve( Ri )
    logR <- as.numeric( determinant( Ri, log = TRUE )$modulus )
    #- compute ll:
    ll <- ll + logD + logR + t(ei)%*%InvRi%*%(ei) + t(bhat[ii,])%*%invD%*%bhat[ii,]
  }
  return(-ll)
}

#' Iteration function for Tree-based MM
#'
#' @param formula
#' @param data
#' @param random
#' @param shrinkage
#' @param loss
#' @param interaction.depth
#' @param n.trees
#' @param minsplit
#' @param subsample
#' @param conv_memboost
#' @param maxIter_memboost
#' @param minIter_memboost
#' @param verbose_memboost
#'
#' @return
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
    var_random_effects = Dhat,
    errorVar = Sigma2hat,
    logLik = llnew,
    raneffs = bhat,
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
