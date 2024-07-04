#' Log-Likelihood convergence function
#'
#' @description
#' Function that takes the estimates of the \code{boost_mem()} and checks for convergence
#'
#' @param Z An \eqn{n_i \times k} matrix representing the random effects
#' @param ID Domains of random effects
#' @param bhat random effects
#' @param ehat Error terms
#' @param UniqueID Unique identifier
#' @param NID Number of domains
#' @param D An \eqn{k \times k} covariance matrix
#' @param Sigma2 Variance of error terms
#' \deqn{GLL(f, b_i|y_i) = \sum_{i=1}^n \left[ \left( y_i - f(X_i) - Z_i b_i \right)^T \Sigma_i^{-1} \left( y_i - f(X_i) - Z_i b_i \right) + b_i^T D^{-1} b_i + \log(\left| D \right|) + \log(\left| \Sigma_i \right|) \right]}
#' @references
#' Marie Salditt, Sarah Humberg & Steffen Nestler (2023) Gradient Tree Boosting for Hierarchical Data,
#' Multivariate Behavioral Research, 58:5, 911-937, DOI: 10.1080/00273171.2022.2146638
#'
#' Krennmair, P., & Schmid, T. (2022). Flexible Domain Prediction using Mixed Effects Random Forests.
#' Applied Statistics/Journal Of The Royal Statistical Society. Series C, Applied Statistics, 71(5), 1865-1894. https://doi.org/10.1111/rssc.12600
#'
#' @return Value of GLL
#' @export
#'
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
  logD <- as.numeric(determinant(D, logarithm = TRUE)$modulus)
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
    logR <- as.numeric(determinant(Ri, logarithm = TRUE)$modulus)
    #- compute ll:
    ll <- ll + logD + logR + t(ei) %*% InvRi %*% (ei) + t(bhat[ii, ]) %*%
      invD %*% bhat[ii, ]
  }
  return(-ll)
}
