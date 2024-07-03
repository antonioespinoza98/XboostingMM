#' Log-Likelihood convergence function
#'
#' @description
#' Function that takes the estimates of the `boost_mem()` and checks for convergence
#'
#' @param Z An \eqn{n_i \times k} matrix representing the random effects
#' @param ID Domains of random effects
#' @param bhat random effects
#' @param ehat Error terms
#' @param UniqueID Unique identifier
#' @param NID Number of domains
#' @param D An \eqn{k \times k} covariance matrix
#' @param Sigma2 Variance of error terms
#'
#' @references Marie Salditt, Sarah Humberg & Steffen Nestler (2023) Gradient Tree Boosting for Hierarchical Data, Multivariate Behavioral Research, 58:5, 911-937, DOI: 10.1080/00273171.2022.2146638
#' @return Value of GLL
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
