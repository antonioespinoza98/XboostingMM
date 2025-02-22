
<!-- README.md is generated from README.Rmd. Please edit that file -->

# XboostingMM

<!-- badges: start -->
<!-- badges: end -->

XboostingMM is a semi-parametric model that harnesses the
high-performance properties of XGBoost, including its speed,
scalability, and superior performance compared to other methods. By
integrating mixed-effects models for hierarchical data, this package
offers a flexible estimation approach that can bypass traditional linear
model assumptions and handle data limitations effectively. This makes it
particularly valuable for applications in Small Area Estimation.

The XboostingMM package comprises four key functions: `boost_mem()`,
`mem_boost_gll()`, `predict.xgb()`, and `xboosting()`.

1.  **`mem_boost_gll()`**: This function checks the convergence of the
    algorithm.
2.  **`xboosting()`**: This function estimates the XGBoost trees.
3.  **`boost_mem()`**: This function leverages the XGBoost trees for the
    estimation of random effects, iterating until convergence is
    reached.
4.  **`predict.xgb()`**: This function makes predictions using the
    XGBoost model.

To use the package, start by running `xboosting()` to estimate the
XGBoost trees. Then, use `boost_mem()` to apply these trees for random
effects estimation. The `mem_boost_gll()` function can be used to
monitor the convergence of the algorithm. Finally, `predict.xgb()` can
be utilized for making predictions with the trained model.

For further information regarding each function, please refer to the
function documentation.

## Installation

You can install the development version of XboostingMM like so:

``` r
devtools::install_github("antonioespinoza98/XboostingMM")
```

## References

Chen T, He T, Benesty M, Khotilovich V, Tang Y, Cho H, Chen K, Mitchell
R, Cano I, Zhou T, Li M, Xie J, Lin M, Geng Y, Li Y, Yuan J (2024).
*xgboost: Extreme Gradient Boosting*. R package version 1.7.7.1,
<https://CRAN.R-project.org/package=xgboost>.

Salditt, M., Humberg, S., & Nestler, S. (2023). Gradient Tree Boosting
for Hierarchical Data. Multivariate Behavioral Research, 58(5), 911–937.
<https://doi.org/10.1080/00273171.2022.2146638>
