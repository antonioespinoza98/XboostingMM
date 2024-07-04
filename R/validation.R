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
#' @importFrom srvyr as_survey_design
#' @importFrom stats as.formula confint
#' @importFrom dplyr summarise group_by
#' @importFrom survey svyby svymean
#' @importFrom data.table :=
#' @importFrom data.table data.table
#' @importFrom dplyr sym summarise group_by
#' @importFrom ggplot2 ggplot aes geom_point scale_color_manual labs geom_errorbar theme_minimal
#' @return Returns a list of values and a \code{ggplot()} object with the Direct Estimates.
#' @export
#'
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
