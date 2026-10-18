data_path <- "data/saheart.csv"
output_dir <- "saheart-analysis/artifacts"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

seed <- 432
fold_count <- 10L
test_fraction <- 0.20
predictors <- c(
  "sbp", "tobacco", "ldl", "adiposity", "famhist",
  "typea", "obesity", "alcohol", "age"
)
candidate_names <- c("LDA, all original predictors", "QDA, all original predictors")
data <- read.csv(data_path, stringsAsFactors = FALSE)
stopifnot(
  nrow(data) == 462L,
  identical(names(data), c(predictors[1:4], "famhist", predictors[6:9], "chd")),
  all(data$famhist %in% c("Absent", "Present")),
  all(data$chd %in% c(0, 1))
)

missing_counts <- colSums(is.na(data))
data$famhist <- ifelse(data$famhist == "Absent", 1, 2)
data$chd <- factor(data$chd, levels = c(0, 1))
data <- data[c(predictors, "chd")]
stopifnot(!anyNA(data))

capture_warnings <- function(expr) {
  messages <- character()
  value <- withCallingHandlers(expr, warning = function(warning) {
    messages <<- c(messages, conditionMessage(warning))
    invokeRestart("muffleWarning")
  })
  list(value = value, warnings = messages)
}

fit_predict <- function(method, training, testing) {
  formula <- stats::reformulate(predictors, response = "chd")
  if (method == "LDA") {
    fit <- MASS::lda(formula, data = training)
    prediction <- predict(fit, newdata = testing)$class
  } else {
    fit <- MASS::qda(formula, data = training)
    prediction <- predict(fit, newdata = testing)$class
  }
  factor(prediction, levels = levels(training$chd))
}

confusion_matrix <- function(actual, predicted) {
  table(
    actual = factor(actual, levels = c("0", "1")),
    predicted = factor(predicted, levels = c("0", "1"))
  )
}

set.seed(seed)
test_indices <- sort(unlist(lapply(levels(data$chd), function(class_label) {
  class_indices <- which(data$chd == class_label)
  sample(class_indices, size = round(length(class_indices) * test_fraction))
}), use.names = FALSE))
training_indices <- setdiff(seq_len(nrow(data)), test_indices)
training <- data[training_indices, , drop = FALSE]
testing <- data[test_indices, , drop = FALSE]

fold_id <- integer(nrow(training))
for (class_label in levels(training$chd)) {
  class_indices <- which(training$chd == class_label)
  fold_id[class_indices] <- sample(rep(seq_len(fold_count), length.out = length(class_indices)))
}

fold_feasibility <- do.call(rbind, lapply(seq_len(fold_count), function(fold) {
  fold_training <- training[fold_id != fold, , drop = FALSE]
  class_counts <- table(factor(fold_training$chd, levels = c("0", "1")))
  ranks <- vapply(levels(fold_training$chd), function(class_label) {
    class_data <- fold_training[fold_training$chd == class_label, predictors, drop = FALSE]
    qr(stats::cov(class_data))$rank
  }, integer(1))
  data.frame(
    fold = fold, training_n = nrow(fold_training),
    class_0_n = unname(class_counts["0"]), class_1_n = unname(class_counts["1"]),
    class_0_covariance_rank = unname(ranks["0"]), class_1_covariance_rank = unname(ranks["1"])
  )
}))
write.csv(fold_feasibility, file.path(output_dir, "fold-feasibility.csv"), row.names = FALSE)
stopifnot(
  all(fold_feasibility$class_0_n > length(predictors)),
  all(fold_feasibility$class_1_n > length(predictors)),
  all(fold_feasibility$class_0_covariance_rank == length(predictors)),
  all(fold_feasibility$class_1_covariance_rank == length(predictors))
)

fold_predictions <- list()
fold_metrics <- list()
warnings_log <- character()
for (candidate_index in seq_along(candidate_names)) {
  method <- c("LDA", "QDA")[candidate_index]
  for (fold in seq_len(fold_count)) {
    fold_training <- training[fold_id != fold, , drop = FALSE]
    fold_validation <- training[fold_id == fold, , drop = FALSE]
    fit_result <- capture_warnings(fit_predict(method, fold_training, fold_validation))
    if (length(fit_result$warnings)) {
      warnings_log <- c(warnings_log, paste(candidate_names[candidate_index], "fold", fold,
                                            paste(fit_result$warnings, collapse = " | ")))
    }
    predicted <- fit_result$value
    matrix <- confusion_matrix(fold_validation$chd, predicted)
    fold_metrics[[length(fold_metrics) + 1L]] <- data.frame(
      candidate = candidate_names[candidate_index], fold = fold, n = nrow(fold_validation),
      correct = sum(fold_validation$chd == predicted),
      accuracy = mean(fold_validation$chd == predicted),
      error = mean(fold_validation$chd != predicted),
      TN = unname(matrix["0", "0"]), FP = unname(matrix["0", "1"]),
      FN = unname(matrix["1", "0"]), TP = unname(matrix["1", "1"])
    )
    fold_predictions[[length(fold_predictions) + 1L]] <- data.frame(
      candidate = candidate_names[candidate_index],
      original_row = training_indices[which(fold_id == fold)], fold = fold,
      actual = as.character(fold_validation$chd), predicted = as.character(predicted)
    )
  }
}
fold_metrics <- do.call(rbind, fold_metrics)
fold_predictions <- do.call(rbind, fold_predictions)
write.csv(fold_metrics, file.path(output_dir, "cv-fold-metrics.csv"), row.names = FALSE)
write.csv(fold_predictions, file.path(output_dir, "cv-fold-predictions.csv"), row.names = FALSE)
writeLines(if (length(warnings_log)) warnings_log else "No model warnings.",
           file.path(output_dir, "model-warnings.txt"))

summarize_candidate <- function(candidate_label) {
  candidate_folds <- fold_metrics[fold_metrics$candidate == candidate_label, , drop = FALSE]
  data.frame(
    candidate = candidate_label,
    mean_fold_accuracy = mean(candidate_folds$accuracy),
    mean_fold_error = mean(candidate_folds$error),
    TN_mean = mean(candidate_folds$TN), TN_SE = sd(candidate_folds$TN) / sqrt(fold_count),
    FP_mean = mean(candidate_folds$FP), FP_SE = sd(candidate_folds$FP) / sqrt(fold_count),
    FN_mean = mean(candidate_folds$FN), FN_SE = sd(candidate_folds$FN) / sqrt(fold_count),
    TP_mean = mean(candidate_folds$TP), TP_SE = sd(candidate_folds$TP) / sqrt(fold_count)
  )
}
candidate_summary <- do.call(rbind, lapply(candidate_names, summarize_candidate))
write.csv(candidate_summary, file.path(output_dir, "cv-summary.csv"), row.names = FALSE)

minimum_error <- min(candidate_summary$mean_fold_error)
selected <- candidate_summary$candidate[abs(candidate_summary$mean_fold_error - minimum_error) < 1e-12]
if (length(selected) != 1L) stop("CV tie: report the tie rather than selecting by test results.")
selected_method <- if (startsWith(selected, "LDA")) "LDA" else "QDA"
test_result <- capture_warnings(fit_predict(selected_method, training, testing))
test_prediction <- test_result$value
test_matrix <- confusion_matrix(testing$chd, test_prediction)
test_metrics <- data.frame(
  selected_model = selected, n_test = nrow(testing),
  correct = sum(testing$chd == test_prediction), incorrect = sum(testing$chd != test_prediction),
  accuracy = mean(testing$chd == test_prediction), error = mean(testing$chd != test_prediction),
  TN = unname(test_matrix["0", "0"]), FP = unname(test_matrix["0", "1"]),
  FN = unname(test_matrix["1", "0"]), TP = unname(test_matrix["1", "1"])
)
test_metrics$type_I <- if (test_metrics$TN + test_metrics$FP == 0) NA_real_ else test_metrics$FP / (test_metrics$TN + test_metrics$FP)
test_metrics$type_II <- if (test_metrics$FN + test_metrics$TP == 0) NA_real_ else test_metrics$FN / (test_metrics$FN + test_metrics$TP)
write.csv(test_metrics, file.path(output_dir, "test-metrics.csv"), row.names = FALSE)
write.csv(data.frame(original_row = test_indices, actual = as.character(testing$chd),
                     predicted = as.character(test_prediction)),
          file.path(output_dir, "test-predictions.csv"), row.names = FALSE)
write.csv(as.data.frame.matrix(test_matrix), file.path(output_dir, "test-confusion-matrix.csv"))
split_index <- data.frame(original_row = seq_len(nrow(data)), split = "training", fold = NA_integer_)
split_index$split[test_indices] <- "test"
split_index$fold[training_indices] <- fold_id
write.csv(split_index, file.path(output_dir, "split-index.csv"), row.names = FALSE)
writeLines(if (length(test_result$warnings)) test_result$warnings else "No final-fit warnings.",
           file.path(output_dir, "test-model-warnings.txt"))

descriptives <- do.call(rbind, lapply(names(data), function(variable) {
  values <- data[[variable]]
  if (is.numeric(values)) {
    quartiles <- quantile(values, c(0.25, 0.5, 0.75))
    iqr <- IQR(values)
    data.frame(variable = variable, type = class(values)[1], n = length(values),
      missing = sum(is.na(values)), minimum = min(values), q1 = quartiles[1],
      median = quartiles[2], mean = mean(values), q3 = quartiles[3], maximum = max(values),
      iqr_outlier_count = sum(values < quartiles[1] - 1.5 * iqr | values > quartiles[3] + 1.5 * iqr),
      class_0 = NA_integer_, class_1 = NA_integer_)
  } else {
    data.frame(variable = variable, type = "factor", n = length(values),
      missing = sum(is.na(values)), minimum = NA, q1 = NA, median = NA, mean = NA,
      q3 = NA, maximum = NA, iqr_outlier_count = NA_integer_,
      class_0 = sum(values == "0"), class_1 = sum(values == "1"))
  }
}))
write.csv(descriptives, file.path(output_dir, "data-descriptives.csv"), row.names = FALSE)
write.csv(data.frame(variable = names(missing_counts), missing = unname(missing_counts)),
          file.path(output_dir, "missingness.csv"), row.names = FALSE)
writeLines(c(
  paste("rows:", nrow(data)), paste("predictors:", paste(predictors, collapse = ", ")),
  paste("duplicate rows:", sum(duplicated(read.csv(data_path, stringsAsFactors = FALSE)))),
  paste("class counts:", paste(names(table(data$chd)), as.integer(table(data$chd)), collapse = "; ")),
  paste("training rows:", nrow(training)), paste("test rows:", nrow(testing)),
  paste("fold count:", fold_count), paste("seed:", seed), paste("selected:", selected)
), file.path(output_dir, "run-summary.txt"))

cat("Selected:", selected, "\n")
print(candidate_summary, row.names = FALSE)
print(test_metrics, row.names = FALSE)