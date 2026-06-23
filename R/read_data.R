



#' Title
#'
#' @param dataframe_train Train dataframe
#' @param exclude_columns Columns that the pretraining model should avoid (i.e target columns)
#' @param create_validation Indicate whether a validation set should be created
#' @param validation_proportion Proportion of the training samples that will be used to create the validation set, if required.
#'
#' @returns Preprocessed train dataset (and validation set if required) and the recipe used for preprocessing
#' @export
#'
#' @examples
#' data(iris)
#'
#' data_ready <- prepare_scarf_data(dataframe_train = iris, exclude_columns = "Species", create_validation = TRUE)
#'
#' dim(data_ready$train_set)
prepare_scarf_data = function(dataframe_train, exclude_columns = NULL, create_validation = FALSE, validation_proportion = 0.1) {

  df_train_data <- as.data.frame(dataframe_train)

  # Remove unneeded columns and create "y" as the target column
  x_train_orig <- df_train_data[, !(names(df_train_data) %in% exclude_columns), drop=FALSE]

  # Validation set
  if(create_validation){
    n_samples <- nrow(x_train_orig)
    validation_size <- floor(validation_proportion * n_samples)
    val_indices <- sample(seq_len(n_samples), size=validation_size)

    x_val <- x_train_orig[val_indices, , drop=FALSE]
    x_train <- x_train_orig[-val_indices, , drop=FALSE]

  } else {
    x_train <- x_train_orig
    x_val <- NULL
  }

  # One hot encoding + standard scaler
  rec <- recipes::recipe(~ ., data=x_train)

  rec <- recipes::step_novel(rec, recipes::all_nominal_predictors(), new_level = "unknown") |>  # New categorical levels (should not be used)
    recipes::step_normalize(recipes::all_numeric_predictors()) |>  # Standard normalization
    recipes::step_dummy(recipes::all_nominal_predictors(), one_hot = TRUE)  # One-hot encoding


  # Fit recipe to training set
  trained_recipe <- recipes::prep(rec, training = x_train)

  # Apply preprocessing to train and create matrix
  x_train_processed <- recipes::bake(trained_recipe, new_data = x_train)
  x_train_mat <- as.matrix(x_train_processed)



  # Bake validation set (if exists)
  x_val_mat <- NULL


  if(create_validation){
    x_val_processed <- recipes::bake(trained_recipe, new_data = x_val)
    x_val_mat <- as.matrix(x_val_processed)
  }

  print("Train set: ")
  print(dim(x_train_mat))
  if(create_validation){
    print("Validation set: ")
    print(dim(x_val_mat))
  }


  return (list("train_set" = x_train_mat,
               "val_set" = x_val_mat,
               "recipe" = trained_recipe))

}





#' Read parquet data and prepare for training
#'
#' @param filename_train Path to the training file
#' @param filename_test Path to the testing file
#'
#' @returns An object type 'list' with training, validation and testing (data and labels) subsets
#' @export
#'
#' @examples
#' train_path <- system.file("extdata", "UNSW_NB15_training-set.parquet", package = "scaRf")
#' test_path <- system.file("extdata", "UNSW_NB15_testing-set.parquet", package = "scaRf")
#' if (train_path != "" && test_path != "") {
#'   datasets <- read_parquet_data(train_path, test_path)
#'
#'   train_ds <- datasets$train_set
#'   train_label <- datasets$train_label
#'   validation_ds <- datasets$train_set
#'   validation_label <- datasets$validation_label
#'   test_ds <- datasets$train_set
#'   test_label <- datasets$test_label
#' }
read_parquet_data = function(filename_train, filename_test){

  # TODO: modificar esto para que le llegue un dataframe, columna target y columnas a excluir

  df_train <- arrow::read_parquet(filename_train)
  df_test <- arrow::read_parquet(filename_test)

  df_train <- as.data.frame(df_train)
  df_test <- as.data.frame(df_test)

  # Separate target columns
  target_cols <- c("label", "attack_cat")

  x_train_orig <- df_train[, !(names(df_train) %in% target_cols), drop=FALSE]
  y_train_orig <- df_train$attack_cat

  x_test_orig <- df_test[, !(names(df_test) %in% target_cols), drop=FALSE]
  y_test_orig <- df_test$attack_cat

  # Validation set (10%)
  n_samples <- nrow(x_train_orig)
  val_size <- floor(0.1 * n_samples)
  val_indices <- sample(seq_len(n_samples), size=val_size)

  x_val <- x_train_orig[val_indices, , drop=FALSE]
  y_val <- y_train_orig[val_indices]

  x_train <- x_train_orig[-val_indices, , drop=FALSE]
  y_train <- y_train_orig[-val_indices]

  # Label encoder
  y_train_factor <- as.factor(y_train)
  train_levels <- levels(y_train_factor)

  y_train_encoded <- as.integer(y_train_factor)
  y_val_encoded <- as.integer(factor(y_val, levels = train_levels))
  y_test_encoded <- as.integer(factor(y_test_orig, levels = train_levels))

  # One hot encoding + standard scaler
  rec <- recipes::recipe(~ ., data=x_train)

  rec <- recipes::step_novel(rec, recipes::all_nominal_predictors(), new_level = "unknown") |> # New categorical levels (should not be used)
    recipes::step_normalize(recipes::all_numeric_predictors()) |> # Standard normalization
    recipes::step_dummy(recipes::all_nominal_predictors(), one_hot = TRUE)  # One-hot encoding


  # Fit recipe to training set
  trained_recipe <- recipes::prep(rec, training = x_train)

  # Apply preprocessing to train, validation and test sets
  x_train_processed <- recipes::bake(trained_recipe, new_data = x_train)
  x_val_processed <- recipes::bake(trained_recipe, new_data = x_val)
  x_test_processed <- recipes::bake(trained_recipe, new_data = x_test_orig)

  # Convert to matrices
  x_train_mat <- as.matrix(x_train_processed)
  x_val_mat <- as.matrix(x_val_processed)
  x_test_mat <- as.matrix(x_test_processed)

  # print("Train set: ")
  # print(dim(x_train_mat))
  # print("Validation set: ")
  # print(dim(x_val_mat))
  # print("Test set:")
  # print(dim(x_test_mat))


  return (list("train_set" = x_train_mat,
               "train_label" = y_train_encoded,
               "val_set" = x_val_mat,
               "val_label" = y_val_encoded,
               "test_set" = x_test_mat,
               "test_label" = y_test_encoded))

}


# train_path <- "inst/extdata/UNSW_NB15_training-set.parquet"
# test_path <- "inst/extdata/UNSW_NB15_testing-set.parquet"
# datasets <- read_parquet_data(train_path, test_path)
# train_ds <- datasets$train_set
#
# ejemplo_x <- datasets$train_set[1, , drop = FALSE]
# ejemplo_y <- datasets$train_label[54000]
# ejemplo_y


