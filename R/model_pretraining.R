



#' SCARF pretraining
#'
#' @param dataframe_train Train dataframe
#' @param exclude_columns Columns that the pretraining model should avoid (i.e target or ID columns).
#' @param create_validation Indicate whether a validation set should be created.
#' @param validation_proportion Proportion of the training samples that will be used to create the validation set, if required.
#' @param batch_size Batch size used during pretraining.
#' @param n_epochs Number of pretraining epochs.
#' @param save_path Path where the pretrained model will be saved.
#'
#' @returns A pretrained .PT model.
#' @export
#'
#' @examples
#' a <- 1
scarf_fit = function(dataframe_train, exclude_columns = NULL, create_validation = FALSE, validation_proportion = 0.1, batch_size = 256, n_epochs = 1, save_path = "trained_luz.pt") {

  # Load and preprocess data
  preprocessed_datasets <- prepare_scarf_data(dataframe_train, exclude_columns = exclude_columns, create_validation = create_validation, validation_proportion = validation_proportion)

  x_train <- preprocessed_datasets$train_set
  x_val <- preprocessed_datasets$val_set
  recipe <- preprocessed_datasets$recipe

  # Create training dataset and dataloader
  train_ds <- create_tensor_dataset(x_train)

  train_dl <- torch::dataloader(train_ds,
                         batch_size = batch_size,
                         shuffle = TRUE)

  # Create training dataset and dataloader (if required)
  val_dl <- NULL

  if(create_validation) {
    val_ds <- create_tensor_dataset(x_val)

    val_dl <- torch::dataloader(val_ds,
                         batch_size = batch_size,
                         shuffle=FALSE)
  }

  fitted <- SCARF_wrapper |>
    luz::setup(
      loss = nt_xent_loss(temperature = 0.5),
      optimizer = torch::optim_adam
    ) |>
    luz::set_hparams(
      in_dim = dim(x_train)[2],
      hidden_dim = 256,
      num_hidden = 4,
      head_hidden_dim = 256,
      head_num_hidden = 2,
      dropout = 0.0,
    ) |>
    luz::set_opt_hparams(
      lr = 0.0001,
    ) |>
    luz::fit(
      train_dl,
      epochs = n_epochs,
      valid_data = val_dl,
      callbacks = list(custom_scarf_step_callback(corruption_rate = 0.6))
    )


  # Save trained model AND the recipe required to apply the same preprocessing to the test set

  encoder_weights <- fitted$model$main_encoder$state_dict()

  hparams <- list(
    in_dim = dim(x_train)[2],
    hidden_dim = 256,
    num_hidden = 4,
    dropout = 0.0
  )

  model_bundle <- list(
    encoder_state_dict = encoder_weights,
    encoder_hparams = hparams,
    recipe = serialize(recipe, NULL),
    bundle_type = "scarf_bundle"
  )

  torch::torch_save(model_bundle, path = save_path)







  #luz::luz_save(fitted, "scarf_trained.rds")




}




custom_scarf_step_callback <- luz::luz_callback(

  name = "SCARF_custom_steps",

  initialize = function(corruption_rate = 0.6) {
    self$corruption_rate = corruption_rate
  },



  # Train. It receives a batch from the dataloader / tensor_dataset
  on_train_batch_begin = function() {

    #print(ctx$batch[[1]]$device)

    batch <- ctx$batch
    #target <- batch$y  # Label. Not used during pre-training

    x <- batch$x  # Data

    batch_size <- x$size(1)
    num_features <- x$size(2)

    mask <- torch::torch_rand_like(x) < self$corruption_rate

    random_indices <- torch::torch_randint(
      low = 1,
      high = batch_size + 1,  # 1 and +1 because R indices start at 1
      size = c(batch_size, num_features),
      device = x$device,
      dtype = torch::torch_long()
    )

    x_random <- torch::torch_gather(x, dim=1, index = random_indices)
    x_corrupted <- torch::torch_where(mask, x_random, x)

    #ctx$batch[[2]] <- batch$y
    ctx$batch[[1]] <- c(x, x_corrupted)



  },


  # Validation
  on_valid_batch_begin = function() {
    batch = ctx$batch
    #target = batch$y  # Label. Not used during pre-training

    x = batch$x  # Data

    batch_size = x$size(1)
    num_features = x$size(2)

    mask = torch::torch_rand_like(x) < self$corruption_rate

    random_indices <- torch::torch_randint(
      low = 1,
      high = batch_size + 1,  # 1 and +1 because R indices start at 1
      size = c(batch_size, num_features),
      device = x$device,
      dtype = torch::torch_long()
    )

    x_random <- torch::torch_gather(x, dim=1, index = random_indices)
    x_corrupted <- torch::torch_where(mask, x_random, x)

    #ctx$target <- batch$y
    ctx$input <- c(x, x_corrupted)
  },


  # Test / predict

)


# preprocessed_datasets = read_parquet_data("inst/extdata/UNSW_NB15_training-set.parquet", "inst/extdata/UNSW_NB15_testing-set.parquet")
#
# x_train <- preprocessed_datasets$train_set
# x_val <- preprocessed_datasets$val_set
# x_test <- preprocessed_datasets$test_set
#
# ejemplo_x <- x_train[1, , drop = FALSE]
#
#
# y_train <- preprocessed_datasets$train_label
# y_val <- preprocessed_datasets$val_label
# y_test <- preprocessed_datasets$test_label
#
# print(dim(x_train))
# print(dim(x_val))
# print(dim(x_test))
# print(dim(y_train))
# print(dim(y_val))
# print(dim(y_test))
#
# # Create tensor datasets
# train_ds <- create_tensor_dataset(x_train, y_train)
# val_ds <- create_tensor_dataset(x_val, y_val)
# test_ds <- create_tensor_dataset(x_test, y_test)
#
# print(train_ds[1]$x)
#
# # Create dataloader
# train_dl <- dataloader(train_ds,
#                        batch_size = 32,
#                        shuffle = TRUE)
#
# val_dl <- dataloader(val_ds,
#                      batch_size = 32,
#                      shuffle=FALSE)
#
# test_dl <- dataloader(test_ds,
#                       batch_size = 32,
#                       shuffle = FALSE)
#
#
#
# scarf_pretraining("inst/extdata/UNSW_NB15_training-set.parquet", "inst/extdata/UNSW_NB15_testing-set.parquet")











