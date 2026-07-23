




############ SCARF PRETRAINING ############

# Train a neural network with the SCARF pretraining objective
scarf_fit = function(
  dataframe_train,
  exclude_columns = NULL,
  create_validation = FALSE,
  validation_proportion = 0.1,
  batch_size = 256,
  n_epochs = 1,
  preprocess = TRUE
) {


  # Load and preprocess data
  preprocessed_datasets <- prepare_scarf_data(dataframe_train, exclude_columns = exclude_columns, create_validation = create_validation, validation_proportion = validation_proportion, preprocess = preprocess)

  x_train <- preprocessed_datasets$train_set
  x_val <- preprocessed_datasets$val_set  # May be null
  recipe <- preprocessed_datasets$recipe  # May be null
  feature_groups <- preprocessed_datasets$feature_groups


  #print(feature_groups)

  #stop("paramo ninio")

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
      dropout = 0.0
    ) |>
    luz::set_opt_hparams(
      lr = 0.0001
    ) |>
    luz::fit(
      train_dl,
      epochs = n_epochs,
      valid_data = val_dl,
      callbacks = list(custom_scarf_step_callback(feature_groups, corruption_rate = 0.6))
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

  return(invisible(model_bundle))


  # If save_path is not null, save model locally (it will be NULL when using it as a recipe, when stored in RAM)
  # if (!is.null(save_path)){
  #   torch::torch_save(model_bundle, path = paste0(save_path, ".pt"))
  #   message("Pretrained model saved in ", save_path, ".pt")
  # }
  #
  # # Return invisible for the recipe prep and bake
  # return(invisible(model_bundle))


}



custom_scarf_step_callback <- luz::luz_callback(

  name = "SCARF_custom_steps",

  initialize = function(feature_groups, corruption_rate = 0.6) {
    self$corruption_rate <- corruption_rate
    self$feature_groups <- feature_groups
    self$num_groups <- length(feature_groups)
  },



  corrupt_batch = function(x) {
    batch_size <- x$size(1)
    num_features <- x$size(2)

    # Mask on the original features
    mask_original_features <- torch::torch_rand(
      c(batch_size, self$num_groups),
      device = x$device
    ) < self$corruption_rate

    rand_idx_original <- torch::torch_randint(
      low = 1,
      high = batch_size + 1,
      size = c(batch_size, self$num_groups),
      device = x$device,
      dtype = torch::torch_long()
    )

    # Project mask to the one-hotted-features
    mask_full <- torch::torch_empty(
      c(batch_size, num_features),
      device = x$device,
      dtype = torch::torch_bool()
    )

    rand_idx_full <- torch::torch_empty(
      c(batch_size, num_features),
      device = x$device,
      dtype = torch::torch_long()
    )


    for (g in seq_len(self$num_groups)) {
      cols <- self$feature_groups[[g]]

      for (col in cols) {
        mask_full[, col] <- mask_original_features[, g]
        rand_idx_full[, col] <- rand_idx_original[, g]
      }
    }


    # Switch features between instances
    x_random <- torch::torch_gather(x, dim = 1, index = rand_idx_full)
    x_corrupted <- torch::torch_where(mask_full, x_random, x)

    return(list(x, x_corrupted))

  },


  # Train. It receives a batch from the dataloader / tensor_dataset
  on_train_batch_begin = function() {
    ctx$batch[[1]] <- self$corrupt_batch(x = ctx$batch[[1]])

    #
    #
    # #print(ctx$batch[[1]]$device)
    #
    # batch <- ctx$batch
    # #target <- batch$y  # Label. Not used during pre-training
    #
    # x <- batch$x  # Data
    #
    # batch_size <- x$size(1)
    # num_features <- x$size(2)
    #
    # mask <- torch::torch_rand_like(x) < self$corruption_rate
    #
    # random_indices <- torch::torch_randint(
    #   low = 1,
    #   high = batch_size + 1,  # 1 and +1 because R indices start at 1
    #   size = c(batch_size, num_features),
    #   device = x$device,
    #   dtype = torch::torch_long()
    # )
    #
    # x_random <- torch::torch_gather(x, dim=1, index = random_indices)
    # x_corrupted <- torch::torch_where(mask, x_random, x)
    #
    # #ctx$batch[[2]] <- batch$y
    # ctx$batch[[1]] <- c(x, x_corrupted)
    #
    #

  },


  # Validation
  on_valid_batch_begin = function() {
    ctx$batch[[1]] <- self$corrupt_batch(x = ctx$batch[[1]])
  }

)











############ VIME PRETRAINING ############








