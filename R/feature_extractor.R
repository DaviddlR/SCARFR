


#' Title
#'
#' @param dataframe Dataframe to process and extract features
#' @param pretrained_model_path Path to the pretrained model used to extract representations
#' @param exclude_columns Columns that the pretraining model should avoid (i.e target or ID columns). Default = NULL
#' @param batch_size Batch size used during feature extraction. Default = 32
#'
#' @returns TODO
#' @export
#'
#' @examples
#' a <- 1
scarf_feature_extractor = function(dataframe, pretrained_model_path, exclude_columns = NULL, batch_size = 32) {

  # Load pretrained model and recipe
  model_bundle <- torch::torch_load(pretrained_model_path)

  # Validate input
  if(is.list(model_bundle) && identical(model_bundle$bundle_type, "scarf_bundle")) {

    # Load encoder hyperparameters
    hparams <- model_bundle$encoder_hparams

    print(hparams)

    # Create a new encoder
    fitted_encoder <- scarf_encoder(
      in_dim = hparams$in_dim,
      hidden_dim = hparams$hidden_dim,
      num_hidden = hparams$num_hidden,
      dropout = hparams$dropout
    )

    # Load trained weights
    fitted_encoder$load_state_dict(model_bundle$encoder_state_dict)

    # Load recipe
    trained_recipe <- unserialize(model_bundle$recipe)

  } else {
    stop("The input is not a scarf_bundle. Please, train the model using scarf_fit() and set the pretrained_model_path to the path in which the trained model is stored")
  }

  # Prepare data
  dataframe_cleaned <- prepare_scarf_data_for_feature_extraction(dataframe, trained_recipe, exclude_columns)

  dataset_ready <- create_tensor_dataset(dataframe_cleaned)

  dataloader_ready <- torch::dataloader(dataset_ready,
                         batch_size = batch_size,
                         shuffle = FALSE)

  # Prepare model
  fitted_encoder$eval()

  # Extract latent features using the trained model
  features_list <- list()

  torch::with_no_grad({
    coro::loop(
      for(batch in dataloader_ready) {

        # Take batch
        x_batch <- batch[[1]]

        # Forward pass
        batch_latent <- fitted_encoder(x_batch)

        # Store representations
        features_list[[length(features_list) + 1]] <- batch_latent$cpu()

      }
    )
  })

  # Concatenate representations
  all_features <- torch::torch_cat(features_list, dim=1)

  # Convert to matrix
  all_features <- as.matrix(all_features)

  print(dim(all_features))

  return(all_features)


}







