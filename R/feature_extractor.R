


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
scarf_feature_extractor = function(dataframe, pretrained_model_path, exclude_columns = NULL, want_labels = FALSE, label_column = NULL, batch_size = 32) {

  # Extract pretrained model and recipe
  bundle <- load_scarf_bundle(pretrained_model_path)

  fitted_encoder <- bundle$encoder
  trained_recipe <- bundle$recipe

  # Prepare data
  dataframe_cleaned_xy <- prepare_scarf_data_for_feature_extraction(dataframe, trained_recipe, exclude_columns, want_labels = want_labels, label_column = label_column)
  dataframe_cleaned <- dataframe_cleaned_xy$x
  dataframe_labels <- dataframe_cleaned_xy$y

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

  print("Extracted features: ")
  print(dim(all_features))
  print(length(dataframe_labels))

  return(list(
    features = all_features,
    features_labels = dataframe_labels
    ))


}







