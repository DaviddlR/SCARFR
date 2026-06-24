


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
  if(inherits(model_bundle, "scarf_bundle")){
    fitted_encoder <- model_bundle$fitted_model$main_encoder  # SCARF_wrapper -> SCARF_encoder
    trained_recipe <- model_bundle$recipe
  } else {
    stop("The input is not a scarf_bundle. Please, train the model using scarf_fit() and set the pretrained_model_path to the path in which the trained model is stored")
  }

  # Prepare data
  dataframe <- prepare_scarf_data_for_feature_extraction(dataframe, trained_recipe, exclude_columns)

  dataset_ready <- create_tensor_dataset(dataframe)

  dataloader_ready <- dataloader(dataset_ready,
                         batch_size = batch_size,
                         shuffle = FALSE)

  # Prepare model
  fitted_encoder$eval()

  # Extract latent features using the trained model
  torch::with_no_grad(
    coro
  )


}







