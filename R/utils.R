

#' Function to read and check a SCARF bundle
#'
#' @param bundle_path Path to file storing a "scarf_bundle" object
#'
#' @returns Pretrained model's weights and trained recipe for preprocessing
#'
load_scarf_bundle = function(bundle_path) {

  # Load pretrained model and recipe
  model_bundle <- torch::torch_load(paste(bundle_path, ".pt", sep=""))

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

    return(list(
      encoder = fitted_encoder,
      recipe = trained_recipe
    ))

  } else {
    stop("The input is not a scarf_bundle. Please, train the model using scarf_fit() and set the pretrained_model_path to the path in which the trained model is stored")
  }
}




load_classifier_bundle = function(bundle_path) {

  # Load pretrained model and recipe
  model_bundle <- torch::torch_load(bundle_path)

  # Validate input
  if(is.list(model_bundle) && identical(model_bundle$bundle_type, "classifier_bundle")) {

    # Load encoder hyperparameters
    classifier_weights <- model_bundle$classifier_state_dict
    classifier_hparams <- model_bundle$classifier_hparams

    classifier_net <- classifier_network(
      input_dim = classifier_hparams$in_dim,
      n_classes = classifier_hparams$n_classes,
      dropout = classifier_hparams$dropout
    )

    # Load weights
    classifier_net$load_state_dict(classifier_weights)


    # Load levels
    trained_levels <- unserialize(model_bundle$levels)

    return(list(
      classifier = classifier_net,
      levels = trained_levels
    ))

  } else {
    stop("The input is not a scarf_bundle. Please, train the model using scarf_fit() and set the pretrained_model_path to the path in which the trained model is stored")
  }

}




