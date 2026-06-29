
# Entrenar un clasificador sobre las representaciones latentes
train_classifier_on_representations = function(df_train, pretrained_model_path, label_column, num_classes, exclude_columns = NULL, classification_model_type = "MLP", dropout = 0.2, save_path = "classifier") {

  # Extract latent features of train set
  extracted_features <- scarf_feature_extractor(df_train,
                                                pretrained_model_path,
                                                exclude_columns = exclude_columns,
                                                want_labels = TRUE,
                                                label_column = label_column)

  features <- extracted_features$features
  y_train <- extracted_features$features_labels


  # Label encoder
  y_train_factor <- as.factor(y_train)
  train_levels <- levels(y_train_factor)  # Have to save train_levels
  y_train_encoded <- as.integer(y_train_factor)
  print(train_levels)

  # y_val_encoded <- as.integer(factor(y_val, levels = train_levels))

  # Set X and Y as tensors
  features_tensor <- torch::torch_tensor(features, dtype = torch::torch_float())
  y_train_tensor <- torch::torch_tensor(y_train_encoded, dtype = torch::torch_long())

  # Create dataset and dataloader and validation if required
  train_ds <- torch::tensor_dataset(features_tensor, y_train_tensor)

  train_dl <- torch::dataloader(
    train_ds,
    batch_size = 256,
    shuffle = TRUE,
  )

  val_dl <- NULL  # TODO

  # Create and train classification head
  if(!identical(classification_model_type, "MLP")){
    stop("train_classifier_on_representations: the classification_model_type is not known. Please use one of these options: MLP")
  }

  fitted_classification_head <- classifier_network |>
    luz::setup(
      loss = torch::nn_cross_entropy_loss(),
      optimizer = torch::optim_adam,
    ) |>
    luz::set_hparams(
      input_dim = dim(features)[[2]],
      n_classes = num_classes,
      dropout = dropout,
    ) |>
    luz::set_opt_hparams(
      lr = 0.0001,
    ) |>
    luz::fit(
      train_dl, # TODO
      epochs = 1,
      valid_data = val_dl,
    )

  print(fitted_classification_head)


  # Save classification model and train levels.


  classifier_weights <- fitted_classification_head$model$classifier$state_dict()

  hparams <- list(
    in_dim = dim(features)[[2]],
    n_classes = num_classes,
    dropout = dropout
  )

  model_bundle <- list(
    classifier_state_dict = classifier_weights,
    classifier_hparams = hparams,
    levels = serialize(train_levels, NULL),
    bundle_type = "classifier_bundle"
  )

  torch::torch_save(model_bundle, path = paste(save_path, ".pt", sep=""))






}




# Una vez entrenado el clasificador, evaluarlo sobre un conjunto de test concreto
downstream_prediction = function(df_test, target_column, exclude_columns, pretrained_model_path, classification_model_path, return_classification_report = FALSE) {

  # Load model
  fitted_model <- load_classifier_bundle(paste(classification_model_path, ".pt", sep=""))
  print("1")

  print(fitted_model$classifier)
  print(fitted_model$levels)





  # Load test data



  # Extract latent features of test set

  # Predict on the test set

  # Evaluate if required

}














