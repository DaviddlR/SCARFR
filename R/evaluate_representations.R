

# TODO: si se incluyen más clasificadores, cómo guardarlos?

#' Train a classifier on top of the latent representations created by a pretrained model.
#'
#' @param df_train Train dataframe in which the classification model will be trained.
#' @param pretrained_model_path Path to the pretrained model.
#' @param label_column Column of the dataframe that store the sample's label.
#' @param num_classes Number of possible classes.
#' @param exclude_columns Columns that the classification model should ignore during training and inference (i.e target or ID columns). Default: NULL.
#' @param classification_model_type Type of classifier to be created and trained. Options: "MLP". Default: "MLP".
#' @param dropout If classification_model_type = "MLP", this parameter sets the dropout probability. Default: 0.2.
#' @param doitsmall Specify if the training set should be reduced for experimental purposes (i.e. what happen if we only use 1% of training samples?). Default: FALSE.
#' @param save_path Path where the classification model will be saved.
#'
#' @returns A pretrained .PT classifier model
#' @export
#'
#' @examples
#' a <- 1
train_classifier_on_representations = function(df_train, pretrained_model_path, label_column, num_classes, exclude_columns = NULL, classification_model_type = "MLP", dropout = 0.2, doitsmall = FALSE, save_path = "classifier") {

  if(doitsmall) {

    print("Doing it small...")
    label_proportion <- 0.01

    df_train <- df_train |>
      dplyr::group_by(attack_cat) |>
      dplyr::sample_frac(label_proportion) |>
      dplyr::ungroup()
  }

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
      train_dl,
      epochs = 50,
      valid_data = val_dl,
    )

  print(fitted_classification_head)


  # Save classification model and train levels.


  classifier_weights <- fitted_classification_head$model$state_dict()

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
#' Title
#'
#' @param df_test Test dataframe in which the classification model will predict each sample.
#' @param pretrained_model_path Path to the pretrained model.
#' @param target_column Column of the dataframe that store the sample's label.
#' @param classification_model_path Path to the classification model.
#' @param exclude_columns Columns that the classification model should ignore during training and inference (i.e target or ID columns). Default: NULL.
#' @param return_classification_report Indicate whether the user want to produce a classification report (TRUE) or just the predictions (FALSE). Default: FALSE.
#'
#' @returns A "list" with the predicted label and probability score of each sample.
#' @export
#'
#' @examples
#' a <- 1
downstream_prediction = function(df_test, pretrained_model_path, label_column, classification_model_path, exclude_columns = NULL, return_classification_report = FALSE) {

  # Load model
  fitted_classifier_bundle <- load_classifier_bundle(paste(classification_model_path, ".pt", sep=""))

  fitted_classifier <- fitted_classifier_bundle$classifier

  # Extract latent features of test set
  extracted_features_test <- scarf_feature_extractor(df_test,
                                                     pretrained_model_path = pretrained_model_path,
                                                     exclude_columns = exclude_columns,
                                                     want_labels = TRUE,
                                                     label_column = label_column,
                                                     batch_size = 32)

  features_test <- extracted_features_test$features
  labels_test <- extracted_features_test$features_labels

  # Label encoder
  y_test_encoded <- as.integer(factor(labels_test, levels = fitted_classifier_bundle$levels))

  # Set X and Y as tensors
  features_test_tensor <- torch::torch_tensor(features_test, dtype = torch::torch_float())
  y_test_tensor <- torch::torch_tensor(y_test_encoded, dtype = torch::torch_long())

  print(y_test_tensor[2])

  # Create dataset and dataloader
  test_ds <- torch::tensor_dataset(features_test_tensor, y_test_tensor)

  test_dl <- torch::dataloader(
    test_ds,
    batch_size = 256,
    shuffle = FALSE
  )

  # Predict on the test set

  # Prepare model
  device <- if(torch::cuda_is_available()) torch::torch_device("cuda") else torch::torch_device("cpu")
  message("Ejecutando inferencia en: ", if (torch::cuda_is_available()) "GPU (CUDA)" else "CPU")

  fitted_classifier$to(device = device)
  fitted_classifier$eval()

  # Loop on the dataloader
  predictions <- list()

  torch::with_no_grad({
    coro::loop(
      for(batch in test_dl) {

        # Take batch
        x_batch <- batch[[1]]$to(device = device)

        # Forward pass
        batch_prediction <- fitted_classifier(x_batch)

        # Store predictions
        predictions[[length(predictions) + 1]] <- batch_prediction$cpu()
      }
    )
  })

  # Concatenate batches
  predictions <- torch::torch_cat(predictions, dim=1)

  print("Raw predictions: ")
  print(dim(predictions))
  print(predictions[2])



  # Get probabilities
  sm <- torch::nn_softmax(dim = 2)
  probabilities <- sm(predictions)
  probabilities <- as.array(probabilities)

  print("Probabilities: ")
  print(dim(probabilities))
  print(probabilities[2, ])

  # Get predicted class index
  pred_indices <- as.integer(torch::torch_argmax(probabilities, dim=2))
  print(pred_indices[2])

  # Get predicted class name (using train_levels)
  pred_label <- fitted_classifier_bundle$levels[pred_indices]
  print(pred_label[2])

  # Evaluate if required
  if(return_classification_report){
    print("CLASSIFICATION REPORT")

    # Get num classes and adjust confusion matrix in case some classes were not predicted
    num_classes <- length(fitted_classifier_bundle$levels)
    all_levels <- 1:num_classes

    pred_factor <- factor(pred_indices, levels = all_levels)
    real_factor <- factor(y_test_encoded, levels = all_levels)

    confusion <- table(Predicted = pred_factor, Real = real_factor)
    print(confusion)
    accuracy <- sum(diag(confusion)) / sum(confusion)

    cat("Accuracy Global:", round(accuracy * 100, 2), "%\n\n")

  }

  # Return predictions and probabilities
  return(list(
    predictions = pred_label,
    probabilities = probabilities
  ))




}














