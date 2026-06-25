
# Entrenar un clasificador sobre las representaciones latentes
train_classifier_on_representations = function(df_train, exclude_columns, pretrained_model_path, classification_model = NULL, classification_model_type = "MLP") {

  # Load train and test data



  # Extract latent features of train set



  # Train classification model on train set



  # If required, evaluate classification model on test set


  # Return classification model

}




# Una vez entrenado el clasificador, evaluarlo sobre un conjunto de test concreto
downstream_prediction = function(df_test, target_column, exclude_columns, pretrained_model_path, classification_model_path, return_classification_report = FALSE) {

  # Load test data



  # Extract latent features of test set

  # Predict on the test set

  # Evaluate if required

}














