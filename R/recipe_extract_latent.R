


#' @importFrom recipes rand_id add_step step
#' @export
step_extract_latent <- function(
  recipe,
  ...,
  role = NA,
  trained = FALSE,
  pretraining_type = "SCARF",  # De aquí hacia abajo, los parámetros que necesito yo
  exclude_columns = NULL,
  create_validation = FALSE,
  validation_proportion = 0.1,
  batch_size = 256,
  epochs = 1,
  want_labels = FALSE,
  label_column = NULL,
  batch_size_inference = 32,
  pretrained_model = NULL,
  skip = FALSE,  # Skip and ID last arguments
  id = rand_id("extract_latent")

  ) {
    # Add step
    add_step(
      recipe,
      step_extract_latent_new(
        terms = rlang::enquos(...),
        role = role,
        trained = trained,
        pretraining_type = pretraining_type,
        exclude_columns = exclude_columns,
        create_validation = create_validation,
        validation_proportion = validation_proportion,
        batch_size = batch_size,
        epochs = epochs,
        want_labels = want_labels,
        label_column = label_column,
        batch_size_inference = batch_size_inference,
        pretrained_model = pretrained_model,
        skip = skip,
        id = id
      )
    )
  }





# Constructor
step_extract_latent_new <- function(terms, role, trained, pretraining_type, exclude_columns, create_validation, validation_proportion, batch_size, epochs, want_labels, label_column, batch_size_inference, pretrained_model, skip, id) {
  recipes::step(
    terms = terms,
    role = role,
    trained = trained,
    pretraining_type = pretraining_type,
    exclude_columns = exclude_columns,
    create_validation = create_validation,
    validation_proportion = validation_proportion,
    batch_size = batch_size,
    epochs = epochs,
    want_labels = want_labels,
    label_column = label_column,
    batch_size_inference = batch_size_inference,
    pretrained_model = pretrained_model,
    skip = skip,
    id = id
  )
}








# Prep
#' Title
#'
#' @param x The step_extract_latent object
#' @param training a
#' @param info a
#' @param ... a
#'
#' @returns a
#' @importFrom recipes prep
#' @export
#'
#' @examples
#' a <- 1
prep.step_extract_latent <- function(x, training, info = NULL, ...) {

  col_names <- recipes::recipes_eval_select(x$terms, training, info)  # Select columns that the user listed
  training_data <- as.data.frame(training[, col_names])

  # Prep logic. Adjust to new data (training)

  # We have SCARF_fit, so we can reuse this method
  # TODO: hay que gestionar el problema de read_data() dentro de scarf fit.
  # TODO: step_novel, step_normalize y step_dummy que necesita SCARF, se presupone que lo hace el usuario? Entiendo que si
  pretrained_SCARF <- scarf_fit(
    dataframe_train = training_data,
    exclude_columns = x$exclude_columns,
    create_validation = x$create_validation,
    validation_proportion = x$validation_proportion,
    batch_size = x$batch_size,
    n_epochs = x$epochs,
    save_path = NULL,  # Force NULL so that the model is stored in RAM, ready for bake and avoiding the need to store it locally.
    preprocess = FALSE,
  )


  # Use the constructor function to return the updated object.
  step_extract_latent_new(
    terms = x$terms,
    role = x$role,
    trained = TRUE,  # As prep is completed, we set trained to TRUE
    pretraining_type = x$pretraining_type,
    exclude_columns = x$exclude_columns,
    create_validation = x$create_validation,
    validation_proportion = x$validation_proportion,
    batch_size = x$batch_size,
    epochs = x$epochs,
    want_labels = x$want_labels,
    label_column = x$label_column,
    batch_size_inference = x$batch_size_inference,
    pretrained_model = pretrained_SCARF,  # Store the pretrained model
    skip = x$skip,
    id = x$id
  )

}




# Bake
#' Title
#'
#' @param object The updated step function that has been through prep()
#' @param new_data Tibble of data to be processed
#' @param ... a
#'
#' @returns a
#' @importFrom recipes bake
#' @export
#'
#' @examples
#' a <- 1
bake.step_extract_latent <- function(object, new_data, ...) {
  col_names <- recipes::recipes_eval_select(object$terms, training, info)  # Select columns that the user listed
  data_to_extract <- as.data.frame(training[, col_names])

  # Bake logic. Apply the pretrained model to new data
  extracted_data <- scarf_feature_extractor(
    dataframe = data_to_extract,
    pretrained_model = object$pretrained_model,
    exclude_columns = object$exclude_columns,
    want_labels = object$want_labels,
    label_column = object$label_column,
    batch_size = object$batch_size_inference
  )

  # Create output as extracted features + label column if required
  features <- extracted_data$features
  labels <- extracted_data$features_labels

  colnames(features) <- paste0("extracted_dim_", 1:ncol(features))

  if (is.null(labels)) {
    # Just return the extracted features
    return(tibble::as_tibble(features))
  } else {
    # Return the extracted features plus associated labels
    return(tibble::as_tibble(cbind(features, labels)))
  }


}

