




step_extract_latent <- function(
  recipe,
  ...,
  role = NA,
  trained = FALSE,
  pretraining_type = "SCARF",  # De aquí hacia abajo, los parámetros que necesito yo
  skip = FALSE,  # Skip and ID last arguments
  id = recipes::rand_id("extract_latent")


  ) {
    # Add step
    recipes::add_step(
      recipe,
      step_extract_latent_new(
        terms = rlang::enquos(...),
        role = role,
        trained = trained,
        pretraining_type = pretraining_type,
        skip = skip,
        id = id
      )
    )
  }



# Constructor
step_extract_latent_new <- function(terms, role, trained, pretraining_type, skip, id) {
  step(
    terms = terms,
    role = role,
    trained = trained,
    pretraining_type = pretraining_type,
    skip = skip,
    id = id
  )
}








# Prep
recipes::prep.step_extract_latent <- function(x, training, info = NULL, ...) {
  col_names <- recipes::recipes_eval_select(x$terms, training, info)  # Select columns that the user listed


  # Prep logic. Adjust to new data


}




# Bake
recipes::bake.step_extract_latent <- function(object, new_data, ...) {
  a <- 1
}

