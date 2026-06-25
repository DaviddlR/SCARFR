
#' Title
#'
#' @param data Original data samples
#'
#' @returns A 'torch::dataset' that contains the data.
#
# @examples
# X_fake <- matrix(runif(50*4), nrow=50, ncol=4)
#
# my_dataset <- create_tensor_dataset(X_fake)
#
# first_item <- my_dataset$.getitem(1)
# print(first_item$x)
create_tensor_dataset <- torch::dataset(

  name = "create_tensor_dataset",


  initialize = function(data) {
    self$data <- as.matrix(data)
  },


  .getitem = function(i) {
    data <- self$data[i, ]  # All columns of a row

    data_tensor <- torch::torch_tensor(data, dtype = torch::torch_float32())

    list(x = data_tensor)

  },


  .length = function(){
    nrow(self$data)
  }

)







#' Creation of a tensor dataset.
#'
#' @param data Original data samples
#' @param target Original label samples
#'
#' @return A 'torch::dataset' that contains the data.
#
#
# @examples
# X_fake <- matrix(runif(50*4), nrow=50, ncol=4)
# y_fake <- sample(0:1, 50, replace=TRUE)
#
# my_dataset <- create_tensor_dataset_with_label(X_fake, y_fake)
#
# first_item <- my_dataset$.getitem(1)
# print(first_item$x)
# print(first_item$y)
create_tensor_dataset_with_label <- torch::dataset(
  name = "tensor_dataset_no_label",

  initialize = function(data, target) {
    self$data <- as.matrix(data)
    self$target <- as.vector(target)
  },


  .getitem = function(i) {
    data <- self$data[i, ]  # All columns of a row
    label <- self$target[i]

    data_tensor <- torch::torch_tensor(data, dtype = torch::torch_float32())
    label_tensor <- torch::torch_tensor(label, dtype = torch::torch_long())

    list(x = data_tensor, y = label_tensor)

  },


  .length = function(){
    nrow(self$data)
  }


)






