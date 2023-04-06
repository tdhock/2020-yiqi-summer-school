library(data.table)
library(ggplot2)
library(torch)
source("download.R")
zip.dt <- data.table::fread("zip.gz")
zip.y.array <- array(zip.dt$V1, nrow(zip.dt))
zip.y.tensor <- torch::torch_tensor(zip.y.array+1L, torch::torch_long())
head(zip.y.tensor)

zip.feature.dt <- zip.dt[,-1]
(zip.size <- sqrt(ncol(zip.feature.dt)))
zip.X.array <- array(
  unlist(zip.feature.dt),
  c(nrow(zip.dt), 1, zip.size, zip.size))
zip.X.tensor <- torch::torch_tensor(zip.X.array)
str(zip.X.tensor)

image(zip.X.array[1,,,])
str(zip.y.tensor)

## some digits to display.
zip.some <- zip.dt[1:12]
zip.some[, observation.i := 1:.N]
zip.some.tall <- zip.some[, {
  data.table::data.table(
    label=V1,
    col.i=rep(1:zip.size, zip.size),
    row.i=rep(1:zip.size, each=zip.size),
    intensity=as.numeric(.SD[, paste0("V", 2:257), with=FALSE]))
}, by=observation.i]
gg.digits <- ggplot()+
  geom_tile(aes(
    x=col.i, y=-row.i, fill=intensity),
    data=zip.some.tall)+
  facet_wrap(observation.i + label ~ ., labeller=label_both)+
  scale_x_continuous(breaks=c(1, 16))+
  scale_y_continuous(breaks=-c(1, 16))+
  coord_equal()+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  scale_fill_gradient(low="black", high="white")
png("figure-validation-loss-torch-digits.png",
    width=5.5, height=5, units="in", res=200)
print(gg.digits)
dev.off()

## Linear model is defined by a vector of weights, one for each input
## feature, and intercept/bias.
(n.classes <- length(unique(zip.y.array)))
(linear.model <- torch::nn_sequential(
  torch::nn_flatten(),
  torch::nn_linear(ncol(zip.feature.dt), n.classes)))
linear.obj <- torch::nn_linear(ncol(zip.feature.dt), n.classes)
linear.obj$weight
linear.obj$bias
linear.obj$weight$grad
linear.obj$bias$grad

linear.seq <- torch::nn_sequential(
  torch::nn_flatten(),
  linear.obj)
pred.tensor <- linear.seq(zip.X.tensor)
loss.fun <- torch::nn_cross_entropy_loss()
loss.tensor <- loss.fun(pred.tensor, zip.y.tensor)
loss.tensor$backward()

get_loss <- function(L, model, index.vec=seq_along(L$y)){
  X <- L$X[index.vec,,,,drop=FALSE]
  y <- L$y[index.vec]
  pred <- model(X)
  loss.fun(pred, y)
}

get_loss(list(X=zip.X.tensor, y=zip.y.tensor), linear.seq)



step.size <- 0.1
linear.obj$bias
linear.obj$bias$grad
## sub_ is in-place subtraction
linear.obj$bias-step.size*linear.obj$bias$grad
torch::with_no_grad({
  linear.obj$bias$sub_(step.size*linear.obj$bias$grad)
})
linear.obj$weight$grad

optimizer <- torch::optim_sgd(linear.seq$parameters, lr=step.size)
linear.obj$bias
optimizer$step()

get_batch_list <- function(n_data, batch_size=200){
  n_batches <- ceiling(n_data / batch_size)
  batch.vec <- sample(rep(1:n_batches,each=batch_size)[1:n_data])
  split(seq_along(batch.vec), batch.vec)
}

## torch batching.
zip_dataset <- torch::dataset(
  name = "subtrain_dataset",
  initialize = function(X.tensor, y.tensor) {
    self$X <- X.tensor
    self$y <- y.tensor
  },
  .getbatch = function(index.vec) {
    list(X=self$X[index.vec,,,,drop=FALSE], y=self$y[index.vec])
  },    
  .length = function() {
    self$y$size()
  }
)
zip_dataset <- zip_dataset(zip.X.tensor, zip.y.tensor)
zip.batch.list <- zip_dataset$.getbatch(1:2)
zip_loader <- torch::dataloader(zip_dataset, batch_size=batch_size, shuffle=TRUE)

## my batching.
take_steps <- function(subtrain.list, model, batch_size=200, step.size=0.1){
  optimizer <- torch::optim_sgd(model$parameters, lr=step.size)
  batch.list <- get_batch_list(length(subtrain.list$y), batch_size)
  for(batch.number in seq_along(batch.list)){
    batch.indices <- batch.list[[batch.number]]
    batch.loss <- get_loss(subtrain.list, model, batch.indices)
    optimizer$zero_grad()
    batch.loss$backward()
    optimizer$step()
  }
}

n.folds <- 3
num_epochs <- 50
uniq.folds <- 1:n.folds
set.seed(1)
fold.vec <- sample(rep(uniq.folds, l=nrow(zip.dt)))
loss.dt.list <- list()
test.fold <- 1
is.train <- fold.vec!=test.fold
train.i <- which(is.train)
is.subtrain <- rep(c(TRUE,TRUE,FALSE),l=length(train.i))
is.set.list <- list(
  test=!is.train,
  subtrain=train.i[is.subtrain],
  validation=train.i[!is.subtrain])
tensor.list <- list()
for(set.name in names(is.set.list)){
  is.set <- is.set.list[[set.name]]
  tensor.list[[set.name]] <- list(
    X=zip.X.tensor[is.set,,,,drop=FALSE],
    y=zip.y.tensor[is.set])
}
linear.model <- torch::nn_sequential(
  torch::nn_flatten(),
  torch::nn_linear(ncol(zip.feature.dt), n.classes))
n.subtrain <- length(tensor.list$subtrain$y)
for(epoch in 1:num_epochs){
  take_steps(tensor.list$subtrain, linear.model)
  torch::with_no_grad({
    for(set in c("subtrain", "validation")){
      set.data <- tensor.list[[set]]
      set.loss <- get_loss(set.data, linear.model)
      loss.dt.list[[paste(test.fold, epoch, set)]] <- print(data.table(
        test.fold, epoch, set, loss=as.numeric(set.loss)))
    }
  })
}
(loss.dt <- rbindlist(loss.dt.list))

min.dt <- loss.dt[, .SD[which.min(loss)], by=set]
gg <- ggplot()+
  ggtitle("Linear model")+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  geom_line(aes(
    epoch, loss, color=set),
    data=loss.dt)+
  geom_point(aes(
    epoch, loss, color=set, shape=point),
    data=data.table(point="min", min.dt))+
  scale_y_log10("loss value (lower for better predictions)")+
  scale_x_continuous(
    "epoch (gradient descent passes through subtrain data)",
    limits=c(0, num_epochs*1.2),
    breaks=seq(0, num_epochs, by=10))
dl <- directlabels::direct.label(gg, "right.polygons")
png(
  "figure-validation-loss-torch-linear.png", 
  width=6, height=4, units="in", res=200)
print(dl)
dev.off()

linear.metrics <- linear.model %>% fit(
  zip.X.array, zip.y.mat,
  batch_size = 100,
  epochs = num_epochs,
  validation_split = 0.2
)
linear.wide <- do.call(data.table::data.table, linear.metrics$metrics)
linear.wide[, epoch := 1:.N]
(linear.tall <- nc::capture_melt_single(
  linear.wide,
  set="val_|", function(x)ifelse(x=="val_", "validation", "subtrain"),
  metric="loss|acc"))
linear.min.loss <- linear.tall[metric=="loss", .SD[which.min(value)], by=set]
linear.min.loss[, loss := "min"]

linear.gg <- ggplot()+
  ggtitle("Linear model")+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  geom_line(aes(
    epoch, value, color=set),
    data=linear.tall[metric=="loss"])+
  geom_point(aes(
    epoch, value, color=set, shape=loss),
    data=linear.min.loss)+
  scale_y_log10("loss value (lower for better predictions)")+
  scale_x_continuous(
    "epoch (gradient descent passes through subtrain data)",
    limits=c(0, num_epochs*1.2),
    breaks=seq(0, num_epochs, by=10))
linear.dl <- directlabels::direct.label(linear.gg, "right.polygons")
png("figure-validation-loss-linear.png",
    width=6, height=4, units="in", res=200)
print(linear.dl)
dev.off()

## Deep sparse model.
conv.model <- keras_model_sequential() %>%
  layer_conv_2d(
    input_shape = dim(zip.X.array)[-1],
    filters = 20,
    kernel_size = c(3,3),
    activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 100, activation = 'relu') %>% 
  layer_dense(units = ncol(zip.y.mat), activation = 'softmax')
str(conv.model)
conv.model %>% compile(
  loss = loss_categorical_crossentropy,#for multi-class classification
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)
conv.metrics <- conv.model %>% fit(
  zip.X.array, zip.y.mat,
  batch_size = 100,
  epochs = num_epochs,
  validation_split = 0.2
)
conv.wide <- do.call(data.table::data.table, conv.metrics$metrics)
conv.wide[, epoch := 1:.N]
(conv.tall <- nc::capture_melt_single(
  conv.wide,
  set="val_|", function(x)ifelse(x=="val_", "validation", "subtrain"),
  metric="loss|acc"))
conv.min.loss <- conv.tall[metric=="loss", .SD[which.min(value)], by=set]
conv.min.loss[, loss := "min"]

conv.gg <- ggplot()+
  ggtitle("Convolutional neural network")+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  geom_line(aes(
    epoch, value, color=set),
    data=conv.tall[metric=="loss"])+
  geom_point(aes(
    epoch, value, color=set, shape=loss),
    data=conv.min.loss)+
  scale_y_log10("loss value (lower for better predictions)")+
  scale_x_continuous(
    "epoch (gradient descent passes through subtrain data)",
    limits=c(0, num_epochs*1.2),
    breaks=seq(0, num_epochs, by=10))
conv.dl <- directlabels::direct.label(conv.gg, "right.polygons")
png("figure-validation-loss-conv.png",
    width=6, height=4, units="in", res=200)
print(conv.dl)
dev.off()

both.tall <- rbind(
  data.table(model="convolutional", conv.tall),
  data.table(model="linear", linear.tall))
both.gg <- ggplot()+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  facet_grid(. ~ model)+
  geom_line(aes(
    epoch, value, color=set),
    data=both.tall[metric=="loss"])+
  scale_y_log10("loss value (lower for better predictions)")+
  scale_x_continuous(
    "epoch (gradient descent passes through subtrain data)",
    limits=c(0, num_epochs*1.2),
    breaks=seq(0, num_epochs, by=10))
directlabels::direct.label(both.gg, "right.polygons")

## QUIZ 1. purpose of train/subtrain/validation/test
## sets. 2. overfitting/underfitting. 3. data input format for
## ML. 4. cross-validation fold ID / test/train sets.

## deep dense model.
dense.model <- keras_model_sequential() %>%
  layer_flatten(
    input_shape = dim(zip.X.array)[-1]) %>%
  layer_dense(units = 100, activation = 'relu') %>% 
  layer_dense(units = 100, activation = 'relu') %>% 
  layer_dense(units = 100, activation = 'relu') %>% 
  layer_dense(units = 100, activation = 'relu') %>%
  layer_dense(units = 100, activation = 'relu') %>%
  layer_dense(units = 100, activation = 'relu') %>% 
  layer_dense(units = 100, activation = 'relu') %>%   
  layer_dense(units = 100, activation = 'relu') %>% 
  layer_dense(units = ncol(zip.y.mat), activation = 'softmax')
dense.model %>% compile(
  loss = loss_categorical_crossentropy,#for multi-class classification
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)
dense.metrics <- dense.model %>% fit(
  zip.X.array, zip.y.mat,
  batch_size = 100,
  epochs = num_epochs,
  validation_split = 0.2
)

dense.wide <- do.call(data.table::data.table, dense.metrics$metrics)
dense.wide[, epoch := 1:.N]
(dense.tall <- nc::capture_melt_single(
  dense.wide,
  set="val_|", function(x)ifelse(x=="val_", "validation", "subtrain"),
  metric="loss|acc"))
dense.min.loss <- dense.tall[metric=="loss", .SD[which.min(value)], by=set]
dense.min.loss[, loss := "min"]
dense.gg <- ggplot()+
  ggtitle("Dense (fully connected) neural network with 8 hidden layers")+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  geom_line(aes(
    epoch, value, color=set),
    data=dense.tall[metric=="loss"])+
  geom_point(aes(
    epoch, value, color=set, shape=loss),
    data=dense.min.loss)+
  scale_y_log10("loss value (lower for better predictions)")+
  scale_x_continuous(
    "epoch (gradient descent passes through subtrain data)",
    limits=c(0, num_epochs*1.2),
    breaks=seq(0, num_epochs, by=10))
dense.dl <- directlabels::direct.label(dense.gg, "right.polygons")
png("figure-validation-loss-dense.png",
    width=6, height=4, units="in", res=200)
print(dense.dl)
dev.off()

three.tall <- rbind(
  both.tall,
  data.table(model="dense", dense.tall))
three.gg <- ggplot()+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  facet_grid(. ~ model)+
  geom_line(aes(
    epoch, value, color=set),
    data=three.tall[metric=="loss"])+
  scale_y_log10("loss value (lower for better predictions)")+
  scale_x_continuous(
    "epoch (gradient descent passes through subtrain data)",
    limits=c(0, num_epochs*1.2),
    breaks=seq(0, num_epochs, by=10))
three.dl <- directlabels::direct.label(three.gg, "right.polygons")
png("figure-validation-loss-three.png",
    width=6, height=4, units="in", res=200)
print(three.gg)
dev.off()
