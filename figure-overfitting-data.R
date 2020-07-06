library(data.table)
library(ggplot2)
library(keras)
## from https://github.com/rstudio/keras/issues/937
if(FALSE){
  keras::install_keras(version = "2.1.6", tensorflow = "1.5")
}
keras::use_implementation("keras")
keras::use_backend("tensorflow")

true.f.list <- list(
  linear=function(x)2*x + 5,
  quadratic=function(x)x^2,
  sin=function(x)5*sin(2*x)+5)
set.seed(1)
N <- 100
n.folds <- 4
unique.folds <- 1:n.folds
set.seed(1)
fold.vec <- sample(rep(unique.folds, l=N))
min.x <- -3
max.x <- 3
x <- runif(N, min.x, max.x)
sim.data.list <- list()
for(pattern in names(true.f.list)){
  true.f <- true.f.list[[pattern]]
  set.seed(1)
  y <- true.f(x) + rnorm(N, 0, 2)
  sim.data.list[[pattern]] <- data.table(
    pattern, x, y, fold=factor(fold.vec))
}
sim.data <- do.call(rbind, sim.data.list)
sim.data[, set := ifelse(fold==test.fold, "validation", "subtrain")]
sim.data[, folds := ifelse(
  fold==test.fold,
  test.fold,
  paste(unique.folds[-test.fold], collapse=","))]
sim.gg <- ggplot()+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  facet_grid(. ~ pattern, labeller=label_both)+ 
  geom_point(aes(
    x, y),
    data=sim.data)
print(sim.gg)

folds.gg <- ggplot()+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  facet_grid(. ~ pattern, labeller=label_both)+ 
  geom_point(aes(
    x, y, color=fold),
    data=sim.data)
print(folds.gg)

test.fold <- 1
sets.gg <- ggplot()+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  facet_grid(set + folds ~ pattern, labeller=label_both)+ 
  geom_point(aes(
    x, y),
    data=sim.data)
print(sets.gg)

model.list <- list(
  linear=keras_model_sequential() %>%
    layer_dense(
      input_shape = 1, units=1, activation="linear"),
  dense=keras_model_sequential() %>%
    layer_dense(
      input_shape = 1, units=50, activation="sigmoid") %>%
    layer_dense(
      units=1, activation="linear")
)
num_steps <- 10
epochs_per_step <- 100
pred.dt.list <- list()
model.name.vec <- names(model.list)
##model.name.vec <- "dense"
for(pattern in names(true.f.list)){
  select.pattern <- data.table(pattern)
  pattern.data <- sim.data[select.pattern, on="pattern"]
  set.data <- list()
  for(set.name in unique(pattern.data$set)){
    set.data[[set.name]] <- with(pattern.data[set==set.name], list(x=x, y=y))
  }
  for(model.name in model.name.vec){
    model <- model.list[[model.name]]
    model %>% compile(
      loss = loss_mean_squared_error,#for regression
      optimizer = optimizer_adadelta()
    )
    for(step in 1:num_steps){
      epoch <- step*epochs_per_step
      cat(paste(pattern, model.name, epoch, "\n"))
      model.metrics <- model %>% fit(
        set.data$subtrain$x, set.data$subtrain$y,
        epochs = epochs_per_step,
        batch_size = length(set.data$subtrain$x),
        validation_data = unname(set.data$validation),
        verbose=0
      )
      pred.dt.list[[paste(pattern, model.name, epoch)]] <- data.table(
        pattern, model.name, epoch,
        step,
        x=pattern.data$x,
        pred.y=as.numeric(predict(model, pattern.data$x)))
    }#for(epoch
  }#for(model.name
}#for(pattern)
pred.dt <- do.call(rbind, pred.dt.list)

sets.gg+
  geom_line(aes(
    x, pred.y, color=model.name),
    data=pred.dt)+
  coord_cartesian(ylim=range(sim.data$y))

sets.gg+
  geom_line(aes(
    x, pred.y, color=model.name),
    data=pred.dt[epoch==200])

nnet.pred.dt.list <- list()
nnet.loss.dt.list <- list()
model.name.vec <- c("dense", "linear")
for(pattern in names(true.f.list)){
  select.pattern <- data.table(pattern)
  pattern.data <- sim.data[select.pattern, on="pattern"]
  set.data <- list()
  for(set.name in unique(pattern.data$set)){
    set.data[[set.name]] <- with(pattern.data[set==set.name], list(x=x, y=y))
  }
  for(hidden.units in c(0, 10, 200)){
    if(hidden.units==0){
      skip <- TRUE
    }else{
      skip <- FALSE
    }
    for(maxit in 10^seq(0, 4)){
      cat(paste(pattern, hidden.units, maxit, "\n"))
      fit <- nnet::nnet(
        y ~ x,
        set.data$subtrain,
        size=hidden.units,
        skip=skip,
        linout=TRUE,
        maxit=maxit)
      pred.x <- sort(c(x, seq(min.x, max.x, l=200)))
      nnet.pred.dt.list[[paste(pattern, hidden.units, maxit)]] <- data.table(
        pattern, hidden.units, maxit,
        x=pred.x,
        pred.y=as.numeric(predict(fit, data.frame(x=pred.x))))
      pattern.data[, pred.y := as.numeric(predict(fit, pattern.data$x))]
      nnet.loss.dt.list[[paste(pattern, hidden.units, maxit)]] <- data.table(
        pattern, hidden.units, maxit,
        pattern.data[, .(mse=mean((pred.y-y)^2)), by=set])
    }#for(epoch
  }#for(model.name
}#for(pattern)

out.list <- list(
  pred=do.call(rbind, nnet.pred.dt.list),
  loss=do.call(rbind, nnet.loss.dt.list))

saveRDS(out.list, "figure-overfitting-data.rds")


nnet.min.dt <- nnet.loss.dt[
set=="validation", .SD[which.min(mse)], by=.(hidden.units, set, pattern)]
nnet.min.dt[, loss := "min"]
ggplot()+
  geom_point(aes(
    maxit, mse, color=hidden.units, shape=loss),
    data=nnet.min.dt)+
  geom_line(aes(
    maxit, mse, color=hidden.units, linetype=set),
    data=nnet.loss.dt)+
  scale_x_log10()+
  scale_y_log10()+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  facet_grid(hidden.units ~ pattern, labeller=label_both, scales="free")

ggplot()+
  geom_point(aes(
    maxit, mse, color=factor(hidden.units), shape=loss),
    data=nnet.min.dt)+
  geom_line(aes(
    maxit, mse, color=factor(hidden.units)),
    data=nnet.loss.dt[set=="validation"])+
  scale_x_log10()+
  scale_y_log10()+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  facet_grid(pattern ~ ., labeller=label_both, scales="free")

sets.gg+
  geom_line(aes(
    x, pred.y, color=factor(hidden.units)),
    data=nnet.pred.dt[maxit==10000])+
  coord_cartesian(ylim=range(sim.data$y))

