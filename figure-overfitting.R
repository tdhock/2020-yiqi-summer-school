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
set.seed(1) #for reproducibility.
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
test.fold <- 1
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
    data=sim.data)+
  xlab("input/feature x")+
  ylab("output/label y")
png("figure-overfitting-data.png", width=5, height=3, units="in", res=200)
print(sim.gg)
dev.off()

folds.gg <- ggplot()+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  facet_grid(. ~ pattern, labeller=label_both)+ 
  geom_point(aes(
    x, y, color=fold),
    data=sim.data)+
  xlab("input/feature x")+
  ylab("output/label y")
png("figure-overfitting-data-folds.png", width=5, height=3, units="in", res=200)
print(folds.gg)
dev.off()

sets.gg <- ggplot()+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  facet_grid(set + folds ~ pattern, labeller=label_both)+ 
  geom_point(aes(
    x, y),
    data=sim.data)+
  xlab("input/feature x")+
  ylab("output/label y")
png("figure-overfitting-data-sets.png", width=5, height=3, units="in", res=200)
print(sets.gg)
dev.off()

nnet.pred.dt.list <- list()
nnet.loss.dt.list <- list()
hidden.units.vec <- c(0, 10, 200)
maxit.vec <- 10^seq(0, 4)
for(pattern in names(true.f.list)){
  select.pattern <- data.table(pattern)
  pattern.data <- sim.data[select.pattern, on="pattern"]
  set.data <- list()
  for(set.name in unique(pattern.data$set)){
    set.data[[set.name]] <- with(pattern.data[set==set.name], list(x=x, y=y))
  }
  for(hidden.units in hidden.units.vec){
    if(hidden.units==0){
      skip <- TRUE
    }else{
      skip <- FALSE
    }
    for(maxit in maxit.vec){
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
nnet.pred.dt <- do.call(rbind, nnet.pred.dt.list)
nnet.loss.dt <- do.call(rbind, nnet.loss.dt.list)

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

slides.list <- list()
for(units in c(10, 0, 200)){
  for(it in maxit.vec){
    gg <- sets.gg+
      ggtitle(paste(
        "Neural network,",
        units,
        "hidden units,",
        it,
        "gradient descent iterations"))+
      geom_line(aes(
        x, pred.y),
        color="red",
        data=nnet.pred.dt[maxit==it & hidden.units==units])+
      coord_cartesian(ylim=range(sim.data$y))
    out.png <- sprintf(
      "figure-overfitting-pred-units=%d-maxit=%d.png", units, it)
    print(out.png)
    png(out.png,
        width=7, height=3, units="in", res=200)
    print(gg)
    dev.off()
    slides.list[[length(slides.list)+1]] <- out.png
  }
  gg.units <- ggplot()+
    ggtitle(paste(
      "Neural network,",
      units,
      "hidden units"))+
    geom_point(aes(
      maxit, mse, shape=loss),
      data=nnet.min.dt[hidden.units==units])+
    geom_line(aes(
      maxit, mse, linetype=set),
      data=nnet.loss.dt[hidden.units==units])+
    scale_x_log10(
      "Max iterations of gradient descent learning algorithm",
      limits=c(1, 5e4))+
    scale_y_log10(
      "Mean Squared Error")+
    theme_bw()+
    theme(panel.spacing=grid::unit(0, "lines"))+
    facet_grid(. ~ pattern, labeller=label_both, scales="free")
  out.png <- sprintf("figure-overfitting-data-loss-%d.png", units)
  png(out.png,
      width=7, height=3, units="in", res=200)
  print(gg.units)
  dev.off()
  slides.list[[length(slides.list)+1]] <- out.png
}

slides.vec <- sprintf("
\\begin{frame}
  \\includegraphics[width=\\textwidth]{%s}
\\end{frame}
", slides.list)
writeLines(slides.vec, "figure-overfitting.tex")
