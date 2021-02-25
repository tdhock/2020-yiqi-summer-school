library(data.table)
library(ggplot2)

n.outer.folds <- 3
unique.outer.folds <- 1:n.outer.folds
n.inner.folds <- 4
unique.inner.folds <- 1:n.inner.folds
sim.data.list <- list()
csv.vec <- Sys.glob("data_*.csv")
for(csv in csv.vec){
  pattern <- gsub("data_|.csv", "", csv)
  sim.dt <-  data.table::fread(csv)
  set.seed(1)
  outer.fold.vec <- sample(rep(unique.outer.folds, l=nrow(sim.dt)))
  sim.data.list[[pattern]] <- data.table(
    pattern, sim.dt, outer.fold=factor(outer.fold.vec))
}
sim.data <- do.call(rbind, sim.data.list)
pred.x <- sort(c(sim.data$x, seq(min(sim.data$x), max(sim.data$x), l=200)))
grid.dt <- data.table(x=pred.x)

folds.gg <- ggplot()+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  facet_grid(. ~ pattern, labeller=label_both)+ 
  geom_point(aes(
    x, y, color=outer.fold),
    data=sim.data)+
  geom_text(aes(
    -Inf, -Inf, label=paste0("N=", N)),
    vjust=-0.5,
    hjust=-0.1,
    size=3,
    data=sim.data[, .(N=.N), by=pattern])+
  xlab("input/feature x")+
  ylab("output/label y")+
  theme(legend.position="bottom")
png(
  "figure-overfitting-cv-data-outer-folds.png",
  width=5, height=3, units="in", res=200)
print(folds.gg)
dev.off()

test.loss.dt.list <- list()
hidden.units.vec <- c(2, 20, 200)
maxit.vec <- 10^seq(0, 3)
for(test.fold in unique.outer.folds){
  sim.data[, outer.set := ifelse(outer.fold==test.fold, "test", "train")]
  sim.data[, outer.folds := ifelse(
    outer.fold==test.fold,
    test.fold,
    paste(unique.outer.folds[-test.fold], collapse=","))]
  sets.gg <- ggplot()+
    theme_bw()+
    theme(panel.spacing=grid::unit(0, "lines"))+
    facet_grid(outer.set + outer.folds ~ pattern, labeller=label_both)+ 
    geom_point(aes(
      x, y),
      data=sim.data)+
    geom_text(aes(
      -Inf, -Inf, label=paste0("N=", N)),
      vjust=-0.5,
      hjust=-0.1,
      size=3,
      data=sim.data[, .(N=.N), by=.(pattern, outer.set, outer.folds)])+
    xlab("input/feature x")+
    scale_y_continuous("output/label y", limits=c(-6, 16))
  png(
    sprintf("figure-overfitting-cv-data-test-fold-%d.png", test.fold),
    width=5, height=3, units="in", res=200)
  print(sets.gg)
  dev.off()
  is.train <- sim.data$outer.set == "train"
  set.seed(1)
  sim.data[, inner.fold := NA_character_]
  sim.data[is.train, inner.fold := {
    inner.fold.vec <- sample(rep(unique.inner.folds, l=.N))
    factor(inner.fold.vec)
  }, by=pattern]
  folds.gg <- sets.gg+
    geom_point(aes(
      x, y, fill=inner.fold),
      shape=21,
      color="black",
      data=sim.data[is.train])
  png(
    sprintf("figure-overfitting-cv-data-inner-folds-%d.png", test.fold),
    width=5, height=3, units="in", res=200)
  print(folds.gg)
  dev.off()
  valid.loss.dt.list <- list()
  for(validation.fold in unique.inner.folds){
    sim.data[, inner.set := ifelse(
      inner.fold==validation.fold, "validation", "subtrain")]
    sim.data[, inner.folds := ifelse(
      inner.fold==validation.fold,
      validation.fold,
      paste(unique.inner.folds[-validation.fold], collapse=","))]
    inner.gg <- ggplot()+
      theme_bw()+
      theme(panel.spacing=grid::unit(0, "lines"))+
      geom_text(aes(
        -Inf, -Inf, label=paste0("N=", N)),
        vjust=-0.5,
        hjust=-0.1,
        size=3,
        data=sim.data[, .(
          N=.N
        ), by=.(
          pattern, outer.set, outer.folds, inner.set, inner.folds
        )])+
      facet_grid(
        outer.set + outer.folds + inner.set + inner.folds ~ pattern,
        labeller=label_both)+ 
      geom_point(aes(
        x, y),
        data=sim.data)+
      scale_x_continuous("input/feature x")+
      scale_y_continuous("output/label y", limits=c(-6, 16))
    png(
      sprintf(
        "figure-overfitting-cv-data-inner-folds-%d-%d.png",
        test.fold, validation.fold),
      width=6, height=4, units="in", res=200)
    print(inner.gg)
    dev.off()
    combo.df <- expand.grid(
      hidden.units=hidden.units.vec,
      maxit=maxit.vec,
      pattern=names(sim.data.list))
    for(combo.i in 1:nrow(combo.df)){
      combo.row <- combo.df[combo.i,]
      pattern.data <- sim.data[pattern==combo.row$pattern]
      subtrain.df <- pattern.data[inner.set=="subtrain"]
      set.seed(1)
      fit <- with(combo.row, nnet::nnet(
        y ~ x,
        subtrain.df,
        size=hidden.units,
        skip=FALSE,
        linout=TRUE,
        maxit=maxit))
      valid.dt <- pattern.data[inner.set=="validation"]
      valid.dt[, pred := predict(fit, .SD)]
      valid.loss.dt.list[[paste(validation.fold, combo.i)]] <- data.table(
        validation.fold, 
        combo.row,
        valid.dt[, .(mse=mean((pred-y)^2))])
    }#combo.i
  }#validation.fold
  valid.loss.dt <- do.call(rbind, valid.loss.dt.list)
  valid.loss.stats <- valid.loss.dt[, .(
    mean.mse=mean(mse),
    sd.mse=sd(mse),
    median.mse=median(mse),
    q25=quantile(mse, 0.25),
    q75=quantile(mse, 0.75),
    folds=.N
  ), by=.(pattern, hidden.units, maxit)]
  units.colors <- c(
    "2"="#6BAED6", #"#4292C6", 
    "20"="#2171B5", #"#08519C",
    "200"="#08306B")
  valid.loss.stats[, hidden.units.fac := factor(hidden.units)]
  (valid.only.min <- valid.loss.stats[, .SD[which.min(median.mse)], by=pattern])
  gg <- ggplot()+
    ggtitle(paste0("Train set for outer fold ID=", test.fold))+
    geom_ribbon(aes(
      maxit,
      ymax=q75,
      ymin=q25,
      fill=hidden.units.fac),
      size=1,
      alpha=0.3,
      data=valid.loss.stats)+
    geom_line(aes(
      maxit, median.mse, color=hidden.units.fac),
      size=1,
      data=valid.loss.stats)+
    directlabels::geom_dl(aes(
      maxit, median.mse, label=sprintf(
        "min maxit=%d\nunits=%d", maxit, hidden.units)),
      method=list(cex=0.8, "bottom.polygons"),
      color="white",
      data=valid.only.min)+
    geom_point(aes(
      maxit, median.mse),
      fill="white",
      shape=21,
      data=valid.only.min)+
    scale_color_manual("hidden\nunits", values=units.colors)+
    scale_fill_manual("hidden\nunits", values=units.colors)+
    scale_x_log10(
      "Maximum number of iterations/epochs in gradient descent learning algorithm",
      limits=valid.loss.stats[, c(min(maxit), max(maxit)*5)])+
    scale_y_log10(
      "Mean squared error\n(median/quartiles over four validation sets)")+
    coord_cartesian(ylim=c(0.5, max(valid.loss.stats$median.mse)))+
    theme_bw()+
    theme(panel.spacing=grid::unit(0, "lines"))+
    facet_grid(. ~ pattern, labeller=label_both)
  (dl <- directlabels::direct.label(gg, "right.polygons"))
  png(
    sprintf("figure-overfitting-cv-data-median-mse-%d.png", test.fold),
    width=7, height=3, units="in", res=200)
  print(dl)
  dev.off()
  disp.models.list <- list()
  disp.resid.list <- list()
  for(pat in names(sim.data.list)){
    pattern.data <- sim.data[pattern==pat]
    pattern.train <- pattern.data[outer.set=="train"]
    pattern.test <- pattern.data[outer.set=="test"]
    pattern.min <- valid.only.min[pattern==pat]
    fit <- pattern.min[, nnet::nnet(
      y ~ x,
      pattern.train,
      size=hidden.units,
      skip=FALSE,
      linout=TRUE,
      maxit=maxit)]
    featureless.pred <- mean(pattern.train$y)
    pred.list <- list(
      nnet=as.numeric(predict(fit, pattern.test)),
      featureless=rep(featureless.pred, nrow(pattern.test)))
    disp.models.list[[pat]] <- data.table(pattern=pat, rbind(
      data.table(
        algorithm="nnet",
        grid.dt, pred=as.numeric(predict(fit, grid.dt))),
      data.table(
        algorithm="featureless",
        grid.dt[, .(x=range(x), pred=featureless.pred)])))
    for(algorithm in names(pred.list)){
      pred <- pred.list[[algorithm]]
      test.loss.dt.list[[paste(test.fold, pat, algorithm)]] <- data.table(
        test.fold,
        pattern=pat,
        algorithm,
        mse=mean((pattern.test$y - pred)^2))
      disp.resid.list[[paste(pat, algorithm)]] <- data.table(
        algorithm,
        pred,
        pattern.test[, .(x, y, pattern, outer.folds, outer.set)])
    }
  }
  disp.models <- do.call(rbind, disp.models.list)
  disp.resid <- do.call(rbind, disp.resid.list)
  sets.gg <- ggplot()+
    theme_bw()+
    theme(panel.spacing=grid::unit(0, "lines"))+
    facet_grid(outer.set + outer.folds ~ pattern, labeller=label_both)+ 
    geom_point(aes(
      x, y),
      color="grey50",
      data=sim.data)+
    xlab("input/feature x")+
    scale_y_continuous("output/label y", limits=c(-6, 16))+
    ## geom_segment(aes(
    ##   x, pred,
    ##   xend=x, yend=y,
    ##   color=algorithm),
    ##   data=disp.resid)+
    geom_line(aes(
      x, pred, color=algorithm),
      size=1,
      data=disp.models)+
    theme(legend.position="bottom")
  png(
    sprintf("figure-overfitting-cv-data-test-fold-%d-pred.png", test.fold),
    width=6, height=3.5, units="in", res=200)
  print(sets.gg)
  dev.off()
}

test.loss.dt <- do.call(rbind, test.loss.dt.list)
gg <- ggplot()+
  geom_point(aes(
    mse, algorithm, color=factor(test.fold)),
    alpha=0.75,
    data=test.loss.dt)+
  scale_color_discrete("Test fold")+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  facet_grid(. ~ pattern, labeller=label_both)+
  scale_x_continuous(
    "Mean Squared Error of predictions on test set")
png(
  "figure-overfitting-cv-data.png",
  width=6, height=1.5, units="in", res=200)
print(gg)
dev.off()
  
