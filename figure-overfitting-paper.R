library(data.table)
library(ggplot2)

n.folds <- 4
unique.folds <- 1:n.folds
sim.data.list <- list()
csv.vec <- Sys.glob("data_*.csv")
for(csv in csv.vec){
  pattern <- gsub("data_|.csv", "", csv)
  sim.dt <-  data.table::fread(csv)
  set.seed(1)
  fold.vec <- sample(rep(unique.folds, l=nrow(sim.dt)))
  sim.data.list[[pattern]] <- data.table(
    pattern, sim.dt, fold=factor(fold.vec))
}
sim.data <- do.call(rbind, sim.data.list)

validation.fold <- 1
sim.data[, set := ifelse(fold==validation.fold, "validation", "subtrain")]
sim.data[, folds := ifelse(
  fold==validation.fold,
  validation.fold,
  paste(unique.folds[-validation.fold], collapse=","))]
sim.gg <- ggplot()+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  facet_grid(. ~ pattern, labeller=label_both)+ 
  geom_point(aes(
    x, y),
    data=sim.data)+
  xlab("input/feature x")+
  ylab("output/label y")
print(sim.gg)

folds.gg <- ggplot()+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  facet_grid(. ~ pattern, labeller=label_both)+ 
  geom_point(aes(
    x, y, color=fold),
    data=sim.data)+
  xlab("input/feature x")+
  ylab("output/label y")
print(folds.gg)

sets.gg <- ggplot()+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  facet_grid(set + folds ~ pattern, labeller=label_both)+ 
  geom_point(aes(
    x, y),
    data=sim.data)+
  xlab("input/feature x")+
  ylab("output/label y")
print(sets.gg)

nnet.pred.dt.list <- list()
nnet.loss.dt.list <- list()
nnet.resid.dt.list <- list()
epochs.vec <- 2^seq(0, 10)
pattern <- "sin"
select.pattern <- data.table(pattern)
pattern.data <- sim.data[select.pattern, on="pattern"]
set.data <- list()
for(set.name in unique(pattern.data$set)){
  set.data[[set.name]] <- with(pattern.data[set==set.name], list(x=x, y=y))
}
hidden.units <- 50
for(epochs in epochs.vec){
  cat(paste(pattern, hidden.units, epochs, "\n"))
  set.seed(1)
  fit <- nnet::nnet(
    y ~ x,
    set.data$subtrain,
    size=hidden.units,
    linout=TRUE,
    maxit=epochs)
  pred.x <- sort(c(sim.data$x, seq(min(sim.data$x), max(sim.data$x), l=200)))
  nnet.pred.dt.list[[paste(epochs)]] <- data.table(
    epochs,
    x=pred.x,
    pred.y=as.numeric(predict(fit, data.frame(x=pred.x))))
  pattern.data[, pred.y := as.numeric(predict(fit, pattern.data))]
  nnet.resid.dt.list[[paste(epochs)]] <- data.table(
    epochs, pattern.data)
  nnet.loss.dt.list[[paste(epochs)]] <- data.table(
    epochs,
    pattern.data[, .(mse=mean((pred.y-y)^2)), by=set])
}#for(epoch
nnet.pred.dt <- do.call(rbind, nnet.pred.dt.list)
nnet.resid.dt <- do.call(rbind, nnet.resid.dt.list)
nnet.loss.dt <- do.call(rbind, nnet.loss.dt.list)
nnet.min.dt <- nnet.loss.dt[
  set=="validation", .SD[which.min(mse)] ]

to.hilite <- sort(c(
  epochs.vec[c(3, length(epochs.vec)-1)],
  nnet.min.dt$epochs))
hilite.dt <- data.table(
  epochs=to.hilite,
  y=c(1,20,1),
  fit=c("underfit", "optimal", "overfit"))
hilite.colors <- structure(
  c("#1B9E77", "#D95F02", "#7570B3"),
  names=to.hilite)
gg <- ggplot()+
  scale_color_manual(values=hilite.colors)+
  geom_vline(aes(
    xintercept=epochs, color=factor(epochs)),
    size=0.7,
    data=hilite.dt)+
  geom_text(aes(
    epochs, y, label=fit, color=factor(epochs)),
    angle=90,
    size=3,
    vjust=1.2,
    data=hilite.dt)+
  geom_line(aes(
    epochs, mse, linetype=set),
    size=0.7,
    data=nnet.loss.dt)+
  scale_x_log10(breaks=10^seq(0, 3))+
  scale_y_log10("mean squared error")+
  coord_cartesian(
    xlim=nnet.loss.dt[, c(min(epochs), max(epochs)*30)])+
  theme_bw()+
  theme(
    legend.position="none",
    panel.spacing=grid::unit(0, "lines"))+
  directlabels::geom_dl(aes(
    epochs, mse, label=set),
    method=list(cex=0.8, "last.qp"),
    data=nnet.loss.dt)
png("figure-overfitting-paper-loss.png", width=2.5, height=2, units="in", res=300)
print(gg)
dev.off()
dput(RColorBrewer::brewer.pal(Inf, "Dark2"))

hilite.pred <- nnet.pred.dt[hilite.dt, on="epochs"]
hilite.resid <- nnet.resid.dt[hilite.dt, on="epochs"]
sets.gg <- ggplot()+
  theme_bw()+
  theme(
    legend.position="none",
    panel.spacing=grid::unit(0, "lines"))+
  facet_grid(set + folds ~ epochs, labeller=label_both)+ 
  geom_point(aes(
    x, y),
    shape=1,
    color="grey50",
    data=pattern.data)+
  geom_line(aes(
    x, pred.y, color=factor(epochs)),
    size=0.8,
    alpha=0.75,
    data=hilite.pred)+
  geom_segment(aes(
    x, y,
    xend=x, yend=pred.y),
    data=hilite.resid)+
  scale_color_manual(values=hilite.colors)+
  xlab("input/feature x")+
  ylab("output/label y")+
  coord_cartesian(ylim=c(-10, 20))
png("figure-overfitting-paper.png", width=5, height=3, units="in", res=300)
print(sets.gg)
dev.off()
 
