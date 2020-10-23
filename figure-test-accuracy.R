library(data.table)
library(ggplot2)
cv.results.list <- readRDS("figure-test-accuracy-data.rds")
cv.results <- do.call(rbind, cv.results.list)
cv.stats <- cv.results[, .(
  mean.accuracy=mean(acc),
  median=median(acc),
  q25=quantile(acc, 0.25),
  q75=quantile(acc, 0.75),
  sd.accuracy=sd(acc)
), by=.(model.name)][order(median)]
mfac <- function(x)factor(x, cv.stats$model.name)
cv.results[, Model := mfac(model.name)]
cv.stats[, Model := mfac(model.name)]
gg.baseline <- ggplot()+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  geom_point(aes(
    acc, Model),
    data=cv.results)+
  scale_x_continuous(
    "Accuracy rate (correctly predicted labels, four test folds)")
png("figure-test-accuracy-baseline.png",
    width=5, height=1.5, units="in", res=200)
print(gg.baseline)
dev.off()

long.panel <- c(
  folds="Zoom out to include baseline
four test folds",
stats="Zoom in to best models
median and quartiles")
pfac <- function(x)factor(long.panel[x], long.panel)
cv.results[, panel := pfac("folds")]
some.stats[, panel := pfac("stats")]
gg.both <- ggplot()+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  facet_grid(. ~ panel, scales="free")+
  geom_blank(aes(
    x, y),
    data=data.table(x=0, y=mfac("baseline"), panel=pfac("folds")))+
  geom_point(aes(
    acc, Model),
    shape=1,
    data=cv.results)+
  scale_y_discrete("Model", drop=FALSE)+
  scale_x_continuous(
    "Accuracy rate (correctly predicted labels)")+
  geom_segment(aes(
    q25, Model,
    xend=q75, yend=Model),
    data=some.stats)+
  geom_point(aes(
    median, Model),
    data=some.stats)
png("figure-test-accuracy-both.png",
    width=5, height=1.5, units="in", res=200)
print(gg.both)
dev.off()

some.stats <- cv.stats[Model != "baseline"]
gg <- ggplot()+
  theme_bw()+
  theme(panel.spacing=grid::unit(0, "lines"))+
  geom_segment(aes(
    q25, Model,
    xend=q75, yend=Model),
    data=some.stats)+
  geom_point(aes(
    median, Model),
    data=some.stats)+
  scale_x_continuous(
    "Accuracy rate (correctly predicted labels, median and quartiles)")
png("figure-test-accuracy.png", width=5, height=1.5, units="in", res=200)
print(gg)
dev.off()

