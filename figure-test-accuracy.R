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
cv.results[, Model := factor(model.name, cv.stats$model.name)]
cv.stats[, Model := factor(model.name, cv.stats$model.name)]
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
    "Accuracy rate (correctly predicted labels, mean +/- sd)")
png("figure-test-accuracy.png", width=5, height=1.5, units="in", res=200)
print(gg)
dev.off()

