library(ggplot2)
library(data.table)

test.dt <- data.table::fread("figure-proda-cv-data-test.csv")
multi.dt <- data.table::fread("figure-proda-cv-data-multitask-test.csv")
both.dt <- rbind(test.dt, multi.dt)

ggplot()+
  geom_point(aes(
    test.mse, model, color=fold.type),
    data=both.dt)+
  facet_wrap("out.name", scales="free")

ggplot()+
  geom_point(aes(
    test.mse, model),
    data=both.dt[out.name=="diffus"])+
  facet_grid(fold.type ~ ., labeller=label_both)

gg <- ggplot()+
  geom_point(aes(
    test.mse, model),
    data=both.dt)+
  facet_grid(fold.type ~ out.name, labeller=label_both, scales="free")+
  theme(panel.spacing.x=grid::unit(0.3, "in"))
png("figure-proda-cv-all-out.png", width=70, height=3, res=100, units="in")
print(gg)
dev.off()

some.out <- c(
  "cryo",# essentially no difference
  "maxpsi",
  "tau4s3",
  "fs2s3")
##"diffus") #better than baseline
model.map <- c(
  "baseline"="baseline",
  "NNet"="Single-task",
  "Multi-task"="Multi-task")
both.dt[, fold := factor(test.fold)]
both.dt[, Model := factor(model.map[model], model.map)]
some.test <- both.dt[some.out, on="out.name"]
some.test[, output := factor(out.name, some.out)]
gg <- ggplot()+
  geom_point(aes(
    test.mse, Model, color=fold),
    data=some.test)+
  facet_grid(fold.type ~ output, labeller=label_both, scales="free")+
  theme(panel.spacing.x=grid::unit(0.4, "in"))+
  ylab("Learned function")+
  scale_x_continuous("Mean squared error on test set", n.breaks=4)
png("figure-proda-cv-some-out.png", width=11, height=3, res=300, units="in")
print(gg)
dev.off()

test.wide <- dcast(
  test.dt, test.fold + fold.type + out.name ~ model, value.var="test.mse")

ggplot()+
  geom_point(aes(
    NNet, baseline, color=fold.type),
    data=test.wide)+
  geom_abline(color="grey")+
  facet_wrap("out.name", scales="free")
