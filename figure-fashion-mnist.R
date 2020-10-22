library(ggplot2)
library(data.table)
## from https://github.com/rstudio/keras/issues/937
if(FALSE){
  keras::install_keras(version = "2.1.6", tensorflow = "1.5")
}
keras::use_implementation("keras")
keras::use_backend("tensorflow")

if(file.exists("figure-fashion-mnist-data.rds")){
  data.list <- readRDS("figure-fashion-mnist-data.rds")
}else{
  data.list <- list(
    fashion=keras::dataset_fashion_mnist(),
    digits=keras::dataset_mnist())
  saveRDS(data.list, "figure-fashion-mnist-data.rds")
}

obs.per.label <- 7
breaks.vec <- c(10, 20, 28)
for(data.name in names(data.list)){
  test.list <- data.list[[data.name]][["test"]]
  y <- test.list[["y"]]
  some.intensity.dt.list <- list()
  design.row <- 0
  for(label in unique(y)){
    obs.i.vec <- which(y == label)[1:obs.per.label]
    for(observation in seq_along(obs.i.vec)){
      obs.i <- obs.i.vec[[observation]]
      intensity.mat <- test.list[["x"]][obs.i,,]
      colnames(intensity.mat) <- rownames(intensity.mat) <- 1:nrow(intensity.mat)
      design.row <- design.row+1
      some.intensity.dt.list[[paste(label, observation)]] <- data.table(
        label, observation,
        design.row,
        design.col=1:length(intensity.mat),
        row=as.integer(row(intensity.mat)),
        col=as.integer(col(intensity.mat)),
        intensity=as.numeric(intensity.mat))
    }
  }
  some.intensity.dt <- do.call(rbind, some.intensity.dt.list)
  some.intensity.dt[, ex := observation]
  gg <- ggplot()+
    theme(
      panel.border=element_rect(fill=NA, color="white", size=0.5),
      panel.spacing=grid::unit(0, "lines"))+
    facet_grid(ex ~ label, labeller=label_both)+
    coord_equal(expand=FALSE)+
    geom_tile(aes(
      col, row, fill=intensity),
      data=some.intensity.dt)+
    scale_fill_gradient(low="black", high="white")+
    scale_y_reverse("Pixel row index", breaks=breaks.vec)+
    scale_x_continuous("Pixel column index", breaks=breaks.vec)
  png(paste0("figure-fashion-mnist-", data.name, ".png"),
      width=7, height=4.5, res=100, units="in")
  print(gg)
  dev.off()
  gg <- ggplot()+
    theme(
      panel.border=element_rect(fill=NA, color="white", size=0.5),
      panel.spacing=grid::unit(0, "lines"))+
    coord_cartesian(expand=FALSE)+
    geom_tile(aes(
      design.col, design.row, fill=intensity),
      data=some.intensity.dt)+
    scale_fill_gradient(low="black", high="white")+
    scale_y_reverse("Example/row")+
    scale_x_continuous(
      "Pixel/feature/column", breaks=c(1, seq(100, 700, by=100), nrow(last.image.dt)))
  gg <- ggplot()+
    facet_grid(label ~ .)+
    theme(
      panel.border=element_rect(fill=NA, color="white", size=0.5),
      panel.spacing=grid::unit(0, "lines"))+
    coord_cartesian(expand=FALSE)+
    geom_tile(aes(
      design.col, observation, fill=intensity),
      data=some.intensity.dt)+
    scale_fill_gradient(low="black", high="white")+
    scale_y_reverse("Example/row")+
    scale_x_continuous(
      "Pixel/feature/column", breaks=c(1, seq(100, 700, by=100), nrow(last.image.dt)))
  png(paste0("figure-fashion-mnist-", data.name, "-design.png"),
      width=7, height=2, res=100, units="in")
  print(gg)
  dev.off()
}

one.breaks <- c(1, breaks.vec)
last.image.dt <- some.intensity.dt.list[[length(some.intensity.dt.list)]]
gg <- ggplot()+
  theme(
    panel.border=element_rect(fill=NA, color="white", size=0.5),
    panel.spacing=grid::unit(0, "lines"))+
  coord_equal(expand=FALSE)+
  geom_tile(aes(
    col, row, fill=intensity),
    data=last.image.dt)+
  scale_fill_gradient(low="black", high="white")+
  scale_y_reverse("Pixel row index", breaks=one.breaks)+
  scale_x_continuous("Pixel column index", breaks=one.breaks)
png("figure-fashion-mnist-one-example.png",
    width=3, height=2.3, res=100, units="in")
print(gg)
dev.off()

gg <- ggplot()+
  theme(
    panel.border=element_rect(fill=NA, color="white", size=0.5),
    panel.spacing=grid::unit(0, "lines"))+
  ##coord_equal(expand=FALSE)+
  geom_tile(aes(
    design.col, 1, fill=intensity),
    data=last.image.dt)+
  scale_fill_gradient(low="black", high="white")+
  scale_x_continuous(
    "Pixel index", breaks=c(1, seq(100, 700, by=100), nrow(last.image.dt)))
print(gg)

