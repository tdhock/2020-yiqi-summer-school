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
  for(label in unique(y)){
    obs.i.vec <- which(y == label)[1:obs.per.label]
    for(observation in seq_along(obs.i.vec)){
      obs.i <- obs.i.vec[[observation]]
      intensity.mat <- test.list[["x"]][obs.i,,]
      some.intensity.dt.list[[paste(label, observation)]] <- data.table(
        label, observation,
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
      width=7, height=5, res=100, units="in")
  print(gg)
  dev.off()
}
