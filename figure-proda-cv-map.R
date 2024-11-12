library(data.table)
library(ggplot2)
shown.outputs <- c("cryo", "maxpsi", "tau4s3", "fs2s3")
## But in the original paper, I did not used all the environmental
## variables in EnvInfo4NN_SoilGrids.mat to train the NN. Only 60
## variables were used (line 146 to 164 in nn_clm_cen.py).
var4nn <- c('IGBP', 'Climate', 'Soil_Type', 'NPPmean', 'NPPmax', 'NPPmin', 'Veg_Cover', 'BIO1', 'BIO2', 'BIO3', 'BIO4', 'BIO5', 'BIO6', 'BIO7', 'BIO8', 'BIO9', 'BIO10', 'BIO11', 'BIO12', 'BIO13', 'BIO14', 'BIO15', 'BIO16', 'BIO17', 'BIO18', 'BIO19', 'Abs_Depth_to_Bedrock', 'Bulk_Density_0cm', 'Bulk_Density_30cm', 'Bulk_Density_100cm','CEC_0cm', 'CEC_30cm', 'CEC_100cm', 'Clay_Content_0cm', 'Clay_Content_30cm', 'Clay_Content_100cm', 'Coarse_Fragments_v_0cm', 'Coarse_Fragments_v_30cm', 'Coarse_Fragments_v_100cm', 'Depth_Bedrock_R', 'Garde_Acid', 'Occurrence_R_Horizon', 'pH_Water_0cm', 'pH_Water_30cm', 'pH_Water_100cm', 'Sand_Content_0cm', 'Sand_Content_30cm', 'Sand_Content_100cm', 'Silt_Content_0cm', 'Silt_Content_30cm', 'Silt_Content_100cm', 'SWC_v_Wilting_Point_0cm', 'SWC_v_Wilting_Point_30cm', 'SWC_v_Wilting_Point_100cm', 'Texture_USDA_0cm', 'Texture_USDA_30cm', 'Texture_USDA_100cm', 'USDA_Suborder', 'WRB_Subgroup', 'Drought')
in.dt <- fread("figure-proda-cv-matlab.csv")
in.mat <- as.matrix(in.dt)
all.finite <- function(x)apply(is.finite(x), 1, all)
all.mat.list <- list(
  input=scale(in.mat[, var4nn]),
  output=in.mat[,shown.outputs])
keep <- do.call("&", lapply(all.mat.list, all.finite))
keep.mat.list <- lapply(all.mat.list, function(m)m[keep,])
keep.dt.list <- lapply(keep.mat.list, data.table)
keep.EnvInfo <- data.table(in.mat[keep,])

west.to.east <- c("West","Mid","East")
west.to.east <- c("West","East")
n.folds <- length(west.to.east)
unique.folds <- 1:n.folds
set.seed(1)
fold.list <- keep.EnvInfo[, list(
  Lon=ceiling(n.folds*rank(Lon)/.N),
  random=sample(rep(unique.folds, l=.N)))]
with(fold.list, table(Lon, random))
task.dt <- data.table(
  keep.EnvInfo,
  LonSubset=west.to.east[fold.list$Lon]
)
reg.task <- mlr3::TaskRegr$new(
  "EarthSysParam", task.dt,
  target="fs2s3")#easy
reg.task$col_roles$feature <- var4nn
same_other_sizes_cv <- mlr3resampling::ResamplingSameOtherSizesCV$new()
same_other_sizes_cv$param_set$values$folds <- 5
reg.task$col_roles$subset <- "LonSubset" 
same_other_sizes_cv$instantiate(reg.task)

show.iterations <- same_other_sizes_cv$instance$iteration.dt[
  test.fold==1]
set.colors <- c(
  train="red",
  test="black",
  ignored="grey")
##> dput(RColorBrewer::brewer.pal(Inf,"Set1"))
##c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33",
set.colors <- c(
  train="#A65628",
  test="#EEEE33",#"#F781BF",
  ignored="grey")
for(show.i in 1:nrow(show.iterations)){
  one.it <- show.iterations[show.i]
  one.task <- data.table(task.dt)[
  , set := "ignored"
  ]
  for(set.name in c('train','test')){
    i.vec <- one.it[[set.name]][[1]]
    set(one.task, i.vec, "set", set.name)
  }
  table(one.task$set)
  gg <- ggplot()+
    theme_bw()+
    theme(legend.position=c(0.9,0.2))+
    geom_point(aes(
      Lon, Lat, fill=set),
      shape=21,
      data=one.task)+
    geom_text(aes(
      -110, 25, label=sprintf(
        "test=%s train=%s fold=%s",
        test.subset, train.subsets, test.fold)),
      data=one.it)+
    scale_fill_manual(
      values=set.colors)+
    coord_quickmap()+
    scale_x_continuous(
      breaks=seq(-200, 0, by=5))
  out.png <- one.it[, sprintf(
    "figure-proda-cv-map-%s-%s.png",
    test.subset, train.subsets)]
  png(out.png, width=6, height=3.4, units="in", res=200)
  print(gg)
  dev.off()
}
