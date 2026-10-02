library(data.table)
library(ggplot2)
shown.outputs <- c("cryo", "maxpsi", "tau4s3", "fs2s3")
## But in the original paper, I did not used all the environmental
## variables in EnvInfo4NN_SoilGrids.mat to train the NN. Only 60
## variables were used (line 146 to 164 in nn_clm_cen.py).
var4nn <- c('IGBP', 'Climate', 'Soil_Type', 'NPPmean', 'NPPmax', 'NPPmin', 'Veg_Cover', 'BIO1', 'BIO2', 'BIO3', 'BIO4', 'BIO5', 'BIO6', 'BIO7', 'BIO8', 'BIO9', 'BIO10', 'BIO11', 'BIO12', 'BIO13', 'BIO14', 'BIO15', 'BIO16', 'BIO17', 'BIO18', 'BIO19', 'Abs_Depth_to_Bedrock', 'Bulk_Density_0cm', 'Bulk_Density_30cm', 'Bulk_Density_100cm','CEC_0cm', 'CEC_30cm', 'CEC_100cm', 'Clay_Content_0cm', 'Clay_Content_30cm', 'Clay_Content_100cm', 'Coarse_Fragments_v_0cm', 'Coarse_Fragments_v_30cm', 'Coarse_Fragments_v_100cm', 'Depth_Bedrock_R', 'Garde_Acid', 'Occurrence_R_Horizon', 'pH_Water_0cm', 'pH_Water_30cm', 'pH_Water_100cm', 'Sand_Content_0cm', 'Sand_Content_30cm', 'Sand_Content_100cm', 'Silt_Content_0cm', 'Silt_Content_30cm', 'Silt_Content_100cm', 'SWC_v_Wilting_Point_0cm', 'SWC_v_Wilting_Point_30cm', 'SWC_v_Wilting_Point_100cm', 'Texture_USDA_0cm', 'Texture_USDA_30cm', 'Texture_USDA_100cm', 'USDA_Suborder', 'WRB_Subgroup', 'Drought')
in.dt <- fread("figure-proda-cv-matlab.csv")
myround <- function(x)round(x/2)
(keep.EnvInfo <- in.dt[
  #seq(1, .N, by=10)
][
, ll := paste(myround(Lat), myround(Lon))
][
, .SD[1], by=ll
][is.finite(Lon)])

west.to.east <- c("West","Mid","East")
west.to.east <- c("West","East")
n.folds <- length(west.to.east)
unique.folds <- 1:n.folds
set.seed(1)
Lon.thresh <- -110
thresh.dt <- data.table(Lon=Lon.thresh)
fold.list <- keep.EnvInfo[, list(
  Block=ifelse(Lon < Lon.thresh, 1, 2),
  Standard=sample(rep(unique.folds, l=.N)))]
table(fold.list$Block)

std.dt.list <- list()
for(cv in names(fold.list)){
  fold.dt <- data.table(keep.EnvInfo[, .(Lat, Lon)], fold=fold.list[[cv]])
  for(test.fold in unique.folds){
    std.dt.list[[paste(cv, test.fold)]] <- data.table(
      CV=factor(cv, c("Standard", "Block")), test.fold,
      fold.dt[, set := ifelse(fold==test.fold, "test", "train")][]
    )
  }
}
(std.dt <- rbindlist(std.dt.list))

set.colors <- c(
  train="#ef686d",
  test="#53ccff",
  ignored="white")
gg <- ggplot()+
  theme_bw()+
  theme(
    legend.position="none",
    panel.spacing=grid::unit(0,"lines"))+
  ##ggtitle("Previous: standard and blocked train/splits")+
  ## geom_vline(aes(
  ##   xintercept=Lon),
  ##   data=thresh.dt)+
  geom_point(aes(
    Lon, Lat, fill=set),
    shape=21,
    color="grey",
    data=std.dt)+
  scale_fill_manual(
    values=set.colors)+
  coord_quickmap()+
  scale_x_continuous(
    "",
    breaks=NULL)+
  scale_y_continuous(
    "",
    breaks=NULL)+
  facet_grid(CV ~ test.fold, labeller=label_both)
png("figure-proda-cv-map-imbalanced-std.png", width=3.8, height=2.3, units="in", res=200)
print(gg)
dev.off()

task.dt <- data.table(
  keep.EnvInfo,
  LonSubset=west.to.east[fold.list$Block]
)
reg.task <- mlr3::TaskRegr$new(
  "EarthSysParam", task.dt,
  target="fs2s3")#easy
reg.task$col_roles$feature <- var4nn
same_other_sizes_cv <- mlr3resampling::ResamplingSameOtherSizesCV$new()
same_other_sizes_cv$param_set$values$folds <- 2
same_other_sizes_cv$param_set$values$sizes <- 0
same_other_sizes_cv$param_set$values$subsets <- "SO"
reg.task$col_roles$subset <- "LonSubset" 
same_other_sizes_cv$instantiate(reg.task)

show.iterations <- same_other_sizes_cv$instance$iteration.dt

out.dt.list <- list()
for(show.i in 1:nrow(show.iterations)){
  one.it <- show.iterations[show.i]
  one.task <- data.table(task.dt)[
  , set := "ignored"
  ]
  for(set.name in c('train','test')){
    i.vec <- one.it[[set.name]][[1]]
    set(one.task, i.vec, "set", set.name)
  }
  out.dt.list[[show.i]] <- data.table(
    one.it[, .(test.subset, train.subsets, test.fold, groups, n.train.groups)],
    one.task[, .(Lat, Lon, set)]
  )
}
other.D <- "other\ndownsample"
(out.dt <- rbindlist(out.dt.list)[test.subset=="West"][, Train := factor(ifelse(n.train.groups<groups, other.D, train.subsets), c("other","same",other.D))][])
table(out.dt$Train)

gg <- ggplot()+
  theme_bw()+
  theme(
    legend.position="top",
    panel.spacing=grid::unit(0,"lines"))+
  ##ggtitle("Proposed SOKED splits for test=West")+
  ## geom_vline(aes(
  ##   xintercept=Lon),
  ##   data=thresh.dt)+
  geom_point(aes(
    Lon, Lat, fill=set),
    shape=21,
    color="grey",
    data=out.dt)+
  scale_fill_manual(
    values=set.colors)+
  coord_quickmap()+
  scale_x_continuous(
    "",
    breaks=NULL)+
  scale_y_continuous(
    "",
    breaks=NULL)+
  facet_grid(Train ~ test.fold, labeller=label_both)
png("figure-proda-cv-map-imbalanced.png", width=3.8, height=3.7, units="in", res=200)
print(gg)
dev.off()
