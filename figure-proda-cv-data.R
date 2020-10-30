library(data.table)
library(ggplot2)
library(keras)

## from https://github.com/rstudio/keras/issues/937
if(FALSE){
  keras::install_keras(version = "2.1.6", tensorflow = "1.5")
}
keras::use_implementation("keras")
keras::use_backend("tensorflow")

EnvInfo.file <- "Practice session/nau_training_proda/input_data/EnvInfo4NN_SoilGrids.mat"
EnvInfo <- R.matlab::readMat(EnvInfo.file)[[1]]
colnames(EnvInfo) <- c(
  'ProfileNum', 'ProfileID', 'MaxDepth', 'LayerNum', 'Lon', 'Lat',
  'LonGrid', 'LatGrid', 'IGBP', 'Climate', 'Soil_Type',
  'NPPmean', 'NPPmax', 'NPPmin', 
  'Veg_Cover', 
  'BIO1', 'BIO2', 'BIO3', 'BIO4', 'BIO5', 'BIO6', 'BIO7',
  'BIO8', 'BIO9', 'BIO10', 'BIO11', 'BIO12', 'BIO13', 'BIO14',
  'BIO15', 'BIO16', 'BIO17', 'BIO18', 'BIO19', 
  'Abs_Depth_to_Bedrock', 
  'Bulk_Density_0cm', 'Bulk_Density_30cm', 'Bulk_Density_100cm',
  'CEC_0cm', 'CEC_30cm', 'CEC_100cm', 
  'Clay_Content_0cm', 'Clay_Content_30cm', 'Clay_Content_100cm', 
  'Coarse_Fragments_v_0cm', 'Coarse_Fragments_v_30cm',
  'Coarse_Fragments_v_100cm', 
  'Depth_Bedrock_R', 
  'Garde_Acid', 
  'Occurrence_R_Horizon', 
  'pH_Water_0cm', 'pH_Water_30cm', 'pH_Water_100cm', 
  'Sand_Content_0cm', 'Sand_Content_30cm', 'Sand_Content_100cm', 
  'Silt_Content_0cm', 'Silt_Content_30cm', 'Silt_Content_100cm', 
  'SWC_v_Wilting_Point_0cm', 'SWC_v_Wilting_Point_30cm',
  'SWC_v_Wilting_Point_100cm', 
  'Texture_USDA_0cm', 'Texture_USDA_30cm', 'Texture_USDA_100cm', 
  'USDA_Suborder', 
  'WRB_Subgroup', 
  'Drought', 
  'R_Squared')
ParaMean <- R.matlab::readMat(
  "Practice session/nau_training_proda/input_data/ParaMean_V8.4.mat"
)[[1]][, -(1:2)]
colnames(ParaMean) <- c(
  'diffus', 'cryo', 'q10', 'efolding', 'tau4cwd', 'tau4l1', 'tau4l2l3',
  'tau4s1', 'tau4s2', 'tau4s3', 'fl1s1', 'fl2s1', 'fl3s2', 'fs1s2', 'fs1s3',
  'fs2s1', 'fs2s3', 'fs3s1', 'fcwdl2', 'ins', 'beta', 'p4ll', 'p4ml',
  'p4cl', 'maxpsi')

## But in the original paper, I did not used all the environmental
## variables in EnvInfo4NN_SoilGrids.mat to train the NN. Only 60
## variables were used (line 146 to 164 in nn_clm_cen.py).
var4nn <- c('IGBP', 'Climate', 'Soil_Type', 'NPPmean', 'NPPmax', 'NPPmin', 'Veg_Cover', 'BIO1', 'BIO2', 'BIO3', 'BIO4', 'BIO5', 'BIO6', 'BIO7', 'BIO8', 'BIO9', 'BIO10', 'BIO11', 'BIO12', 'BIO13', 'BIO14', 'BIO15', 'BIO16', 'BIO17', 'BIO18', 'BIO19', 'Abs_Depth_to_Bedrock', 'Bulk_Density_0cm', 'Bulk_Density_30cm', 'Bulk_Density_100cm','CEC_0cm', 'CEC_30cm', 'CEC_100cm', 'Clay_Content_0cm', 'Clay_Content_30cm', 'Clay_Content_100cm', 'Coarse_Fragments_v_0cm', 'Coarse_Fragments_v_30cm', 'Coarse_Fragments_v_100cm', 'Depth_Bedrock_R', 'Garde_Acid', 'Occurrence_R_Horizon', 'pH_Water_0cm', 'pH_Water_30cm', 'pH_Water_100cm', 'Sand_Content_0cm', 'Sand_Content_30cm', 'Sand_Content_100cm', 'Silt_Content_0cm', 'Silt_Content_30cm', 'Silt_Content_100cm', 'SWC_v_Wilting_Point_0cm', 'SWC_v_Wilting_Point_30cm', 'SWC_v_Wilting_Point_100cm', 'Texture_USDA_0cm', 'Texture_USDA_30cm', 'Texture_USDA_100cm', 'USDA_Suborder', 'WRB_Subgroup', 'Drought')
length(var4nn)

all.finite <- function(x)apply(is.finite(x), 1, all)
all.mat.list <- list(
  input=scale(EnvInfo[, var4nn]),
  output=ParaMean)
keep <- do.call("&", lapply(all.mat.list, all.finite))
keep.mat.list <- lapply(all.mat.list, function(m)m[keep,])
keep.dt.list <- lapply(keep.mat.list, data.table)
keep.EnvInfo <- data.table(EnvInfo[keep,])
sapply(keep.mat.list, function(m)mean(is.finite(m)))

n.folds <- 5
unique.folds <- 1:n.folds
set.seed(1)
fold.list <- keep.EnvInfo[, list(
  Lon=ceiling(n.folds*rank(Lon)/.N),
  random=sample(rep(unique.folds, l=.N)))]
with(fold.list, table(Lon, random))

fold.dt.list <- list()
for(fold.type in names(fold.list)){
  fold.vec <- fold.list[[fold.type]]
  print(fold.type)
  print(table(fold.vec))
  fold.col <- paste0(fold.type, ".fold")
  fold.dt.list[[fold.type]] <- keep.EnvInfo[, data.table(
    Lat, Lon, fold.type, fold=factor(fold.vec))]
}
fold.dt <- do.call(rbind, fold.dt.list)

ggplot()+
  facet_grid(fold.type ~ ., labeller=label_both)+
  geom_point(aes(
    Lon, Lat, color=fold),
    shape=1,
    data=fold.dt)+
  coord_quickmap()

gg <- ggplot()+
  facet_grid(. ~ fold.type, labeller=label_both)+
  geom_point(aes(
    Lon, Lat, color=fold),
    shape=1,
    data=fold.dt)+
  coord_quickmap()
png("figure-proda-cv-data-map.png", width=10, height=3, units="in", res=200)
print(gg)
dev.off()

keep.output.tall <- melt(
  keep.dt.list[["output"]],
  measure.vars=names(keep.dt.list[["output"]]))

ggplot()+
  geom_histogram(aes(
    value),
    data=keep.output.tall)+
  facet_wrap("variable", scales="free")

## For your questions: 1. US_Loc.mat was used to identify which soil
## profiles are located in the US. Because the whole dataset is a
## global one and I am using it to do another research, I made soil
## profiles outside the US as NaN. You mentioned you excluded all the
## NaN data. Your figure should be the same when you are using the
## US_Loc.mat to indicate soil profiles located in the US.

## 2. The R_squared information is in the last column of EnvInfo.mat
## instead of EnvInfo4NN_SoilGrids.mat. This issue arose from the
## legacy of data updates. I forgot to transfer all the R_squared
## information to EnvInfo4NN_SoilGrids.mat. At line 168 of the code
## nn_clm_cen.py, you can find I used EnvInfo.mat to extract R_squared
## information.
arg.list <- expand.grid(
  test.fold=unique.folds,
  fold.type=names(fold.list),
  out.name=colnames(keep.mat.list$output),
  stringsAsFactors = FALSE)

TrainOne <- function(test.fold, fold.type, out.name, fold.list, keep.mat.list){
  library(data.table)
  library(keras)
  exp.mat.list <- with(keep.mat.list, list(
    input=input,
    output=output[, out.name, drop=FALSE]))
  test.i <- which(apply(fold.list == test.fold, 1, all))
  fold.vec <- fold.list[[fold.type]]
  set.select.list <- list(
    test=test.i,
    train=fold.vec != test.fold)
  set.data.list <- list()
  for(data.type in names(exp.mat.list)){
    mat <- exp.mat.list[[data.type]]
    for(set.name in names(set.select.list)){
      select.vec <- set.select.list[[set.name]]
      set.data.list[[set.name]][[data.type]] <- mat[select.vec, ]
    }
  }
  mean.train.labels <- mean(set.data.list$train$output)
  compile.model <- function(){
    keras_model_sequential() %>%
      layer_dense(
        units = 256, activation = 'relu',
        input_shape = ncol(keep.mat.list[["input"]])) %>%
      layer_dropout(0.3) %>%
      layer_dense(units = 512, activation = 'relu') %>%
      layer_dropout(0.5) %>%
      layer_dense(units = 512, activation = 'relu') %>%
      layer_dropout(0.5) %>%
      layer_dense(units = 256, activation = 'relu') %>%
      layer_dropout(0.3) %>%
      layer_dense(units = 1) %>%
      compile(
        loss = loss_mean_squared_error,
        optimizer = optimizer_adadelta()
      )
  }
  fit.metrics <- function(model, epochs, validation_split){
    model %>% fit(
      set.data.list$train$input,
      set.data.list$train$output,
      batch_size = 64,
      epochs = epochs,
      validation_split = validation_split,
      verbose=0
    )
  }
  valid.model <- compile.model()
  valid.metrics <- fit.metrics(
    valid.model,
    4800,#value from paper: 4800
    0.2)
  best.epochs <- which.min(valid.metrics$metrics$val_loss)
  metrics.dt <- do.call(data.table::data.table, valid.metrics$metrics)
  metrics.dt[, epoch := 1:.N]
  refit.model <- compile.model()
  refit.metrics <- fit.metrics(refit.model, best.epochs, 0)
  ## compute test accuracy.
  test.mse <- with(set.data.list$test, c(
    NNet=as.numeric(evaluate(refit.model, input, output)),
    baseline=mean( (output - mean.train.labels)^2 )))
  meta.dt <- data.table(test.fold, fold.type, out.name)
  test.dt <- data.table(
    meta.dt,
    model=names(test.mse),
    test.mse,
    test.N=length(set.data.list$test$output))
  list(test=test.dt, train=data.table(meta.dt, metrics.dt))
}

reg.name <- "registry-norm-4800"
reg.name <- "registry"
reg.name <- "registry-norm-100"

unlink(reg.name, recursive = TRUE)
batchtools::makeRegistry(reg.name)
batchtools::batchMap(TrainOne, args=arg.list, more.args=list(fold.list=fold.list, keep.mat.list=keep.mat.list))

batchtools::testJob(1)

job.table <- batchtools::getJobTable()
chunks <- data.frame(job.table, chunk=1)
batchtools::submitJobs(chunks, resources=list(
  walltime = 24*60*60,#seconds
  memory = 2000,#megabytes per cpu
  ncpus=1,  #>1 for multicore/parallel jobs.
  ntasks=1, #>1 for MPI jobs.
  chunks.as.arrayjobs=TRUE))

batchtools::loadRegistry(reg.name)

batchtools::getStatus()

jt <- batchtools::getJobTable()
jt[!is.na(error)]

result.dt.list <- list()
for(job.i in 1:nrow(jt)){
  res.list <- batchtools::loadResult(job.i)
  for(tab.name in names(res.list)){
    result.dt.list[[tab.name]][[job.i]] <- res.list[[tab.name]]
  }
}

result.dt.list <- list()
result.rds.vec <- Sys.glob("registry/results/*")
for(job.i in seq_along(result.rds.vec)){
  res.list <- readRDS(result.rds.vec[[job.i]])
  for(tab.name in names(res.list)){
    result.dt.list[[tab.name]][[job.i]] <- res.list[[tab.name]]
  }
}

for(tab.name in names(result.dt.list)){
  result.dt <- do.call(rbind, result.dt.list[[tab.name]])
  out.csv <- paste0("figure-proda-cv-data-", tab.name, ".csv")
  data.table::fwrite(result.dt, out.csv)
}
best.dt <- result.dt[, .(
  best.epochs=which.min(val_loss)
), by=.(test.fold, fold.type, out.name)]
best.dt[max(best.epochs)==best.epochs]
