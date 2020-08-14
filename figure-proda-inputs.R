library(R.matlab)
library(ggplot2)
library(data.table)

## Do we need this?
US_Loc <- R.matlab::readMat(
  "Practice session/nau_training_proda/input_data/US_Loc.mat"
)

EnvInfo <- R.matlab::readMat(
  "Practice session/nau_training_proda/input_data/EnvInfo4NN_SoilGrids.mat"
)[[1]]
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

all.finite <- function(x)apply(is.finite(x), 1, all)
some.inputs <- EnvInfo[, colnames(EnvInfo) != "R_Squared"]
keep <- all.finite(ParaMean) & all.finite(some.inputs)
data.list <- list(
  input=some.inputs[keep,],
  output=ParaMean[keep,])

gg <- ggplot()+
  with(data.list, ggtitle(sprintf(
    "%d observations/rows, %d inputs/features, %d outputs/targets",
    nrow(input), ncol(input), ncol(output))))+
  geom_point(aes(
    Lon, Lat),
    shape=1,
    data=data.table(data.list$input))+
  coord_quickmap()
png("figure-proda-inputs.png", width=6, height=4, units="in", res=100)
print(gg)
dev.off()
