library(sf)
library(tidyverse)
library(lwgeom)
cwd=getwd()
#setwd("./Documents/PGP/Dashboard")
#ga <- st_read('GA_gerrychain_input.shp')
#ga_circle <- st_minimum_bounding_circle(ga, nQuadSegs = 50)
#reock <- sum(st_area(ga)) / sum(st_area(ga_circle))
#print(reock)


calc_Reock <- function(df_part, shp_path){
  shp <- st_read(shp_path)
  part <- unlist(df_part)
  subset_shp <- shp[part,]
  min_circle = st_minimum_bounding_circle(subset_shp, nQuadSegs = 50)
  reock <- sum(st_area(subset_shp)) / sum(st_area(min_circle))
  return(reock)
}

