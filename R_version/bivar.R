require(car)
require(jpeg)
library(data.table)
library(dplyr)
options(stringsAsFactors = F)

dpath <- '~/Documents/operator_model/'
dat3 <- as.data.frame(fread("~/Documents/operator_model/data_for_bivar.csv",encoding='UTF-8',sep = ","))

source('~/Documents/python_project/anaconda/lib/python3.5/my_tools/Hongye.bivariate.R')


y = 'is_overdue'
# drop_name = names(dat3)[80:104]
drop_name = c('overdue_day', 'is_overdue', 'ref_time')

Hongye.bivariate(
  Indt = dat3, 
  op_ds = file.path(dpath, 'bivar_v2', 'select_'),
  batch = T,
  drop_list = drop_name,
  target_score = y,
  cap_value = 8000,
  min_nbin = 2, # normal 2, changed for avoiding numeric_bivar in Line 797, with -1, -1.0!
  max_nbin = 1000,
  dens = F,
  WOE_binning ='Histogram',
  line = F,
  SAS_WOE = F,
  mul_bin_factor = 3,
  WOE_decrease = T,
  cap = T,
  floor = F,
  max_line_target_rate = 1,
  min_line_target_rate = 0,
  LO_scale = F,
  LO_cap_floor = F,
  target_rate_plot_color = 'blue'
)

