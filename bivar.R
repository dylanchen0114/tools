require(car)
require(jpeg)
library(data.table)
library(dplyr)
options(stringsAsFactors = F)

dpath <- '~/Documents/youmeng/'
dat3 <- as.data.frame(fread("~/Documents/youmeng/clean/train_cleaned.csv",encoding='UTF-8',sep = ","))
code <- read.csv('~/Documents/youmeng/doc/行政区代码.csv', fileEncoding = 'gb18030', header = F, col.names = c('code', 'city'))
names(code)[1] = 'city_freq'
dat3 = left_join(dat3, code, by = c('city_freq'))

dat3 = dat3[, c('city', 'is_15+')]

dat3[dat3==''] = NA


dat3 = as.data.table(dat3)
temp = dat3[, .(count = .N, mean = mean(`is_15+`)), by = .(device_brand)][order(-count)]

source('~/Documents/python_project/anaconda/lib/python3.5/my_tools/Hongye.bivariate.R')


y = 'is_15+'
drop_name = names(dat3)[80:104]


Hongye.bivariate(
  Indt = dat3, 
  op_ds = file.path(dpath, 'bivar', 'bv_'),
  batch = T,
  drop_list = drop_name,
  target_score = y,
  cap_value = 8000,
  min_nbin = 1, # normal 2, changed for avoiding numeric_bivar in Line 797, with -1, -1.0!
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

