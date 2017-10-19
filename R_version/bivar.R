require(car)
require(jpeg)
library(data.table)
library(dplyr)
options(stringsAsFactors = F)

dpath <- '~/Documents/外部数据分析/cleaned/'
dat3 <- as.data.frame(fread("./银联新客测试样本6.3W.csv",encoding='UTF-8',sep = ","))
select <- read.csv('../yinlian/select_feature.csv', fileEncoding= 'gb18030')
drop = select$feature_names

# target <- as.data.frame(fread("~/Documents/operator_model/documents/origin_train_y.csv",encoding='UTF-8',sep = ",", col.names = c('is_overdue')))
# dat3 <- cbind(dat3, target)

source('~/Documents/python_project/anaconda/lib/python3.5/my_tools/R_version/Hongye.bivariate.R')
dat3[, select]


y = 'is_overdue'
# drop_name = names(dat3)[80:104]
drop_name = c('order_id', 'user_id', 'ref_time', 'overdue_day', 'order_time', 
              'mobile_phone', 'order_time_month', 'name', 'id', 'card_no', 'ym', drop, names(dat3)[grep('>', names(dat3))])                                                
 
Hongye.bivariate(
  Indt = dat3, 
  op_ds = file.path(dpath, '/total/total_'),
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

