#Function Definition Starts Here
Hongye.bivariate <-
  function(Indt,                                    # Input dataset
           op_ds,					     # Output folder
           batch = T,					     # Decide the mode of the function. T is batch mode, F is interactive mode
           file_prefix = '',				     # File prefix of the output images
           drop_list = '',				     # The variables not intended to run bivariate (only valid in batch mode)
           keep_list = '',
           target_score,				     # The dependent variable
           pred_var,					     # The independent variable (only valid in interactive mode)
           dict,						     # Dictionary
           breaks,					     # User-input WOE binning breaks (only valid in interactive mode)
           cap = F, 				           # Whether cap the independent variable, default value is 99% percentile. as.logical() can recognize T, TRUE, 'T', 'True', 'TRUE', 'true', but not t, True, true, 't'
           floor = F,					     # Whether floor the independent variable
           cap_value,					     # The cap value (only valid if cap = T)
           floor_value,				     # The floor value (only valid if floor = T)
           dens = T,					     # Draw density curve
           bw_scale = 1,				     # bandwidth scale of density curve
           line = T,					     # Draw target rate fitting line
           LO_scale = F,				     # The representation of target rate. T means Log Odds scale, F means target rate scale.
           LO_cap_floor = F,				     # Whether cap/floor the Log Odds of target rate
           LO_radius = 9,				     # The cap/floor value of Log Odds of target rate (only valid if LO_cap_floor = T)
           min_nbin = 5,                            # The minimum number of bins for a numerical variable not to be treated as categorical var
           max_nbin = 30,				     # The maximum number of bins for numerical variables
           WOE_binning = 'Histogram',		     # WOE binning method
           mul_bin_factor = 1,			     # The binning factor of WOE_binning option 2
           smoothing = '',				     # WOE smoothing method
           smoothing_factor = 200,			     # smoothing factor
           eq_var_nbin = 10,				     # The bin number of WOE_binning option 3
           WOE_decrease,				     # The type of monotomy of WOE_binning option 4
           conf_level = 0.95,                       # Confidence level of error bars
           max_line_target_rate = 1,		     # The range of a target rate plot to be taken into account when drawing fitted line
           min_line_target_rate = 0,
           SAS_WOE = F,				     # Generate SAS codes of WOE
           num_column_color = 'darkseagreen',
           char_column_color = 'darkslategray',
           missing_column_color = 'indianred1',
           dens_line_color = 'firebrick1',
           target_rate_plot_color = 'blue',
           bar_color = 'darkgoldenrod1',
           target_rate_line_color = 'black')
  {
    require(car)
    
    ind_dict <-
      ind_binary <-
      ind_automono <-
      ind_autobin <- ind_autocap <- ind_autofloor <- 0
    if (missing(cap_value))
      ind_autocap    <- 1
    if (missing(floor_value))
      ind_autofloor  <- 1
    if (missing(breaks))
      ind_autobin    <- 1
    if (missing(WOE_decrease))
      ind_automono   <- 1
    if (!missing(dict))
      ind_dict       <- 1
    
    lo <- function(x) {
      log(x / (1 - x))
    }
    error_bar <- function(position, target_rate, std_error)
    {
      if (LO_scale)
        arrows(
          position[condition <-
                     !is.na(std_error) &
                     std_error > 0 &
                     target_rate - std_error > 0 &
                     target_rate + std_error < 1], lo((target_rate - std_error)[condition]), position[condition],
          lo((target_rate + std_error)[condition]), length = 0.1, angle = 90, code = 3, col = bar_color
        )
      else
        arrows(
          position[condition <-
                     !is.na(std_error)], (target_rate - std_error)[condition], position[condition], (target_rate + std_error)[condition], length = 0.1, angle = 90, code = 3, col = bar_color
        )
    }
    target_rate_line <- function(position, target_rate)
    {
      if (LO_scale)
        condition <-
          !is.na(target_rate) & target_rate > 0 & target_rate < 1
      else
        condition <- !is.na(target_rate)
      if (ind_binary &
          upper_binary == 1 &
          lower_binary == 0)
        condition <-
          condition &
          target_rate >= min_line_target_rate &
          target_rate <= max_line_target_rate
      if (length(position) > 1 &
          length((target_rate[condition])) > 1)
      {
        if (LO_scale)
          abline(lm(lo(target_rate[condition]) ~ position[condition]), lwd = 2, col = target_rate_line_color)
        else
          abline(lm(target_rate[condition] ~ position[condition]), lwd = 2, col = target_rate_line_color)
      }
    }
    
    if (as.logical(batch) || ind_autobin)
    {
      if (!WOE_binning %in% c('Histogram', 'MultiHistogram', 'EqualVariance', 'Monotonic'))
        stop ('Invalid WOE_binning')
    }
    
    numeric_bivariate <- function()
    {
      if (as.logical(cap))
      {
        if (ind_autocap)
          cap_value <-
            # quantile(selected_column, prob = .99, na.rm = T)
            quantile(selected_column, prob = .9, na.rm = T)
        capped_column <-
          ifelse(selected_column > cap_value, cap_value, selected_column)
        if (length(unique(capped_column)) >= 3)
          selected_column <- capped_column
      }
      if (as.logical(floor))
      {
        if (ind_autofloor)
          floor_value <- min(selected_column, na.rm = T)
        selected_column <-
          ifelse(selected_column < floor_value, floor_value, selected_column)
      }
      
      #Reserve space for the right y label
      par(oma = c(0, 0, 0, 2))
      n <-
        min(max_nbin, ceiling((
          length(selected_column) - missing_pop
        ) ^ (1 / 3) * (
          max(selected_column, na.rm = T) - min(selected_column, na.rm = T)
        ) / (2 * IQR(
          selected_column, na.rm = T
        ))))
      
      #Histogram
      tmp <- hist(selected_column, nclass = n, plot = FALSE)
      
      hist_end <- tail(tmp$breaks, n = 1)
      hist_width <- tmp$breaks[2] - tmp$breaks[1]
      var_range <-
        c(tmp$breaks[1], ifelse(missing_pop, hist_end + hist_width, hist_end))
      if (ind_mixed)
        var_range <-
        c(var_range[1], var_range[2] + hist_width * (2 * nChar + 1))
      
      max_pop_limit <- max(1.25 * max(tmp$counts), missing_pop)
      if (ind_mixed)
        max_pop_limit <-
        max(max_pop_limit, max(table(selected_char_column)))
      
      if (ind_mixed)
        tmp <-
        hist(
          selected_column, nclass = n, xlim = var_range, ylim = c(0, max_pop_limit), xaxt = 'n', xlab = description, freq = T, ylab = 'Population Size', col = num_column_color, main = var, las = 3
        )
      else
        tmp <-
        hist(
          selected_column, nclass = n, xlim = var_range, ylim = c(0, max_pop_limit), xlab = description, freq = T, ylab = 'Population Size', col = num_column_color, main = var, las = 3
        )
      
      #Add legend
      legend(
        x = var_range[1], y = max_pop_limit, legend = c('pop', 'target'), lty = 1, lwd = c(5, 2), col = c(num_column_color,  target_rate_plot_color)
      )
      
      #Add a bin for variable missing population
      if (missing_pop)
        rect(hist_end, 0, hist_end + hist_width, missing_pop, xlim = var_range, col = missing_column_color)
      
      #Add bins for mixed vars
      if (ind_mixed)
      {
        for (i in 1:nChar)
        {
          rect(
            hist_end + (2 * i - .5) * hist_width, 0, hist_end + (2 * i + 1) * hist_width, sum(
              selected_char_column == unique(selected_char_column)[i]
            ),
            col = ifelse(
              unique(selected_char_column)[i] == '', missing_column_color, char_column_color
            )
          )
          axis(
            1, at = c(
              seq(var_range[1], hist_end, length = 5), hist_end + (2 * i + .25) * hist_width
            ),
            labels = c(
              seq(var_range[1], hist_end, length = 5), ifelse(
                as.character(unique(selected_char_column)[i]) == '', 'missing', as.character(unique(selected_char_column)[i])
              )
            ), las = 3
          )
        }
      }
      
      #Density Estimation
      if (as.logical(dens))
      {
        a <- 4 * (var(selected_column, na.rm = T) ^ .5) ^ 5
        b <- 3 * length(selected_column)
        band_width <- (a / b) ^ (1 / 5)
        if (band_width > 0)
        {
          tmp2 <-
            density(selected_column, na.rm = T, bw = band_width * bw_scale)
          par(new = T)
          plot(
            tmp2$x, (max(tmp$counts) / max(tmp2$y)) * tmp2$y, xlim = var_range, ylim = c(0, max_pop_limit), type = 'l', col = dens_line_color, lwd = 1, xaxt = 'n', yaxt = 'n', xlab = '', ylab = ''
          )
        }
        else
        {
          stop("band_width must be positive")
        }
      }
      
      #Manually input breaks in interactive mode
      if (!as.logical(batch) & !ind_autobin)
      {
        breaks <- sort(breaks)
        breaks <-
          c(min(selected_column, na.rm = T), sort(breaks[breaks > min(selected_column, na.rm = T) &
                                                           breaks < max(selected_column, na.rm = T)]), max(selected_column, na.rm = T))
        n <- length(breaks) - 1
        counts <-
          sum(
            !is.na(selected_column) &
              selected_column >= breaks[1] &
              selected_column <= breaks[1]
          )
        for (i in 2:n)
          counts[i] <-
          sum(!is.na(selected_column) &
                selected_column > breaks[i] &
                selected_column <= breaks[i + 1])
        tmp$counts <- counts
        tmp$breaks <- breaks
        tmp$mids <- mean(c(breaks[1], breaks[2]))
        for (i in 2:n)
          tmp$mids[i] <- mean(c(breaks[i], breaks[i + 1]))
      }
      
      #Target Rates and Log Odds for Each bin
      if (as.logical(batch) || ind_autobin)
      {
        if (WOE_binning == 'Histogram')
        {
          
        }
        else if (ind_binary)
        {
          if (WOE_binning == 'MultiHistogram')
          {
            tmp$breaks <-
              seq(
                min(tmp$breaks, na.rm = T), max(tmp$breaks, na.rm = T), length = n / mul_bin_factor + 1
              )
          }
          if (WOE_binning == 'EqualVariance')
          {
            tmp$breaks <-
              unique(quantile(
                selected_column[pred_target == 1], 0:eq_var_nbin / eq_var_nbin, na.rm = T
              ))
          }
          if (WOE_binning == 'Monotonic')
          {
            if (ind_automono)
            {
              if (lm(pred_target ~ selected_column)$coefficients[2] > 0)
                isoreg_result <-
                  isoreg(selected_column[!is.na(selected_column)], pred_target[!is.na(selected_column)])
              else
                isoreg_result <-
                  isoreg(selected_column[!is.na(selected_column)],-pred_target[!is.na(selected_column)])
            }
            else
            {
              if (as.logical(WOE_decrease))
                isoreg_result <-
                  isoreg(selected_column[!is.na(selected_column)],-pred_target[!is.na(selected_column)])
              else
                isoreg_result <-
                  isoreg(selected_column[!is.na(selected_column)], pred_target[!is.na(selected_column)])
            }
            tmp$breaks <-
              sort(unique(c(
                min(selected_column, na.rm = T), knots(as.stepfun(isoreg_result)), max(selected_column, na.rm = T)
              )))
          }
          tmp$mids <- mean(c(tmp$breaks[1], tmp$breaks[2]))
          for (i in 2:length(tmp$breaks) - 1)
          {
            tmp$mids[i] <- mean(c(tmp$breaks[i], tmp$breaks[i + 1]))
          }
          tmp$counts <-
            tapply(pred_target, as.factor(cut(
              selected_column, tmp$breaks, include.lowest = T
            )), length)
        }
      }
      
      bin_target_rate <-
        tapply(pred_target, as.factor(cut(
          selected_column, tmp$breaks, include.lowest = T
        )), mean, na.rm = T)
      dev_target_rate <-
        tapply(pred_target, as.factor(cut(
          selected_column, tmp$breaks, include.lowest = T
        )), sd, na.rm = T)
      error_target_rate <-
        conf_level * sqrt(dev_target_rate / tmp$counts)
      
      if (WOE_binning %in% c('Histogram', 'MultiHistogram'))
      {
        if (smoothing == 'Population')
        {
          a <- tmp$counts / (tmp$counts + smoothing_factor)
          bin_target_rate <-
            a * bin_target_rate + (1 - a) * sum(pred_target, na.rm = T) / sum(!is.na(pred_target))
          dev_target_rate <- a * dev_target_rate
        }
        else if (smoothing == 'Poor')
        {
          smoothed_target_rate <- bin_target_rate
          if (length(bin_target_rate) >= 3)
          {
            if (!is.na(bin_target_rate[1]) &
                !is.na(bin_target_rate[1]))
              smoothed_target_rate[1] = (2 / 3) * bin_target_rate[1] + bin_target_rate[2]
            if (!is.na(bin_target_rate[length(bin_target_rate)]) &
                !is.na(bin_target_rate[length(bin_target_rate) - 1]))
              smoothed_target_rate[length(bin_target_rate)] = (2 / 3) * bin_target_rate[length(bin_target_rate)] + bin_target_rate[length(bin_target_rate) - 1]
            for (i in 2:(length(bin_target_rate) - 1))
            {
              if (!is.na(bin_target_rate[i]) &
                  !is.na(bin_target_rate[i - 1]) &
                  !is.na(bin_target_rate[i + 1]))
                smoothed_target_rate[i] = (1 / 4) * (bin_target_rate[i - 1] + bin_target_rate[i + 1]) + (1 /
                                                                                                           2) * bin_target_rate[i]
            }
          }
          bin_target_rate <- smoothed_target_rate
        }
      }
      
      LogOdds <- lo(bin_target_rate)
      if (as.logical(LO_cap_floor))
        LogOdds <-
        ifelse(LogOdds > LO_radius, LO_radius, ifelse(LogOdds < -LO_radius,-LO_radius, LogOdds))
      
      if (ind_mixed)
      {
        char_target_rate <-
          mean(pred_target_for_char[selected_char_column == unique(selected_char_column)[1]], na.rm = T)
        for (i in 2:nChar)
        {
          char_target_rate[i] <-
            mean(pred_target_for_char[selected_char_column == unique(selected_char_column)[i]], na.rm = T)
        }
        lo_char_target_rate <- lo(char_target_rate)
        if (as.logical(LO_cap_floor))
          lo_char_target_rate <-
          ifelse(
            lo_char_target_rate > LO_radius, LO_radius, ifelse(
              lo_char_target_rate < -LO_radius,-LO_radius, lo_char_target_rate
            )
          )
      }
      
      par(new = T)
      if (as.logical(LO_scale))
      {
        if (ind_mixed)
          LO_range <-
            range((
              all_LO <-
                c(LogOdds, LO_missing_target, lo_char_target_rate)
            )[is.finite(all_LO)], na.rm = T)
        else
          LO_range <-
            range((all_LO <-
                     c(LogOdds, LO_missing_target))[is.finite(all_LO)], na.rm = T)
        LO_range <-
          c(LO_range[1], LO_range[2] + .25 * (LO_range[2] - LO_range[1]))
        plot(
          tmp$mids[is.finite(LogOdds)], LogOdds[is.finite(LogOdds)], col = target_rate_plot_color, xlim = var_range, ylim = LO_range, lwd = 2, xaxt = 'n', yaxt = 'n', xlab = '', ylab = ''
        )
      }
      else
        plot(
          tmp$mids, bin_target_rate, col = target_rate_plot_color, xlim = var_range, ylim = c(
            min(c(
              bin_target_rate, missing_target_rate
            ), na.rm = T), 1.25 * max(c(
              bin_target_rate, missing_target_rate
            ), na.rm = T) - .25 * min(c(
              bin_target_rate, missing_target_rate
            ), na.rm = T)
          ), lwd = 2, xaxt = 'n', yaxt = 'n', xlab = '', ylab = ''
        )
      
      axis(4)
      if (ind_binary &
          upper_binary == 1 &
          lower_binary == 0)
        mtext(
          ifelse(LO_scale, 'Log Odds of Target Rate', 'Target Rate'), side = 4, line = 3
        )
      else
        mtext(ifelse(LO_scale, paste('Log Odds of', target_score), target_score), side = 4, line = 3)
      
      if (!WOE_binning %in% c('Histogram'))
        axis(
          3, at = tmp$breaks, labels = c('WOE binning', rep('', length(tmp$breaks) - 1)), cex = 0.1, font = 3, col = target_rate_plot_color
        )
      
      error_bar(tmp$mids, bin_target_rate, error_target_rate)
      if (as.logical(line))
        target_rate_line(tmp$mids, bin_target_rate)
      
      #Plot & Bar of Missing Bin
      if (ifelse(LO_scale, is.finite(LO_missing_target), missing_pop))
      {
        par(new = T)
        if (LO_scale)
          plot(
            mean(c(hist_end, var_range[2])), LO_missing_target, col = target_rate_plot_color, xlim = var_range, ylim = LO_range, lwd = 2, xaxt = 'n', yaxt = 'n', xlab = '', ylab = ''
          )
        #else plot(mean(c(hist_end, var_range[2])), missing_target_rate,  col = target_rate_plot_color, xlim = var_range, ylim = c(lower_binary, 1.25 * upper_binary - .25 * lower_binary), lwd = 2, xaxt = 'n', yaxt = 'n', xlab = '', ylab = '')
        else
          plot(
            mean(c(hist_end, var_range[2])), missing_target_rate,  col = target_rate_plot_color, xlim = var_range, ylim = c(
              min(c(
                bin_target_rate, missing_target_rate
              ), na.rm = T), 1.25 * max(c(
                bin_target_rate, missing_target_rate
              ), na.rm = T) - .25 * min(c(
                bin_target_rate, missing_target_rate
              ), na.rm = T)
            ), lwd = 2, xaxt = 'n', yaxt = 'n', xlab = '', ylab = ''
          )
        
        axis(1, at = mean(c(hist_end, var_range[2])), labels = 'missing', las = 3)
        error_bar(mean(c(hist_end, var_range[2])), missing_target_rate, missing_std_error)
      }
      if (ind_mixed)
      {
        for (i in 1:nChar)
        {
          par(new = T)
          if (LO_scale)
            plot(
              hist_end + (2 * i + .25) * hist_width, lo_char_target_rate[i], col = target_rate_plot_color,  xlim = var_range, ylim = LO_range, lwd = 2, xaxt = 'n', yaxt = 'n', xlab = '', ylab = ''
            )
          else
            plot(
              hist_end + (2 * i + .25) * hist_width, char_target_rate[i], col = target_rate_plot_color, xlim = var_range, ylim = c(lower_binary, 1.25 * upper_binary - .25 * lower_binary), lwd = 2, xaxt = 'n', yaxt = 'n', xlab = '', ylab = ''
            )
          error_bar(
            hist_end + (2 * i + .25) * hist_width, char_target_rate[i], conf_level * sqrt(
              sd(pred_target_for_char[selected_char_column == unique(selected_char_column)[i]], na.rm = T) / sum(
                selected_char_column == unique(selected_char_column)[i]
              )
            )
          )
        }
      }
      
      #write SAS code
      if (as.logical(SAS_WOE))
      {
        if (ind_binary & upper_binary == 1 & lower_binary == 0)
        {
          tmp$breaks <-
            c((tmp$breaks[1:length(tmp$breaks) - 1])[is.finite(bin_target_rate) &
                                                       bin_target_rate > 0 &
                                                       bin_target_rate < 1], tmp$breaks[length(tmp$breaks)])
        }
        else
          tmp$breaks <-
            c((tmp$breaks[1:length(tmp$breaks) - 1])[is.finite(bin_target_rate)], tmp$breaks[length(tmp$breaks)])
        
        bin_target_rate <-
          tapply(pred_target, as.factor(cut(
            selected_column, tmp$breaks, include.lowest = T
          )), mean, na.rm = T)
        bin_num <-
          ifelse(sum(is.na(selected_column)) > 0, length(tmp$breaks), length(tmp$breaks) -
                   1)
        
        
        if (ind_binary & upper_binary == 1 & lower_binary == 0)
        {
          lo_selected_column <- paste("LO_", var, sep = "")
          if (sum(is.na(selected_column)) > 0)
          {
            non_missing_bin_code <-
              cbind(
                c('if ', rep('else if ',bin_num - 2)), rep(var, bin_num - 1), c(' >= ', rep(' > ', bin_num - 2)), tmp$breaks[1:(bin_num -
                                                                                                                                  1)],
                " & ", rep(var, bin_num - 1), c(rep(' <= ', bin_num - 1)), tmp$breaks[2:bin_num], rep(
                  paste(" then  ", lo_selected_column, sep = ''), bin_num - 1
                ), rep(' = ', bin_num - 1), lo(bin_target_rate[1:(bin_num - 1)]), rep(' ;', bin_num - 1)
              )
            non_missing_bin_code <-
              apply(non_missing_bin_code, 1, function(c) {
                return(paste(c, collapse = ""))
              })
            non_missing_bin_code <- cbind(non_missing_bin_code)
            missing_bin_code <-
              paste(
                "else if missing (",var," ) then ", lo_selected_column, " = ", lo(missing_target_rate), ';'
              )
            sas_code <- rbind(non_missing_bin_code,missing_bin_code)
          }
          else
          {
            sas_code <-
              cbind(
                c('if ', rep('else if ',bin_num - 1)), rep(var, bin_num), c(' >= ', rep(' > ', bin_num -
                                                                                          1)), tmp$breaks[1:bin_num], rep(" & ", bin_num), rep(var, bin_num), c(rep(' <= ', bin_num)), tmp$breaks[2:(bin_num + 1)],
                rep(paste(
                  " then  ", lo_selected_column, sep = ''
                ), bin_num), rep(' = ', bin_num), lo(bin_target_rate[1:bin_num]), rep(' ;', bin_num)
              )
            sas_code <-
              apply(sas_code,1,function(c) {
                return(paste(c,collapse = ""))
              })
            sas_code <- cbind(sas_code)
          }
        }
        else
        {
          lo_selected_column <- paste("mean_", var, sep = "")
          if (sum(is.na(selected_column)) > 0)
          {
            non_missing_bin_code <-
              cbind(
                c('if ', rep('else if ',bin_num - 2)), rep(var, bin_num - 1), c(' >= ', rep(' > ', bin_num - 2)), tmp$breaks[1:(bin_num - 1)],
                " & ", rep(var, bin_num - 1), c(rep(' <= ', bin_num - 1)), tmp$breaks[2:bin_num], rep(
                  paste(" then  ", lo_selected_column, sep = ''), bin_num - 1
                ), rep(' = ', bin_num - 1), bin_target_rate[1:(bin_num - 1)], rep(' ;', bin_num - 1)
              )
            non_missing_bin_code <-
              apply(non_missing_bin_code, 1, function(c) {
                return(paste(c, collapse = ""))
              })
            non_missing_bin_code <- cbind(non_missing_bin_code)
            missing_bin_code <-
              paste(
                "else if missing (",var,") then ", lo_selected_column, " = ", missing_target_rate, ';'
              )
            sas_code <-
              rbind(non_missing_bin_code, missing_bin_code)
          }
          else
          {
            sas_code <-
              cbind(
                c('if ', rep('else if ',bin_num - 1)), rep(var, bin_num), c(' >= ', rep(' > ', bin_num -
                                                                                          1)), tmp$breaks[1:bin_num], rep(" & ", bin_num), rep(var, bin_num), c(rep(' <= ', bin_num)), tmp$breaks[2:(bin_num + 1)],
                rep(paste(
                  " then  ", lo_selected_column, sep = ''
                ), bin_num), rep(' = ', bin_num), bin_target_rate[1:bin_num], rep(' ;', bin_num)
              )
            sas_code <-
              apply(sas_code, 1, function(c) {
                return(paste(c, collapse = ""))
              })
            sas_code <- cbind(sas_code)
          }
        }
        write.csv(sas_code,paste(op_ds, var, ".csv", sep = ""),row.names = T)
      }
    }
    
    cat_bivariate <- function()
    {
      category_list <-  sort(unique(selected_column, na.rm = T))
      n <- length(category_list)
      
      category_pop <- table(selected_column)
      bin_target_rate <-
        tapply(pred_target, as.factor(selected_column), mean, na.rm = T)
      var_target_rate <-
        tapply(pred_target, as.factor(selected_column), sd, na.rm = T)
      error_target_rate <-
        conf_level * sqrt(var_target_rate / category_pop)
      
      LogOdds <- lo(bin_target_rate)
      if (as.logical(LO_cap_floor))
        LogOdds <-
        ifelse(LogOdds > LO_range, LO_range, ifelse(LogOdds < -LO_range,-LO_range, LogOdds))
      
      #Histogram
      par(oma = c(5, 0, 0, 2))
      var_range <- c(0.25, if (missing_pop)
        n + 1.75
        else
          n + .75)
      
      plot(
        0, 0, xlim = var_range, ylim = c(0, 1.25 * max(category_pop, missing_pop)), type = 'l', xaxt = 'n', xlab = description, ylab = 'Population Size', main = var
      )
      axis(1, at = c(1:n), labels = category_list, las = 3)
      
      #Add legend
      legend(
        x = var_range[1], y = 1.25 * max(category_pop, missing_pop), legend = c('pop', 'target'), lty = 1, lwd = c(5, 2), col = c(char_column_color, target_rate_plot_color)
      )
      
      # if (sum(is.finite(selected_column)) > 0)
      # {
        # for (i in 1:n)
        # {
          # rect(i - .25, 0, i + .25, category_pop[i], col = char_column_color)
        # }
      # }
      for (i in 1:n)
      {
        rect(i - .25, 0, i + .25, category_pop[i], col = char_column_color)
      }
      if (missing_pop)
      {
        rect(n + .75, 0, n + 1.25, missing_pop, col = missing_column_color)
        axis(1, at = n + 1, labels = 'missing', las = 3)
      }
      if (as.logical(LO_scale))
      {
        if (TRUE %in% is.finite(c(LogOdds, LO_missing_target)))
        {
          LO_range <-
            range((all_LO <-
                     c(LogOdds, LO_missing_target))[is.finite(all_LO)], na.rm = T)
          LO_range <-
            c(LO_range[1], LO_range[2] + .25 * (LO_range[2] - LO_range[1]))
          par(new = T)
          plot(
            LogOdds[is.finite(LogOdds)], xlim = var_range, ylim = LO_range, lwd = 2, xaxt = 'n', col = target_rate_plot_color, yaxt = 'n', xlab = '', ylab = ''
          )
          par(new = T)
          if (is.finite(LO_missing_target))
            plot(
              n + 1, LO_missing_target, xlim = var_range, ylim = LO_range, lwd = 2, xaxt = 'n', yaxt = 'n',  col = target_rate_plot_color, xlab = '', ylab = ''
            )
          if (ind_binary &
              upper_binary == 1 &
              lower_binary == 0)
            mtext('Log Odds of Target Rate', side = 4, line = 3)
          else
            mtext(paste('Log Odds of', target_score), side = 4, line = 3)
        }
      }
      else
      {
        par(new = T)
        plot(
          bin_target_rate, xlim = var_range,
          #ylim = c(lower_binary, 1.25 * upper_binary - .25 * lower_binary),
          ylim = c(
            min(c(
              bin_target_rate, missing_target_rate
            ), na.rm = T), 1.25 * max(c(
              bin_target_rate, missing_target_rate
            ), na.rm = T) - .25 * min(c(
              bin_target_rate, missing_target_rate
            ), na.rm = T)
          ),
          lwd = 2, xaxt = 'n', col = target_rate_plot_color, yaxt = 'n', xlab = '', ylab = ''
        )
        
        par(new = T)
        if (missing_pop)
          plot(
            n + 1, missing_target_rate, xlim = var_range,
            #ylim = c(lower_binary, 1.25 * upper_binary - .25 * lower_binary),
            ylim = c(
              min(c(
                bin_target_rate, missing_target_rate
              ), na.rm = T), 1.25 * max(c(
                bin_target_rate, missing_target_rate
              ), na.rm = T) - .25 * min(c(
                bin_target_rate, missing_target_rate
              ), na.rm = T)
            ),
            lwd = 2, xaxt = 'n', yaxt = 'n',  col = target_rate_plot_color, xlab = '', ylab = ''
          )
        
        if (ind_binary &
            upper_binary == 1 &
            lower_binary == 0)
          mtext('Target Rate', side = 4, line = 3)
        else
          mtext(target_score, side = 4, line = 3)
      }
      if (as.logical(line) &
          is.numeric(selected_column))
        target_rate_line(c(1:n), bin_target_rate)
      for (i in 1:n)
        error_bar(i, bin_target_rate[i], error_target_rate[i])
      error_bar(n + 1, missing_target_rate, missing_std_error)
      axis(4)
    }
    
    #start
    for (j in 1:dim(Indt)[2])
    {
      pred_target <- Indt[, target_score]
      upper_binary <- max(unique(pred_target),na.rm = T)
      lower_binary <- min(unique(pred_target),na.rm = T)
      if (length(unique(pred_target)) == 2)
        ind_binary <- 1
      
      var <- names(Indt)[j]
      cat(sprintf("---> Processing %s ...\n", var))
  	  if (sum(is.na(Indt[var])) == dim(Indt)[1])
  	  {
        cat(sprintf("     ALL NA for %s, skipped\n", var))
        next
  	  }
      description <- ''
      if (ind_dict)
      {
        for (DictIndex in 1:dim(dict)[2])
        {
          if (var == names(dict)[DictIndex])
            description <- dict[1, DictIndex]
          if (var == paste("LO_"   , names(dict)[DictIndex], sep = ""))
            description <-
              paste("Log odds of "    , dict[1, DictIndex], sep = "")
          if (var == paste("md_"   , names(dict)[DictIndex], sep = ""))
            description <-
              paste("Missing ind of " , dict[1, DictIndex], sep = "")
          if (var == paste("ind_0_", names(dict)[DictIndex], sep = ""))
            description <-
              paste("0 ind of "       , dict[1, DictIndex], sep = "")
        }
      }
      #if(ifelse(as.logical(batch), var %in% keep_list, var == pred_var))
      if (ifelse(as.logical(batch),!(var %in% drop_list),  var == pred_var))
      {
        selected_column <- Indt[, var]
        #if (length(unique(selected_column[is.character(selected_column)])) <= 30)
        #if (sum(is.finite(selected_column)) > 0)
        #{
        ind_category <- 0
        ind_mixed <- 0
        if (any(!is.na(as.numeric(as.character(
          na.omit(selected_column)
        )))) &
        any(is.na(as.numeric(as.character(
          na.omit(selected_column)
        )))))
          ind_mixed <- 1
        if (is.numeric(selected_column))
        {
          #if (is.integer(selected_column))
          {
            if (length(unique(selected_column[is.finite(selected_column)])) < max(2, min_nbin))
              ind_category <- 1
          }
        }
        else
        {
          ind_category <- 1
        }
        
        missing_pop <- sum(is.na(selected_column))
        missing_target_rate <-
          mean(pred_target[is.na(selected_column)], na.rm = T)
        missing_std_dev <-
          sd(pred_target[is.na(selected_column)], na.rm = T)
        missing_std_error <-
          conf_level * sqrt(missing_std_dev / missing_pop)
        
        LO_missing_target <-
          log(missing_target_rate / (1 - missing_target_rate), base = exp(1))
        if (as.logical(LO_cap_floor))
          LO_missing_target <-
          ifelse(
            LO_missing_target > LO_radius, LO_radius, ifelse(
              LO_missing_target < LO_radius,-LO_radius, LO_missing_target
            )
          )
        
        #Open Graphics Device
        if (as.logical(batch))
          jpeg(
            file = paste(op_ds, file_prefix, var, '.jpeg', sep = ''), width = 640, height = 480
          )
        
        #Make Bivariate Charts
        if (!ind_category)
          numeric_bivariate()
        else if (ind_mixed)
        {
          if (length(unique(selected_column[!is.na(as.numeric(as.character((
            selected_column
          ))))])) < max(2, min_nbin))
            cat_bivariate ()
          else
          {
            selected_char_column <-
              selected_column[is.na(as.numeric(as.character((
                selected_column
              ))))]
            nChar <- length(unique(selected_char_column))
            
            pred_target_for_char <-
              pred_target[is.na(as.numeric(as.character((
                selected_column
              ))))]
            pred_target <-
              pred_target[!is.na(as.numeric(as.character((
                selected_column
              ))))]
            
            selected_column <-
              selected_column[!is.na(as.numeric(as.character((
                selected_column
              ))))]
            selected_column <-
              as.numeric(as.character((selected_column)))
            missing_pop <- sum(is.na(selected_column))
            numeric_bivariate()
          }
        }
        else
          cat_bivariate ()
        
        #Close Graphic Device
        if (as.logical(batch))
          dev.off()
      }
    }
  }
