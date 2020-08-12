#%%

library(lme4)
library(lmerTest)
library(splines)
library(data.table)
library(tidyverse)
library(parallel)
library(ggplot2)

#%%

library(loomR)

sm <- snakemake
#snakemake@input[["myfile"]]

#%%

options(repr.plot.width = 10, repr.plot.height = 8)

#%%

fn_loom <- snakemake@input[["fn_lmdata"]]

#%%

chan <- snakemake@wildcards[["channel"]]

#%%

fol_out <- dirname(fn_loom)
print(fol_out)

#%%
groupfil_var <- "modelfitcondflag_v1"

measurement_name_distrim <- "dist-rim"
refname <- "Empty_nan1"
rev_transf <- function(x) { (10**x) - 0.1 }
transf <- function(x) log10(x + 0.1)
outsuffix <- "_fit.Rdata"

#%%

lfile <- connect(filename = fn_loom, mode = "r")

#%%

dir.create(fol_out)

#%%

get_measmeta_table <- function(lf) {
  rnames <- names(lf$row.attrs)
  cdat <- sapply(rnames, function(rn) {
    lf[[paste0("row_attrs/", rn)]][]
  })
  dat <- as.data.table(cdat)
  dat[, index := 1:.N]
  return(dat)

}

get_obsmeta_table <- function(lf) {
  cnames <- names(lf$col.attrs)
  cdat <- lapply(cnames, function(rn) {
    factor(lf[[paste0("col_attrs/", rn)]][])
  })
  names(cdat) <- cnames
  dat <- as.data.table(cdat)
  dat[, index := 1:.N]
  return(dat)

}


dat_meas <- get_measmeta_table(lfile)
dat_obs <- get_obsmeta_table(lfile)

#%%

prepare_dat <- function(lf, channel, groupfil_var, measurement_name_distrim, refcond, transform = transf) {
  dat_meas <- get_measmeta_table(lf)
  dat_obs <- get_obsmeta_table(lf)

  idx <- dat_meas[measurement_name == measurement_name_distrim, index]
  mat <- lfile[["matrix"]]
  dat_obs[, distrim := mat[, idx] / 500]
  idx <- dat_meas[channel_name == channel, index]
  dat_obs[, value := mat[, idx]]
  if (!is.null(transform)) {
    dat_obs[, value := transform(value)]
  }
  setnames(dat_obs, groupfil_var, "groupfil")

  dat_obs[, modelfitcond := paste0(.BY[[1]], .BY[[2]]), by = .(condition_name, groupfil)]
  dat_obs[, modelfitcond := relevel(as.factor(modelfitcond), ref = refcond)]
  #dat_obs[, condition_name := relevel((condition_name), ref="Empty_nan")]
  return(dat_obs)
}


#%%

fit_model <- function(mdat, lmerctrl = NULL) {
  #dat$FitConditionExt <- relevel(as.factor(dat$FitConditionExt), ref = "Empty_nan_ctrl")
  mod <- lmer("value~bs(distrim, df =12)+
            modelfitcond+
            plate_id+
            (1|condition_id)+
            (1|site_id)+
            (1|image_id)", mdat, REML = F,
              control = lmerctrl)
  summary(mod)
  return(mod)
}

#%%

fit_and_save <- function(lf, channel, fol_out, outsuffix = "_reform.Rdata",
                         lmerctrl = NULL) {
  fn_out <- file.path(fol_out, paste0(channel, outsuffix))
  if (file.exists(fn_out)) {
    return(1)
  }
  mdat <- prepare_dat(lf, channel, groupfil_var = groupfil_var, measurement_name_distrim = measurement_name_distrim,
                      refcond = refname)
  print("Data loaded")
  mod <- fit_model(mdat, lmerctrl = lmerctrl)
  print("Model fitted")
  save(mod, file = fn_out)
  print(paste0(channel, " finished"))
}

#%% md

# Calculate some statistics

#%%

get_model <- function(channel, fol_out, outsuffix = "_fittedall.Rdata") {
  fn_out <- file.path(fol_out, paste0(channel, outsuffix))
  load(fn_out)
  return(mod)
}

pred_dat <- function(dat, mod, refcond) {
  dat$pred_cond <- predict(mod, dat)
  dat$resid <- resid(mod)
  tcond <- dat$modelfitcond
  dat$modelfitcond <- refcond
  dat$pred_ref <- predict(mod, dat)
  dat$modelfitcond <- tcond
  return(dat)

}

calc_stats <- function(dat, refgroup = 3) {
  datfit <- data.table(dat)
  datfit[, abs_delta := rev_transf(pred_cond) - rev_transf(pred_ref)]
  datfit_stats <- datfit[, .(
    mean_abs_delta = mean(abs_delta),
    mean_counts = mean(rev_transf(value)),
    median_counts = median(rev_transf(value)),
    mean_pred_ref = mean(rev_transf(pred_ref)),
    mean_pred_cond = mean(rev_transf(pred_cond)),
    median_pred_ref = median(rev_transf(pred_ref)),
    median_pred_cond = median(rev_transf(pred_cond))
  ), by = .(condition_name, groupfil)]
  datfit_stats[, fc_counts := mean_pred_cond / mean_pred_ref]
  datfit_stats[, fc_vs_oexp := (mean_abs_delta / mean_abs_delta[groupfil == refgroup]), by = condition_name]
  return(datfit_stats)

}

save_stats <- function(fn, dat, suffix = "_absstat.csv") {
  fn_out <- file.path(fol, paste0(fn, suffix))
  data.table::fwrite(dat, fn_out)


}

get_param <- function(mod) {
  dat_coef <- as.data.frame(coef(summary(mod)))
  dat_coef["params"] <- row.names(dat_coef)
  par_conf <- as.data.frame(confint(mod, method = "Wald"))
  par_conf["params"] <- row.names(par_conf)

  dat_coef <- as.data.table(merge(dat_coef, par_conf))
  return(dat_coef)
}

parse_param <- function(dat_coef, dat_cells, groupfil_var) {
  dfil <- dat_cells[, .(modelfitcond, condition_name, groupfil)]
  dfil <- dfil[!duplicated(dfil)]
  dfil[, modelfitcond := as.character(modelfitcond)]
  dat_coef[, modelfitcond := gsub("modelfitcond", "", params)]
  dat_coef <- (merge(dat_coef, dfil, all.x = TRUE))
  setnames(dat_coef, "groupfil", groupfil_var)
  return(dat_coef)

}

get_modeldesc <- function(mod) {
  attrs <- attributes(mod)
  x <- attrs$optinfo$conv
  if (length(x$lme4) == 0) {
    x$lme4code <- 0
    x$lme4message <- ""
  } else {
    x$lme4code <- x$lme4$code
    x$lme4message <- x$lme4$message
  }
  x$lme4 <- NULL
  x$call <- toString(attrs$call)

  return(as.data.frame(x))

}

get_varcomp <- function(mod) {
  as.data.frame(VarCorr(mod))
}

#%%

run_stats <- function(lf, channel, fol_out, mod_suffix, groupfil_var) {
  dat <- prepare_dat(lf = lf, channel = channel, groupfil_var = groupfil_var, measurement_name_distrim = measurement_name_distrim,
                     refcond = refname)
  mod <- get_model(channel, fol_out = fol_out, outsuffix = mod_suffix)
  dat <- pred_dat(dat, mod, refcond = refname)
  dat_stat <- calc_stats(dat)
  setnames(dat_stat, "groupfil", groupfil_var)
  fwrite(dat_stat, file.path(fol_out, paste0(channel, mod_suffix, "_absstat.csv")))

  dat_param <- get_param(mod)
  dat_param <- parse_param(dat_param, dat, groupfil_var)
  fwrite(dat_param, file.path(fol_out, paste0(channel, mod_suffix, "_parameters.csv")))

  dat_var <- get_varcomp(mod)
  fwrite(dat_var, file.path(fol_out, paste0(channel, mod_suffix, "_randomeff.csv")))

  fwrite(dat[, .(object_id, pred_cond, pred_ref, resid, value)], file.path(fol_out, paste0(channel, mod_suffix, "_predict.csv")))
  #save_stats(fn, dat_stat)

  dat_model <- get_modeldesc(mod)
  fwrite(dat_model, file.path(fol_out, paste0(channel, mod_suffix, "_model.csv")))
}

#%%

lmerctrl <- lmerControl(optimizer = "bobyqa")

print("Start fitting")
fit_and_save(lfile, chan, fol_out, outsuffix = outsuffix, lmerctrl = lmerctrl)
print("Start stats")
run_stats(lfile, chan, fol_out, mod_suffix = outsuffix, groupfil_var = groupfil_var)
