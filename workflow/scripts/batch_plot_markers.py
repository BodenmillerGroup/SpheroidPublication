
# coding: utf-8

# In[1]:


import spherpro.bro as spb
import spherpro.datastore as spd
import spherpro.library as spl
import spherpro.configuration as conf
import spherpro.db as db
import imp
import pycytools as pct
import pycytools.library
import re
import os
import pandas as pd
import numpy as np
import spherpro.library as lib
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import spherpro.bromodules.plot_condition_images as sb_ci_module
import spherpro.bromodules.plot_image as pltimg

import src.config.config as C


fn_config = C.fn_config
re_fn = re.compile(r'[\\/*?:"<>|]')

def adapt_fig_clims(figs):
    axs = [a for f in figs for a in f.axes]
    pltimg.adapt_ax_clims(axs)


def get_valid_filename(s):
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)



def plot_hm(m, cond, bro, stack_name, measurement_name):
    sb_ci = sb_ci_module.PlotConditionImages(bro)
    target = bro.helpers.dbhelp.get_target_by_channel(m)
    target = re_fn.sub('', target)
    pts = [sb_ci.plot_hm_conditions(c, m, stack_name=stack_name, measurement_name=measurement_name, minmax=(0.,0.99)) for c in cond]
    adapt_fig_clims(pts)
    [f.savefig(os.path.join(fol_plts, get_valid_filename('_'.join(['hm',m, target, c])+'.png')), dpi=600) for f, c in zip(pts, cond)]
    [plt.close(f) for f in pts]


def plot_sqrthm(m, cond, bro, stack_name, measurement_name):
    sb_ci = sb_ci_module.PlotConditionImages(bro)
    target = bro.helpers.dbhelp.get_target_by_channel(m)
    target = re_fn.sub('', target)
    pts = [sb_ci.plot_hm_conditions(c, m, stack_name=stack_name,
                                    measurement_name=measurement_name, minmax=(0.,0.99),transf=np.sqrt) for c in cond]
    adapt_fig_clims(pts)
    [f.savefig(os.path.join(fol_plts,  get_valid_filename('_'.join(['sqrthm',m, target, c])+'.png'))) for f, c in zip(pts, cond)]
    [plt.close(f) for f in pts]

def plot_loghm(m, cond, bro, stack_name, measurement_name):
    def logtransf(x):
        return np.log10(x+0.1)
    sb_ci = sb_ci_module.PlotConditionImages(bro)
    target = bro.helpers.dbhelp.get_target_by_channel(m)
    target = re_fn.sub('', target)
    pts = [sb_ci.plot_hm_conditions(c, m, stack_name=stack_name,
                                    measurement_name=measurement_name, minmax=(0.,0.99),transf=logtransf) for c in cond]
    adapt_fig_clims(pts)
    [f.savefig(os.path.join(fol_plts,  get_valid_filename('_'.join(['loghm',m, target, c])+'.png'))) for f, c in zip(pts, cond)]
    [plt.close(f) for f in pts]

def plot_imc(m, cond, bro):
    sb_ci = sb_ci_module.PlotConditionImages(bro)
    target = bro.helpers.dbhelp.get_target_by_channel(m)
    target = re_fn.sub('', target)
    pts = [sb_ci.plot_imc_conditions(c, m, minmax=(0.,0.99)) for c in cond]
    adapt_fig_clims(pts)
    [f.savefig(os.path.join(fol_plts,  get_valid_filename('_'.join(['imc',m, target, c])+'.png'))) for f, c in zip(pts, cond)]
    [plt.close(f) for f in pts]

def plot_sqrtimc(m, cond, bro):
    sb_ci = sb_ci_module.PlotConditionImages(bro)
    target = bro.helpers.dbhelp.get_target_by_channel(m)
    target = re_fn.sub('', target)
    pts = [sb_ci.plot_imc_conditions(c, m, minmax=(0.,0.99), transf=np.sqrt) for c in cond]
    adapt_fig_clims(pts)
    [f.savefig(os.path.join(fol_plts, get_valid_filename('_'.join(['sqrtimc',m, target, c])+'.png'))) for f, c in zip(pts, cond)]
    [plt.close(f) for f in pts]


def plot_all(m):
    bro = spb.get_bro(fn_config)
    cond =  [c[0] for c in (bro.session.query(db.conditions.condition_name)
     .join(db.images)
     .join(db.valid_images).distinct())]
    plot_hm(m, cond, bro, stack_name='FullStackFiltered', measurement_name='MeanIntensityComp')
    plot_sqrthm(m, cond, bro, stack_name='FullStackFiltered', measurement_name='MeanIntensityComp')
    plot_imc(m, cond, bro)
    plot_sqrtimc(m, cond, bro)

def get_limits_cells(bro, channel, stack_name, measurement_name, object_type):
    q_meas = (bro.data.get_measmeta_query()
         .filter(db.ref_planes.channel_name == channel)
         .filter(db.stacks.stack_name.key == stack_name)
         .filter(db.measurement_names.measurement_name.key == measurement_name))




if __name__ == '__main__':
    bro = spb.get_bro(fn_config, readonly=True)
    conds = [c[0] for c in (bro.session.query(db.conditions.condition_name)
                            .join(db.images)
                            .join(db.valid_images).distinct())]
    channel = snakemake.wildcards.channel
    plot_type = snakemake.wildcards.plot_type
    if plot_type == 'hm':
        plot_hm(channel, conds, bro,
                stack_name='FullStackFiltered',
                measurement_name='MeanIntensityComp')
    elif plot_type == 'loghm':
        plot_loghm(m, cond, bro, stack_name='FullStackFiltered', measurement_name='MeanIntensityComp')
    plot_imc(m, cond, bro)
        plot_sqrtimc(m, cond, bro)
    fol_plts = C.fol_plts / 'marker_overviews'

os.makedirs(fol_plts,exist_ok=True)
