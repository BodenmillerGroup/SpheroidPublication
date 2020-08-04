import collections
import functools
from itertools import chain, combinations

import numpy as np
import pandas as pd
import re
import spherpro.bro as spb
import spherpro.bromodules.helpers_vz as helpers_vz
import spherpro.db as db
import sqlalchemy as sa
import statsmodels.formula.api as smf


class CurVariableHelper(helpers_vz.VariableBaseHelper):
    COL_D2RIM = 'DistRim'
    COL_GENE = 'gene'
    COL_GENE_UNTAGGED = 'gene_untagged'
    COL_TAG = 'tag'
    COL_DOXO = 'doxocycline'
    COL_DILUTION = 'dilution'
    COL_GOODNAME = 'goodname'
    COL_ISNB = 'isnb'
    COL_FITTED = 'fitted'
    COL_RESID = 'residual'
    COL_IMGID = db.images.image_id.key
    COL_OBJ_NR = db.objects.object_number.key
    COL_FC = 'fc'
    COL_P = 'p'
    COL_DF = 'DF'
    COL_DELTA = 'delta'
    COL_TSTAT = 't'
    COL_TAGSTAT = 'TagStat'
    COL_POSSTAT = 'PosStat'
    COL_WORKING = 'working'
    COL_N = 'n'
    COL_N_OVEREXPR = 'n_overexpr'
    COL_FC_CENS = 'fc_cens'
    COL_NB = 'nb'
    COL_P_CORR = 'p_corrected'
    COL_ISSIG = 'is_sig_sel'
    COL_FITCONDITIONNAME = 'FitConditionName'
    COL_VALUES = db.object_measurements.value.key
    COL_COEFNAME = 'coefname'
    COL_METAL = 'metal'
    COL_MARKER_CLASS = 'marker_class'
    COL_IS_CC = 'is_cc'
    COL_MEASTYPE = db.measurement_types.measurement_type.key
    COL_MEASID = db.measurements.measurement_id.key
    COL_VALUE = db.object_measurements.value.key
    MEAS_INTENSITY = 'Intensity'
    COL_MODTYPE = 'modtype'
    COL_MODEL = 'model'
    COL_MODELCLASS = 'modelclass'
    COL_PARAMS = 'params'
    COL_VAR = 'variable'

V = CurVariableHelper


def get_imgs_for_cond(bro, condition_name):
    q = (bro.session.query(db.images.image_id)
         .join(db.conditions)
         .join(db.valid_images)
         .filter(db.conditions.condition_name.like(f'{condition_name}%')))
    return [i[0] for i in q.all()]


def get_condids_for_cond(bro, condition_name):
    q = (bro.session.query(db.conditions.condition_id)
         .join(db.images)
         .join(db.valid_images)
         .filter(db.conditions.condition_name.like(f'{condition_name}%')))
    return [i[0] for i in q.all()]


# I added consoring as there were hugh outliers in the data
def censor_dat(x, q=99.9):
    x = np.copy(x)
    p = np.percentile(x, q=q)
    x[x > p] = p
    return x


def cur_logtransf(x):
    return np.log10(x + 0.1)


def cur_transf(x):
    x = censor_dat(x)
    x = cur_logtransf(x)
    return x


def transf_intensities(dat, dat_measmeta):
    ids = dat_measmeta.loc[dat_measmeta[V.COL_MEASTYPE] == V.MEAS_INTENSITY, V.COL_MEASID]
    fil = dat[V.COL_MEASID].isin(ids)
    dat = dat.copy()
    dat.loc[fil, V.COL_VALUE] = dat.loc[fil, :].groupby(V.COL_MEASID).transform(cur_transf)
    return dat


class Renamer(object):
    def __init__(self):
        self.d = dict()

    def rename(self, x):
        rx = 'c' + str(x)
        self.d.update({rx: x})
        return rx

    def unrename(self, x):
        return self.d[x]


def prepare_lmdat(data, dat_measmeta, dat_imgmeta, meas_names=None, meas_ids=None):
    if meas_ids is None:
        meas_ids = []
    if meas_names is None:
        meas_names = []
    fil = (
            ((dat_measmeta[V.COL_MEASNAME].isin(meas_names))
            ) | (dat_measmeta[V.COL_MEASID].isin(meas_ids)))
    meas = dat_measmeta.loc[fil, V.COL_MEASID]
    # reshape the data
    tdat = (data.loc[data[V.COL_MEASID].isin(meas)]
            .pivot_table(index=[V.COL_OBJID, V.COL_IMID], columns=V.COL_MEASID,
                         values=V.COL_VALUES).reset_index())
    tdat = tdat.dropna()
    tdat = tdat.merge(dat_imgmeta.loc[:, [V.COL_IMGID, V.COL_IMGLEVEL, V.COL_SITELEVEL, V.COL_CONDLEVEL]])
    return tdat




# creats all possible combinations:


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)  # allows duplicate elements
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


Lmres = collections.namedtuple('lmres', ['summary', 'rsquared', 'params'])

def calc_lm(mark, lm_data, dat_measmeta):
    POLYLEVEL = 6

    meas_dat = set(lm_data.columns)
    # retrieve various lists of channels
    col_curmark = dat_measmeta.loc[
        (dat_measmeta[V.COL_MEASNAME] == 'MeanIntensityComp')
        & (dat_measmeta[V.COL_CHANNELNAME] == mark)
        , V.COL_MEASID].iloc[0]
    col_meas = list(set(dat_measmeta.loc[dat_measmeta[V.COL_MEASTYPE] == 'Intensity', V.COL_MEASID]) & meas_dat)
    col_dist = list(set(dat_measmeta.loc[dat_measmeta[V.COL_MEASNAME] == 'dist-rim', V.COL_MEASID]) & meas_dat)
    col_nb = list(
        set(dat_measmeta.loc[dat_measmeta[V.COL_MEASNAME] == 'NbMeanMeanIntensityComp', V.COL_MEASID]) & meas_dat)

    col_int = list(set(dat_measmeta.loc[dat_measmeta[V.COL_MEASNAME] == 'MeanIntensityComp', V.COL_MEASID]) & meas_dat)
    col_cc = list(set(dat_measmeta.loc[(dat_measmeta[V.COL_MEASNAME] == 'MeanIntensityComp') &
                                       (dat_measmeta[V.COL_IS_CC] == True), V.COL_MEASID]) & meas_dat)
    col_nbcc = list(set(dat_measmeta.loc[(dat_measmeta[V.COL_MEASNAME] == 'NbMeanMeanIntensityComp') &
                                         (dat_measmeta[V.COL_IS_CC] == True), V.COL_MEASID]) & meas_dat)
    col_curmark_meas = list(set(dat_measmeta.loc[dat_measmeta[V.COL_CHANNELNAME] == mark, V.COL_MEASID]) & meas_dat)

    # reshape the data
    tdat = lm_data
    tdat = tdat.dropna()
    # rename the names
    renamer = Renamer()
    rcol_others_int = list(map(renamer.rename, tdat.columns[(~tdat.columns.isin([
                                                                                    V.COL_SITEID] +
                                                                                col_curmark_meas
                                                                                ) & tdat.columns.isin(col_int))]))
    rcol_others_cc = list(map(renamer.rename, tdat.columns[(~tdat.columns.isin([
                                                                                   V.COL_SITEID] +
                                                                               col_curmark_meas
                                                                               ) & tdat.columns.isin(col_cc))]))
    rcol_others_nb = list(map(renamer.rename, tdat.columns[(~tdat.columns.isin([
                                                                                   V.COL_SITEID] +
                                                                               col_curmark_meas
                                                                               ) & tdat.columns.isin(col_nb))]))
    rcol_others_nbcc = list(map(renamer.rename, tdat.columns[(~tdat.columns.isin([
                                                                                     V.COL_SITEID] +
                                                                                 col_curmark_meas
                                                                                 ) & tdat.columns.isin(col_nbcc))]))

    rcol_cur_nb = list(map(renamer.rename, tdat.columns[(tdat.columns.isin(col_curmark_meas
                                                                           ) & tdat.columns.isin(col_nb))]))
    rcol_cur_dist = list(map(renamer.rename, tdat.columns[(tdat.columns.isin(col_dist))]))
    rcol_imglevel = renamer.rename(V.COL_IMGLEVEL)
    rcol_curmark = renamer.rename(col_curmark)
    # adlist(d the site

    rcol_dists_poly = ['np.power(' + c + ',' + str(i) + ')' for c in rcol_cur_dist for i in range(1, POLYLEVEL + 1)]
    rcol_dists = [f'bs({rcol_cur_dist[0]}/200, df=10)']
    # rename the data
    tdatfit = tdat.rename(columns=renamer.rename)

    # base formula
    mdict = {'dist': rcol_dists,
             'int': rcol_others_int,
             'nb': rcol_others_nb,
             'self': rcol_cur_nb}
    seq = ['dist', 'int', 'nb', 'self']

    fms = dict()
    fms['site'] = fm_base = rcol_curmark + '~1'
    for mods in powerset(seq):
        if len(mods) > 0:
            mod_name = '_'.join(mods)
            mod_form = '+'.join([fm_base] + [e for m in mods for e in mdict[m]])
            fms[mod_name] = mod_form

    fms['dist_nb_self_cc'] = '+'.join([fm_base] + rcol_dists + rcol_others_nb + rcol_cur_nb + rcol_others_cc)
    fms['dist_poly'] = '+'.join([fm_base] + rcol_dists_poly)
    fms['cc'] = '+'.join([fm_base] + rcol_others_cc)
    fms['nbcc'] = '+'.join([fm_base] + rcol_others_nbcc)
    res_ols = dict()
    for nm, fm in fms.items():
        m = smf.ols(fm + '+' + rcol_imglevel, tdatfit)
        m = m.fit()
        m.summary_d = m.summary()
        m.summary2_d = m.summary2()
        m.rsquared_d = m.rsquared.copy()
        res_ols[nm] = Lmres(summary=m.summary_d, rsquared=m.rsquared,
                            params=m.params)
        m.remove_data()
    return res_ols


# Section to calculate r2 statistics
def getr2(row):
    return pd.Series(
        {r[i].strip().replace(':', ''): r[i + 1] for r in row['model'].summary.tables[0].data for i in [0, 2] if
         r[i].strip() != ''})


def getr2_sitecor(dat, rvar='r2'):
    ref = 'site'
    refr2 = dat.loc[dat[V.COL_MODELCLASS] == ref, rvar].values[0]
    r2corr = 1 - ((1 - dat[rvar]) / (1 - refr2))
    return r2corr


def get_meta(bro):
    hpr = helpers_vz.HelperVZ(bro)
    dat_pannelcsv = hpr.get_pannelcsv()
    dat_measmeta = hpr.get_measuremeta(dat_pannelcsv,
                                       additional_measfilt=sa.and_(db.stacks.stack_name == 'Dist',
                                                                   db.measurements.measurement_name == 'dist-rim',
                                                                   db.ref_planes.channel_name == 'object'))

    fil_good_meas = hpr.get_fil_good_meas(dat_measmeta)
    dat_imgmeta = hpr.get_imgmeta()
    dat_measmeta = dat_measmeta.merge(dat_pannelcsv, how='left')
    dat_measmeta[V.COL_IS_CC] = dat_measmeta[V.COL_IS_CC] == 1
    dat_measmeta = dat_measmeta.set_index(V.COL_MEASID, drop=False)
    return dat_measmeta, dat_imgmeta


def get_data(bro, cur_cond, obj_type, dat_measmeta):
    hpr = helpers_vz.HelperVZ(bro)
    fil_good_meas = hpr.get_fil_good_meas(dat_measmeta)
    d = hpr.get_data(cond_ids=get_condids_for_cond(bro, cur_cond),
                     fil_good_meas=fil_good_meas, object_type=obj_type)
    d = transf_intensities(d, dat_measmeta)
    return d


def get_working_channels(dat_measmeta):
    cns = [c for c in dat_measmeta.loc[dat_measmeta[V.COL_WORKING] == 1,
                                       V.COL_CHANNELNAME].unique() if c != 'object']
    return cns


def run_per_condition(dat, dat_measmeta, dat_imgmeta):
    d = prepare_lmdat(dat, dat_measmeta, dat_imgmeta,
                      meas_names=['dist-rim', 'NbMeanMeanIntensityComp', 'MeanIntensityComp'])
    cur_calc_lm = functools.partial(calc_lm, lm_data=d,
                                    dat_measmeta=dat_measmeta)
    rdic = dict()
    for c in get_working_channels(dat_measmeta):
        rdic.update({c: cur_calc_lm(c)})
    return rdic


def run_per_replicate(dat, dat_measmeta, dat_imgmeta):
    res_rep = dict()
    for cond_id, lmdat_rep in (dat
            .merge(dat_imgmeta[[V.COL_IMGID, V.COL_CONDID]])
            .groupby([V.COL_CONDID])):
        lmdat = prepare_lmdat(lmdat_rep, dat_measmeta, dat_imgmeta,
                              meas_names=['dist-rim', 'NbMeanMeanIntensityComp', 'MeanIntensityComp'])

        cur_calc_lm = functools.partial(calc_lm, lm_data=lmdat,
                                        dat_measmeta=dat_measmeta)
        rdic = {}
        for c in get_working_channels(dat_measmeta):
            rdic.update({c: cur_calc_lm(c)})
        res_rep[cond_id] = rdic
    return res_rep


def get_datmod(model_dict):
    dat_mod = pd.DataFrame(model_dict)
    dat_mod.columns.name = V.COL_CONDNAME
    dat_mod.index.name = V.COL_CHANNELNAME
    dat_mod = dat_mod.stack().to_frame()[0].apply(pd.Series)
    dat_mod.columns.name = V.COL_MODELCLASS
    dat_mod = dat_mod.stack().to_frame()
    dat_mod.columns = [V.COL_MODEL]
    return dat_mod


def get_model_stats(dat_mod):
    # Extract R2
    dat_mod_stats = dat_mod.apply(getr2, axis=1).reset_index().rename(columns={'R-squared': 'r2'})

    # Convert numeric columns to float
    num_cols = ['r2', 'Adj. R-squared', 'Df Residuals', 'Df Model', 'No. Observations']
    for c in num_cols:
        dat_mod_stats[c] = dat_mod_stats[c].astype(np.float)

    # Calculate corrected r2
    dat_mod_stats['r2_corr'] = dat_mod_stats.groupby([V.COL_CHANNELNAME, V.COL_CONDNAME], group_keys=False).apply(
        getr2_sitecor)
    dat_mod_stats['r2_adj_corr'] = dat_mod_stats.groupby([V.COL_CHANNELNAME, V.COL_CONDNAME], group_keys=False).apply(
        getr2_sitecor, rvar='Adj. R-squared')
    return dat_mod_stats


def get_model_params(dat_mod):
    def get_params(row):
        return row['model'].params

    dat_mod_params = dat_mod.apply(get_params, axis=1)
    dat_mod_params.columns.name = V.COL_VAR
    dat_mod_params = dat_mod_params.stack()
    dat_mod_params.name = V.COL_PARAMS
    dat_mod_params = dat_mod_params.reset_index()
    dat_mod_params[V.COL_VAR] = dat_mod_params[V.COL_VAR].map(lambda x: re.sub('^c', '', x))
    return dat_mod_params


def run_analysis(fn_config, cur_cond, fn_out_stats,
                 fn_out_params, per_replicate=False,
                 obj_type='cell'):
    """
    Input:
        fn_config: path to configurationfile
        cur_cond: current condition
        obj_type: object type
    """

    bro = spb.get_bro(fn_config, readonly=True)
    dat_measmeta, dat_imgmeta = get_meta(bro)
    dat = get_data(bro, cur_cond, obj_type, dat_measmeta)

    if not per_replicate:
        res = run_per_condition(dat, dat_measmeta, dat_imgmeta)
        res = {cur_cond: res}
    else:
        res = run_per_replicate(dat, dat_measmeta, dat_imgmeta)

    # Get parameters and statistics
    dat_mod = get_datmod(res)
    dat_stats = get_model_stats(dat_mod)
    dat_params = get_model_params(dat_mod)

    # Save data
    dat_stats.to_csv(fn_out_stats, index=False)
    dat_params.to_csv(fn_out_params, index=False)


if __name__ == '__main__':
    sm = snakemake
    cur_cond = f'{sm.wildcards.cellline}_c{sm.wildcards.conc}_te%_tp{sm.wildcards.tp}'
    per_rep = int(sm.wildcards.rep) == 1
    run_analysis(sm.input.fn_config,
                 cur_cond,
                 sm.output.fn_out_stats[0],
                 sm.output.fn_out_params[0],
                 per_replicate=per_rep)
