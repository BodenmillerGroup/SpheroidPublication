import pathlib
from src.variables import Vars as V

"""
This contains all project specific configurations, such as file paths to data,
specific naming schemes, output folders.
"""
class Conf:
    fol_config = pathlib.Path('../config')
    fn_config =  fol_config / 'config.yml'
    fol_plts = V.FOL_FIGURES
    fol_paper = pathlib.Path('../../../../paper/figures/oexp_align_fullqc')

    target_fol_figures = pathlib.Path('/mnt/bbvolume/server_homes/vitoz/mnt/tmp/scratch/vitoz/Data/Analysis/20200123_testalign/') / 'plots/'
    # Siteoverview related
    fol_plts_siteoverviews = fol_plts / 'siteoverviews'
    
    # Overexpression related
    fol_plts_oexp = fol_plts / 'all_sign_thesis_norm_oneflag_bobya'
    fn_res = fol_plts / 'res_all_sign_thesis_norm.csv'
    fn_phist = fol_plts/ 'phist.png'
    fn_vulcano = 'vulcano_overall.png'
    fn_vulcano_construct = 'vulcano_{condition}.png'
    fol_lmm = pathlib.Path('/mnt/bbvolume/server_homes/vitoz/mnt/tmp/scratch/vitoz/Data/Analysis/20200123_testalign/results/lmm_ml_oneflag')
    fn_lmdata = fol_lmm / 'lmdata_V1.loom'    
    fn_format_fit = '{channel}_bobyqa_simp.Rdata'
    #fn_format_fit = '{channel}_simple.Rdata'
    #fn_format_fit = '{channel}_reform.Rdata'
    fn_format_parameters = fn_format_fit+'_parameters.csv'
    fn_format_absstat = fn_format_fit+'_absstat.csv'
    fn_format_model = fn_format_fit+'_model.csv'
    fn_out_sig = fol_plts / 'dat_issig.csv'
    
    fn_constructs = pathlib.Path('../config/construct_meta.csv')
    REF_COND = 'ctrl'
    SUFFIX_NB = '-NB'
    FIL_FLAGPOS = 'is-flagpos'
    FIL_FLAGPOSNB = FIL_FLAGPOS+V.SUFFIX_FILNB
    FIL_GFPPOS = 'is-gfppos'
    FIL_GFPPOSNB = FIL_GFPPOS+V.SUFFIX_FILNB
    FIL_LM = 'modelfitcondflag_v1'
    FIL_LM_CLASSES = ['doubt', 'ctrl', 'oexp-NB', 'oexp']
    
    # Variability analysis
    fn_varanalysis_modstats_out ='../../figures/marker_relations_r2_fullqc_percond.csv'
    fn_varanalysis_modstats_perrep ='../../figures/marker_relations_r2_fullqc_perrep.csv'
    fn_varanalysis_params = '../../figures/marker_relations_r2_fullqc_percond_params.csv'
    COL_R2 = 'Adj. R-squared'
    


