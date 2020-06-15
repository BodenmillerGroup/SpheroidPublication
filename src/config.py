import pathlib
from src.variables import Vars as V

"""
This contains all project specific configurations, such as file paths to data,
specific naming schemes, output folders.
"""
class Conf:
    fn_config =  '../config/config.yml'
    fol_plts = V.FOL_FIGURES
    fol_paper = pathlib.Path('../../../../paper/figures/')

    target_fol_figures = pathlib.Path('/home/vitoz/mnt/tmp/scratch/vitoz/Data/Analysis/20200303_phys_p173-p176-physiology_v3') / 'plots/'
    # from 99_site_overviews
    fol_plts_siteoverviews = fol_plts / 'siteoverviews'
    
    # marker variability
    fn_modstat_percond = '../../figures/marker_relations_r2_fullqc_percond_wpher2.csv'
    fn_modstat_perrep = fol_plts / 'marker_relations_r2_perrep.csv'