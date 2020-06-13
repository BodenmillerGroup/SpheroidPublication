# metadata handling
import workflow.scripts.utils_alignment.variables as V
import pandas as pd
import pathlib

class ImcMeta:
    """ A quick and dirty container for all metadata of an IMC acquisition """
    def __init__(self, fol_imc):
        fol = pathlib.Path(fol_imc)
        self.folder = fol
        self.dat_acmeta = pd.read_csv(list(fol.glob('*_Acquisition_meta.csv'))[-1]).rename(columns={'ID': V.ACID})
        self.dat_acroimeta = pd.read_csv(list(fol.glob('*_AcquisitionROI_meta.csv'))[-1]).rename(
            columns={'ID': V.ACROIID})
        self.dat_panometa = pd.read_csv(list(fol.glob('*_Panorama_meta.csv'))[-1]).rename(columns={'ID': V.PANOID})
        self.dat_roipointmeta = pd.read_csv(list(fol.glob('*_ROIPoint_meta.csv'))[-1]).rename(
            columns={'ID': V.ROIPOINTID})

        for d in [self.dat_acmeta, self.dat_acroimeta, self.dat_panometa, self.dat_panometa, self.dat_roipointmeta]:
            d[V.SESSIONID] = fol_imc.stem
