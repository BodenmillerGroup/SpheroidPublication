from src.variables import Vars as V
from src.config import Conf as C

import src.tools as tools


def link_figures(src=C.target_fol_figures,
                 target=V.FOL_FIGURES):
    tools.symlink_folders(src, target, mkdirs=True)

if  __name__ == '__main__':
    link_figures()
