import src.config.variables as V
import src.config.config as C

import src.tools as tools
import src.config.variables as V
import src.config.config as C

def link_figures(src=C.target_fol_figures,
                 target=V.FOL_FIGURES):
    tools.symlink_folders(src, target, mkdirs=True)

if  __name__ == '__main__':
    link_figures()
