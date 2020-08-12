import pycytools.library as pclib
import scipy
import scipy.ndimage as ndimage
import numpy as np

import matplotlib.colors as colors
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib as mpl
import matplotlib.pyplot as plt

import spherpro.db as db

from src.variables import Vars as V

class AnndataImagePlotter:
    """
    A helper class to plot images from an andata
    object
    """
    def __init__(self, bro):
        self.ios = bro.io.stackimg
        self.bro = bro
        
    def _plot_imc(self, planid, imgid, ax=None,
                 winsorize=(0, 0.01),
                 sigma=0.75, cmap='Greys_r', norm=None):
        """
        Plots the image underlying a planeid
        """
        if ax is None:
            fig = plt.figure(figsize=(20,20))
            ax = plt.gca()

        img = self.ios.get_planeimg(imgid, planid)
        img = scipy.stats.mstats.winsorize(img, winsorize)
        if sigma > 0:
            img = ndimage.gaussian_filter(img, sigma=sigma)
        ax.imshow(img, cmap=cmap,norm=norm)
        return ax

    @staticmethod
    def _add_raw_contour(mask, ax=None, linewidths=0.5, linestyles=':', col='Gray'):
        """
        Adds background mask contour
        """
        if ax is None:
            fig = plt.figure(figsize=(20,20))
            ax = plt.gca()
        ax.contour(mask, [0,0.5],colors=[col],linewidths=linewidths, linestyles=linestyles)

    @staticmethod
    def add_scalebar(ax, resolution=0.000001, color='white',pad=0.5,
                                frameon=False, size='small', **kwargs):
        """
        Adds scalebar
        """
        scalebar = ScaleBar(resolution, color=color, pad=pad,
                            frameon=frameon, **kwargs) # 1 pixel = 0.2 meter
        ax.add_artist(scalebar)

    @staticmethod    
    def _add_contour(mask, val, objnr, cmap, ax, **kwargs):
        for i in range(len(cmap)):
            curobj = objnr[val == i]
            AnndataImagePlotter._add_raw_contour(np.isin(mask, curobj), ax=ax, col=cmap[i],**kwargs)
        

    
    @staticmethod
    def add_subplot_axes(ax,rect,labelcolor='b'):
        fig = plt.gcf()
        box = ax.get_position()
        width = box.width
        height = box.height
        inax_position  = ax.transAxes.transform(rect[0:2])
        transFigure = fig.transFigure.inverted()
        infig_position = transFigure.transform(inax_position)    
        x = infig_position[0]
        y = infig_position[1]
        width *= rect[2]
        height *= rect[3]  # <= Typo was here
        subax = fig.add_axes([x,y,width,height])
        x_labelsize = subax.get_xticklabels()[0].get_size()
        y_labelsize = subax.get_yticklabels()[0].get_size()
        x_labelsize *= rect[2]**0.5
        y_labelsize *= rect[3]**0.5
        subax.xaxis.set_tick_params(labelsize=x_labelsize)
        subax.yaxis.set_tick_params(labelsize=y_labelsize)
        subax.xaxis.set_visible(False)
        #subax.yaxis.set_ticks_position('left')
        return subax


    def plot_anndata_subplots(self, imid, ad, figsize=10):
        """
        Plots subplots
        """
        bro = self.bro
        
        d = ad[ad.obs.image_id == imid,:]
        ncol = d.shape[1]
        fig, axs = plt.subplots(nrows=2, ncols=ncol,
                                figsize=(figsize*ncol, figsize*2),
                                sharex=True, sharey=True)
        if ncol == 1:
            axs = np.array([[axs[0]], [axs[1]]])
        mask = bro.helpers.dbhelp.get_mask(imid, object_type='cell')
        for ax, varidx in zip(axs[0], d.var.index):
            img = pclib.map_series_on_mask(mask, d[:, varidx].X, d.obs.object_number)
            bro.plots.heatmask.do_heatplot(img,ax=ax, colorbar=False)
            varmeta = d.var.loc[varidx]
            ax.set_title(f'{varmeta[V.COL_CHANNELNAME]} \n {varmeta[V.COL_MEASNAME]}')
            self.add_scalebar(ax)
            ax.axis('off')


        for ax, varidx in zip(axs[1], d.var.index):
            varmeta = d.var.loc[varidx]
            self._plot_imc(int(varmeta[db.planes.plane_id.key]), imid, ax=ax)
            self._add_raw_contour(mask, ax=ax)
            self.add_scalebar(ax)
            ax.axis('off')
        for ax in axs.flatten():
            cax = self.add_subplot_axes(ax,(0.85,0,0.1,.5))
            plt.colorbar(ax.images[0], cax=cax)
        #plt.subplots_adjust(bottom=0.3, top=0.7, hspace=0)

        return axs