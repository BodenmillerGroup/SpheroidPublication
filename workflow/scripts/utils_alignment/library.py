import workflow.scripts.utils_alignment.variables as V
from workflow.scripts.utils_alignment.czi_scene_reader import CziSceneReader, get_czi_nscenes
from workflow.scripts.utils_alignment.imcmeta import ImcMeta
import pathlib
import pandas as pd
import numpy as np
import skimage.io as skio
import re
import imagesize






def get_scene_bbox(cziscene, transform=None):
    """
    Generates boundingboxes for a czi scene.
    """
    start = cziscene.start[4:6]
    shape = cziscene.shape[4:6]
    y1, x1 = start
    y2, x2 = np.array(start) + np.array(shape)
    # if scale[0] < 0:
    #    x1, x2 = x2, x1
    # if scale[1] < 0:
    #    y1, y2 = y2, y1
    if transform is not None:
        (x1, y1), (x2, y2) = transform(np.array(((x1, y1), (x2, y2))))
    return pd.Series({'x': min(x1, x2),
                      'w': abs(x2 - x1),
                      'y': min(y2, y1),
                      'h': abs(y2 - y1)})


def get_imcpanorama_bbox(row):
    """
    Panorama coordiantes are not entirely accurate, as they can be tilted.
    This just generates the bouning box.

    Params:
    -------
    row: a row from the dat_panometa

    Returns:
    -------
    a dict containing the bounding box coordinates.
    """

    x_cols, y_cols = ([[f'Slide{c}{i + 1}PosUm' for i in range(4)] for c in ('X', 'Y')])

    return pd.Series({'x': row[x_cols].values.min(),
                      'w': row[x_cols].values.max() - row[x_cols].values.min(),
                      'y': row[y_cols].values.min(),
                      'h': row[y_cols].values.max() - row[y_cols].values.min()})


def get_imcacquistion_bbox(row):
    """
    Generates boundingboxes for IMC acquisitions.
    """

    x, y, w, h, dx, dy = [row[c].values[0] for c in
                          ['ROIStartXPosUm', 'ROIStartYPosUm', 'MaxX', 'MaxY', 'AblationDistanceBetweenShotsX',
                           'AblationDistanceBetweenShotsY']]

    return pd.Series({'x': x,
                      'w': w * dx,
                      'y': y - h * dy,
                      'h': h * dy})


def get_scene_meta(fn, transform=None):
    scenes = list()

    scene_idx = list(range(get_czi_nscenes(fn)))
    for i in scene_idx:
        with CziSceneReader(fn, scene=i) as czi:
            d = get_scene_bbox(czi, transform=transform)
            scenes.append(d)

    dat_scene_cord = pd.DataFrame(scenes)
    dat_scene_cord[V.SCENEID] = scene_idx
    dat_scene_cord[V.SLIDESCAN] = fn.stem
    return dat_scene_cord

def get_imcfn_meta(fns, re_str):
    qre = re.compile(re_str)
    m_list = [m.groupdict() for s in fns for m in qre.finditer(s)]
    assert len(m_list) == len(fns), 'Regexp doesnt match all filenames!'
    for m, f in zip(m_list, fns):
        m[V.SESSIONID] = f
        m[V.SLIDENUMBER] = int(m[V.SLIDENUMBER])
    m_list = [pd.Series(m) for m in m_list]

    return pd.DataFrame(m_list)


def get_crop_meta(paths, re_str):
    qre = re.compile(re_str)
    m_list = [m.groupdict() for s in paths for m in qre.finditer(s.stem)]
    assert len(m_list) == len(paths), 'Regexp doesnt match all filenames!'
    for m, p in zip(m_list, paths):
        m[V.CROPPATH] = p.stem
        m[V.CROPH], m[V.CROPW] = imagesize.get(p)

    m_list = [pd.Series(m) for m in m_list]
    return pd.DataFrame(m_list)


def get_imc_meta(m):
    dat_pano_cord = m.dat_panometa.groupby(by=[V.PANOID, V.SESSIONID]).apply(get_imcpanorama_bbox).reset_index()
    dat_ac_cord = m.dat_acmeta.groupby(by=[V.ACID, V.ACROIID, V.SESSIONID]).apply(
        get_imcacquistion_bbox).reset_index().merge(m.dat_acroimeta)
    return dat_pano_cord, dat_ac_cord


# scene matching
def add_center(dat, transform=None):
    dat['center_x'] = dat['x'] + (dat['w'] / 2)
    dat['center_y'] = dat['y'] + (dat['h'] / 2)
    if transform is not None:
        dat[['center_x', 'center_y']] = transform(dat[['center_x', 'center_y']].values)
    return dat


def add_order(dat, group):
    dat['xorder'] = dat.groupby(by=group)['center_x'].transform(lambda x: np.argsort(x))


def get_match(d, dat, idcol):
    dists = np.sqrt(((dat[['center_x', 'center_y']] - d[['center_x', 'center_y']].values) ** 2).sum(axis=1).values)
    idxmin = np.argmax(-dists)
    d['dist'] = dists[idxmin]
    d[idcol] = dat[idcol].values[idxmin]
    return d


# Plotting:
import matplotlib.pyplot as plt
import imctools.io.ometiffparser as ioome


def plot_czi_slide(fn_czi, scenes=None,
                   scale=(1, 1), offset=(0, 0),
                   ax=None, channel=0, cmap='Greys'):
    if ax is None:
        fig, ax = plt.subplots(1, )
        plt.axis('equal')

    if scenes is None:
        scenes = range(get_czi_nscenes(fn_czi))
    for s in scenes:
        with CziSceneReader(fn_czi, scene=s) as czi:
            curimg = czi.asarray()[0, :, 0, channel, :, :].squeeze()
            start = czi.start[4:6]
            shape = czi.shape[4:6]
        y1, x1 = start
        y2, x2 = np.array(start) + np.array(shape)
        x1 = (x1 * scale[0]) + offset[0]
        x2 = (x2 * scale[0]) + offset[0]
        y1 = (y1 * scale[1]) + offset[1]
        y2 = (y2 * scale[1]) + offset[1]
        ax.imshow(np.flipud(np.log(curimg + 1)), extent=(x1, x2, y1, y2), cmap='Greys')
    ax.autoscale_view()
    return ax


def plot_imc_slide(fol_imc, panoid=None,
                   ax=None, channel='Ir191', cmap='viridis', alpha=1):
    if ax is None:
        fig, ax = plt.subplots(1, )
        plt.axis('equal')
    m = ImcMeta(fol_imc)
    d = m.dat_acmeta.merge(m.dat_acroimeta)
    if panoid is not None:
        d = d.query(f"PanoramaID in [{panoid}]")
    for idx, (x, y, w, h, acid) in d.sort_values('AcquisitionID', ascending=False)[
        ['ROIStartXPosUm', 'ROIStartYPosUm', 'MaxX', 'MaxY', 'AcquisitionID']].iterrows():
        fn_ome = next(fol_imc.glob(f'*_a{int(acid)}_ac.ome.tiff'))
        imcac = ioome.OmetiffParser(fn_ome).get_imc_acquisition()
        curimg = imcac.get_img_by_metal(channel)
        # print(x,y,w,h)
        ax.imshow(((np.log(curimg + 1))), extent=(x, x + w, y - h, y), cmap=cmap, alpha=alpha)
    ax.autoscale_view()

    return ax


def plot_imcpano_slide(fol_imc, panoid=None,
                       ax=None, cmap='viridis', alpha=1):
    if ax is None:
        fig, ax = plt.subplots(1, )
        plt.axis('equal')
    m = ImcMeta(fol_imc)
    d = m.dat_panometa
    if panoid is not None:
        d = d.query(f"PanoramaID in [{panoid}]")
    for idx, (x1, y2, x2, y1, panoid) in d[['SlideX1PosUm', 'SlideY1PosUm',
                                            'SlideX3PosUm', 'SlideY3PosUm', 'PanoramaID']].iterrows():
        fn_pano = next(C.fol_imc.glob(f'*_p{int(panoid)}_pano.png'))
        curimg = ((skio.imread(fn_pano)))
        # print(x,y,w,h)
        ax.imshow((curimg), extent=(x1, x2, y1, y2), cmap=cmap, alpha=alpha)
    ax.autoscale_view()

    return ax

def save_scene_images(fn_czi, fol_out):
    fol_out = pathlib.Path(fol_out)
    for s in range(get_czi_nscenes(fn_czi)):
        with CziSceneReader(fn_czi, scene=s) as czi:
            for c in range(czi.shape[3]):
                curimg = czi.asarray()[0, :, 0, c, :, :].squeeze()
                fn = V.TPL_SLIDEPLANE.format(slide=fn_czi.stem, scene=s, channel=c) + '.tiff'
                skio.imsave(fol_out / fn, curimg)

