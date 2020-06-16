import workflow.scripts.utils_alignment.library_java as libj
import workflow.scripts.utils_alignment.variables as V
import skimage.transform as sktransf
import pathlib
import numpy as np
import warnings
from tqdm import tqdm
import skimage.io as skio
import skimage.util as skutil
import matplotlib.pyplot as plt
import tifffile

def align_scene(slide, scene,
                meta_scene,
                meta_imc,
                transf_scene2imc,
                fol_trakem2,
                fol_imgs_scene,
                fol_imgs_imc,
                align_params,
                channel_slide=0,
                desc='v1'):
    meta_scene = meta_scene.query(f'({V.SLIDE}=="{slide}") & ({V.SCENEID}=="{scene}")').iloc[0, :]
    meta_imc = meta_imc.query(f'({V.SLIDE}=="{slide}") & ({V.SCENEID}=="{scene}")')

    tproj = libj.SlideImcAlignment(fol_trakem2)
    fn_scene = V.TPL_SLIDEPLANE_IMG.format(slide=slide, scene=scene, channel=channel_slide)
    fn_align = V.TPL_SLIDESCENE_ALIGN.format(slide=slide, scene=scene,
                                             desc=desc)

    tproj.add_img_scene(fol_imgs_scene / fn_scene, meta_scene[['x', 'y']].values, transf_scene2imc)
    for idx, row in meta_imc.sort_values(by=V.ACID, ascending=False).iterrows():
        file = str(fol_imgs_imc / f'{row[V.FNAC]}.tiff')
        try:
            tproj.add_img_imc(file, row[['x', 'y']].values)
        except libj.JavaException:
            libj.warnings.warn(f'{file} not found!')
    for params in align_params:
        libj.logger.start()
        tproj.align_layers_affine(params)
        libj.logger.stop()

    fn_proj = (fol_trakem2 / fn_align)
    tproj.save_project(fn_proj)
    tproj.close_project()
    
def get_scene_alignparams():
    paramAffine = libj.RAParam()
    """
    Documentation:

    https://github.com/trakem2/TrakEM2/blob/e2c46b0851a090310542e58ee2d7afc895b57a1d/src/amain/java/mpicbg/trakem2/align/AbstractLayerAlignmentParam.java
     "Translation", "Rigid", "Similarity", "Affine"
     
     https://www.ini.uzh.ch/~acardona/howto.html
     "SIFT parameters explained"
     """
    paramAffine.ppm.sift.steps = 3
    paramAffine.ppm.sift.fdSize =8
    paramAffine.ppm.sift.fdBins = 8
    paramAffine.ppm.rod = 0.92
    paramAffine.ppm.clearCache = True
    paramAffine.ppm.maxNumThreadsSift = 12


    paramAffine.minInlierRatio = 0.0
    paramAffine.minNumInliers = 12
    paramAffine.expectedModelIndex = 1
    paramAffine.multipleHypotheses = False
    paramAffine.rejectIdentity = False
    paramAffine.identityTolerance = 5
    paramAffine.maxNumNeighbors = 1
    paramAffine.maxNumFailures = 3
    paramAffine.maxNumThreads = 12

    paramAffine.desiredModelIndex = 1

    paramAffine.regularize = False
    paramAffine.maxIterationsOptimize = 1000
    paramAffine.maxPlateauwidthOptimize = 200
    paramAffine.regularizerIndex = 0
    paramAffine.visualize = False

    setattr(paramAffine,'lambda', 0.1)
    paramAffine.ppm.sift.initialSigma = 10
    paramAffine.maxEpsilon = 200
    paramAffine.ppm.sift.minOctaveSize = 64
    paramAffine.ppm.sift.maxOctaveSize = 1024

    params_coarse = paramAffine.clone()

    paramAffine.ppm.sift.initialSigma = 1
    paramAffine.maxEpsilon = 20
    params_fine = paramAffine.clone()
    return params_coarse, params_fine
    



def get_crop_alignparms():
    paramAffine = libj.RAParam()
    paramAffine.ppm.sift.steps = 3
    paramAffine.ppm.sift.fdSize = 8
    paramAffine.ppm.sift.fdBins = 8
    paramAffine.ppm.rod = 0.92
    paramAffine.ppm.clearCache = True
    paramAffine.ppm.maxNumThreadsSift = 12

    paramAffine.minInlierRatio = 0.0
    paramAffine.minNumInliers = 12
    paramAffine.expectedModelIndex = 1
    paramAffine.multipleHypotheses = False
    paramAffine.rejectIdentity = False
    paramAffine.identityTolerance = 5
    paramAffine.maxNumNeighbors = 1
    paramAffine.maxNumFailures = 3
    paramAffine.maxNumThreads = 12

    paramAffine.desiredModelIndex = 1

    paramAffine.regularize = False
    paramAffine.maxIterationsOptimize = 1000
    paramAffine.maxPlateauwidthOptimize = 200
    paramAffine.regularizerIndex = 0
    paramAffine.visualize = False

    paramAffine.ppm.sift.initialSigma = 1
    paramAffine.maxEpsilon = 50
    paramAffine.ppm.sift.minOctaveSize = 32
    paramAffine.ppm.sift.maxOctaveSize = 1024
    return paramAffine

def align_crops(crop_path,
                meta_scene,
                meta_imc,
                fol_trakem2,
                fol_imgs_scene,
                fol_imgs_imc,
                align_params,
                cropdesc,
                channel_slide=0,
                desc='v1', minwh_scene=150):
    meta_imc = meta_imc.query(f'({V.CROPPATH} == "{crop_path}")').iloc[0]
    meta_scene = meta_scene.query(f'({V.CROPPATH} == "{crop_path}")').iloc[0]

    if meta_scene[[V.CROPW, V.CROPH]].min() < minwh_scene:
        warnings.warn(f'Image {meta_scene} to small!')
        return False
    tproj = libj.SlideImcAlignment(fol_trakem2)
    fn_scene = V.TPL_SLIDESCENE_CROP_IMG.format(crop_path=crop_path,
                                                channel=channel_slide,
                                                cropdesc=cropdesc)
    fn_align = V.TPL_SLIDESCENE_CROP_ALIGN.format(crop_path=crop_path,
                                                  desc=desc,
                                                  cropdesc=cropdesc)

    transf_scene = libj.params_to_affinetransform(meta_scene[V.PARAM_AT_SCENE])

    try:
        tproj.add_img_scene(fol_imgs_scene / fn_scene,
                            meta_scene[[V.CROPY, V.CROPX]].values,
                            transf_scene)
    except libj.JavaException:
        libj.warnings.warn(f'{fn_scene} not found!')
        tproj.close_project()
        return False
    row = meta_imc
    file = str(fol_imgs_imc / f'{row[V.CROPPATH]}.tiff')

    transf_imc = libj.params_to_affinetransform(meta_scene[V.PARAM_AT_IMC])
    try:
        tproj.add_img_imc(file, row[[V.CROPY, V.CROPX]].values,
                          transf_imc)
    except libj.JavaException:
        libj.warnings.warn(f'{file} not found!')

    for params in align_params:
        libj.logger.start()
        tproj.align_layers_affine(params)
        libj.logger.stop()

    fn_proj = (fol_trakem2 / fn_align)
    tproj.save_project(fn_proj)
    tproj.close_project()

def get_params_from_project(path_project):
    alignment = libj.SlideImcAlignment(fn_proj= path_project)
    p_scene = alignment.layer_scene.getDisplayables()[0]
    # re_params = fn_tpl_to_re(V.TPL_SLIDEPLANE_IMG)
    param_scene = get_atparam_string_from_patch(p_scene)

    ps_imc = list(alignment.layer_imc.getDisplayables())
    params_imc = [get_atparam_string_from_patch(p) for p in ps_imc]
    fns_imc = [pathlib.Path(p.filePath).stem for p in ps_imc]
    res = {V.PARAM_AT_IMC: params_imc,
           V.PARAM_AT_SCENE: param_scene,
           V.FNAC: fns_imc}
    alignment.close_project()
    return res

def fn_tpl_to_re(tpl):
    re_str = tpl.replace('{', '(?P<').replace('}', '>.*)')
    return re_str

def get_atparam_string_from_patch(patch):
    params = libj.affinetransform_to_params(patch.getAffineTransform())
    return np.array2string(params, separator=',')

def get_params_from_string(string):
    return np.array(eval(string))

def get_scene_rough_imccrop(row_crop):
    params_scene = get_params_from_string(row_crop[V.PARAM_AT_SCENE])
    transf_scene = sktransf.AffineTransform(params_scene)
    params_imc = get_params_from_string(row_crop[V.PARAM_AT_IMC])
    transf_imc = sktransf.AffineTransform(params_imc)
    bbox_crop = get_crop_bbox_points(row_crop)
    p = transf_points(bbox_crop, transf_imc, transf_scene)
    bbox_scenecrop = get_bbox_from_2D(p)
    row_cropout = row_crop[[V.CROPPATH, V.SLIDE, V.SCENEID, V.CROPID,
                            V.PARAM_AT_SCENE, V.PARAM_AT_IMC]].copy()

    r = row_cropout
    r[V.CROPY], r[V.CROPX], r[V.CROPH], r[V.CROPW] = bbox_scenecrop
    #r[V.PARAM_AT_IMC] = np.array2string(params_imc, separator=',')
    #r[V.PARAM_AT_SCENE] = np.array2string(params_scene, separator=',')
    return row_cropout

def get_scene_fine_croptransf(row_crop, scale=1):
    """
    Get the fine crop parameters:
    - Calculate a transform that:
        has x/y = (0,0) as origin
    - has a final scale of 'scale'
    -> after applying that transform, getting the crop should be a matter
        of cropping using the width & height of the crop * scale
    """
    params_scene = get_params_from_string(row_crop[V.PARAM_AT_SCENE])
    transf_scene = sktransf.AffineTransform(params_scene)
    params_imc = get_params_from_string(row_crop[V.PARAM_AT_IMC])
    transf_imc = sktransf.AffineTransform(params_imc)
    bbox_crop = get_crop_bbox_points(row_crop)
    bbox_coords = transf_imc(np.array(bbox_crop)-np.array(bbox_crop[0]))
    #x, y = bbox_coords.min(axis=0)
    #transf_t = sktransf.AffineTransform(translation=(-x, -y))
    transf_s = sktransf.AffineTransform(scale=(scale, scale))
    return transf_scene + transf_imc.inverse + transf_s

def transf_points(p2d, transf_imc, transf_scene):
    t2d = transf_imc(p2d)
    s2d = transf_scene.inverse(t2d)
    return s2d

def get_crop_bbox_points(dat_crop):
    y, x, h, w = dat_crop[[V.CROPX, V.CROPY, V.CROPW, V.CROPH]]
    points = [(x,y), (x,y+h), (x+w,y), (x+w,y+h)]
    return points

def get_bbox_from_2D(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x = min(xs)
    w = max(xs)-min(xs)
    y = min(ys)
    h = max(ys)-min(ys)
    return int(x), int(y), int(w), int(h)

def get_bbox_points(x,y,w,h):
    points = [(x,y), (x,y+h), (x+w,y), (x+w,y+h)]
    return points


# cropping
def crop_scene(scenecrops, fol_input, fol_out, cropdesc, channel, show=False):
    for (slide, scene), dat in tqdm(scenecrops.groupby(by=[V.SLIDE,
                                                           V.SCENEID])):
        fn = V.TPL_SLIDEPLANE_IMG.format(slide=slide,
                                         scene=scene,
                                         channel=channel)
        img = tifffile.imread(str(fol_input / fn))
        for idx, row in dat.iterrows():
            fn_out = V.TPL_SLIDESCENE_CROP_IMG.format(
                crop_path=row[V.CROPPATH],
                channel=channel,
                cropdesc=cropdesc)
            x, y, w, h = row[[V.CROPX, V.CROPY, V.CROPW, V.CROPH]]
            cropimg = img[x:(x + w), y:(y + h)]
            if np.any(np.array(cropimg.shape) == 0):
                warnings.warn(f'{str(row)}\n Crop outside of bounds.')
            else:
                if show:
                    plt.imshow(cropimg)

                skio.imsave(fol_out / fn_out, cropimg)


def crop_scene_fine(row_params, fol_imgs, fol_out, scale=1, channels=(0, 1),
                    cropdesc_in='rcrop', cropdesc_out='fcrop', desc='v1'):
    img_outshape = np.array(row_params[[V.CROPW, V.CROPH]].values * scale, dtype=np.int)
    transf = get_scene_fine_croptransf(row_params, scale=scale)
    for c in channels:
        fn = fol_imgs / V.TPL_SLIDESCENE_CROP_IMG.format(
            crop_path=row_params[V.CROPPATH], channel=c, cropdesc=cropdesc_in, desc=desc)
        img = skio.imread(fn)
        img_t = sktransf.warp(img, transf.inverse, output_shape=img_outshape)
        fn_out = V.TPL_SLIDEPLANE_CROP_SCALED_IMG.format(
            crop_path=row_params[V.CROPPATH], channel=c, cropdesc=cropdesc_out, desc=desc,
            scale=scale)

        skio.imsave(fol_out / fn_out, skutil.img_as_uint(img_t))

