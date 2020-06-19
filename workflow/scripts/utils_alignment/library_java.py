import scyjava_config
try:
    scyjava_config.add_options('-Xmx25g')
except ValueError:
    pass

import imagej
# Use fiji from container
ij = imagej.init('/Fiji.app')

from jnius import autoclass, signatures, PythonJavaClass, java_method
from jnius.reflect import JavaException
import warnings
import numpy as np

Project = autoclass('ini.trakem2.Project')
ControlWindow = autoclass('ini.trakem2.ControlWindow')
Patch = autoclass('ini.trakem2.display.Patch')
Utils = autoclass('ini.trakem2.utils.Utils')
Align = autoclass('mpicbg.trakem2.align.Align')
AffineTransform = autoclass('java.awt.geom.AffineTransform')

# To check plugins path
menus = autoclass('ij.Menus')
p = menus.getPlugInsPath()
print(str(p))
# classes for alignment
JObject = autoclass('java/lang/Object')
RegularizedAffineLayerAlignment = autoclass('mpicbg.trakem2.align.RegularizedAffineLayerAlignment')
RAParam = autoclass('mpicbg.trakem2.align.RegularizedAffineLayerAlignment$Param')
HashSet = autoclass('java/util/HashSet')
List = autoclass('java.util.ArrayList')

ControlWindow.setGUIEnabled(False)

# Register it with ImageJ.
class MyOutputListener(PythonJavaClass):
    __javainterfaces__ = ['org/scijava/console/OutputListener']
    def __init__(self):
        super().__init__()
        self.out = []
        self.do_log = False
        

    @java_method('(Lorg/scijava/console/OutputEvent;)V')
    def outputOccurred(self, e):
        source = e.getSource().toString()
        output = e.getOutput()
        if self.do_log:
            self.out.append(output)
            
    def start(self):
        self.do_log = True
        
    def stop(self):
        self.do_log = True
        
    def clear_output(self):
        self.out = []
        

# Instantiate our OutputListener.
logger = MyOutputListener()

# Register it with ImageJ.
ij.console().addOutputListener(logger)


class Trakem2Alignment:
    """
    A class to hold a trakem2 alignment project
    """

    def __init__(self, fol_out=None, fn_proj=None):
        assert not ((fol_out is None) and (fn_proj is None)), f'fol_out OR fol_proj should be provided'
        if fol_out is not None:
            self.project = Project.newFSProject("blank", None, str(fol_out), False)
        elif fn_proj is not None:
            self.project = Project.openFSProject(str(fn_proj))

    @property
    def loader(self):
        """ Get the loader class """
        self.project.getLoader()

    @property
    def layerset(self):
        return self.project.getRootLayerSet()

    def update_layers(self):
        self.project.getLayerTree().updateList(self.layerset)
        for l in self.layerset.getLayers():
            l.recreateBuckets()
        self.layerset.setMinimumDimensions()

    def recreate_minimaps(self, sleep=0.5):
        futures = []
        for patch in self.layerset.getDisplayables(Patch):
            futures.append(patch.updateMipMaps())

        # This is done asynchronously, so wait till its finished
        for f in futures:
            f.get()
        #while len([1 for f in futures if f.isDone() == 0]) > 0:
        #    time.sleep(0.5)

    def save_project(self, fn_proj, minimaps=True, overwrite=True):
        if minimaps:
            self.recreate_minimaps()
        self.project.saveAs(str(fn_proj), overwrite)

    def close_project(self):
        self.project.destroy()


class SlideImcAlignment(Trakem2Alignment):
    def __init__(self, fol_out=None, fn_proj=None):
        super().__init__(fol_out, fn_proj)
        if fol_out is not None:
            layerset = self.layerset
            for i in range(3):
                # Somehow image does not get loaded if there are only two scenes...
                layerset.getLayer(i, 1, True)

    def add_img_scene(self, fn_scene, location, transform):
        p = Patch.createPatch(self.project, str(fn_scene))
        at = create_location_transform(location, transform)
        p.setAffineTransform(at)
        self.layer_scene.add(p)

    def add_img_imc(self, fn_img, location=(0, 0), transform=None, minmax=(0, 50)):
        p = Patch.createPatch(self.project, str(fn_img))
        if transform is None:
            # The imc images are flipped in y in relation to the slide coordinates.
            # If no explicit transform is given, flip the image in Y.
            p.scale(1, -1,
                    p.getBoundingBox().width / 2,
                    p.getBoundingBox().height / 2)
            p.translate(float(location[0]), float(location[1]))
        else:
            at = create_location_transform(location, transform)
            p.setAffineTransform(at)
        p.setMinAndMax(minmax[0], minmax[1])
        self.layer_imc.add(p)

    @property
    def layer_scene(self):
        return self.layerset.getLayers()[1]

    @property
    def layer_imc(self):
        return self.layerset.getLayers()[2]

    def align_layers_affine(self, parameters):
        """
        -> based on https://github.com/saalfeldlab/parallel-elastic-alignment/blob/master/import-and-align.bsh

        "JavaException: JVM exception occurred: Illegal Capacity: -1"
        -> Usually means the alignment did fail, eg as no alignment was found using the parameters
        """
        regal = RegularizedAffineLayerAlignment()
        fixed_layers = HashSet()
        fixed_layers.add(self.layer_imc)  # alignment relative to imc

        empty_layers = HashSet()  # First layer is empty
        empty_layers.add(self.layerset.getLayers()[0])

        align_layers = List()
        align_layers.add(self.layer_scene)
        align_layers.add(self.layer_imc)

        self.update_layers()
        bounds = self.layerset.get2DBounds()
        try:
            regal.exec(parameters,
                       align_layers, fixed_layers, empty_layers,
                       bounds,
                       True,
                       True,
                       PatchFilter())
            self.update_layers()
        except JavaException as e:
            warnings.warn(f'Alignment failed - check the parameters!\n {e}')
        return regal


def create_location_transform(location, transform):
    """
    Creates an affine transform that first translates (original coordinates) and then
    transforms.
    """
    at = AffineTransform()
    at.setToTranslation(float(location[0]), float(location[1]))
    at.preConcatenate(transform)
    return at


def params_to_affinetransform(params):
    """
    Converts a (numpy) transformation matrix into a java AffineTransform object
    Input:
        params: a parameter matrix
    Returns:
        A java.awt.geom.AffineTransform
    """
    p = params
    if isinstance(p, str):
        p = eval(p)
    return AffineTransform(p[0][0], p[1][0], p[0][1], p[1][1], p[0][2], p[1][2])

def affinetransform_to_params(at):
    x = list(range(6))
    at.getMatrix(x)
    p = np.empty((3,3))
    p[2, :] = [0, 0, 1]
    p[0, 0], p[1, 0], p[0, 1], p[1, 1], p[0,2], p[1, 2] = x
    return p

class PatchFilter(PythonJavaClass):
    """
    The RegularizedAffineLayerAlignment needs an object that implements the Filter interface as an argument.
    This class implements implements a filter which filters out Patches with the isVisible attribute set to False
    """
    __javainterfaces__ = ['ini/trakem2/utils/Filter', ]

    def __init__(self):
        super(PatchFilter, self).__init__()

    @signatures.with_signature(signatures.jboolean, [JObject])
    def accept(self, patch):
        return patch.isVisible()

