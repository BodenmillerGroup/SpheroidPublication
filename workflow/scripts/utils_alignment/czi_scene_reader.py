import czifile

class CziSceneReader(czifile.CziFile):
    """
    Reads only a determined scene from a czifile.
    Otherwise the interface is identical to the CziFile class
    """
    def __init__(self, *args, scene=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.scene = scene

    def get_filtered_subblock_directory(self):
        """Return sorted list of DirectoryEntryDV filtered for the current scene"""
        scene = self.scene
        super().filtered_subblock_directory
        x = self._filtered_subblock_directory
        filtered = [directory_entry
                    for directory_entry in x
                    if is_scene(directory_entry, scene)]
        return filtered

    def set_filtered_subblock_directory(self, x):
        """Return sorted list of DirectoryEntryDV if mosaic, else all."""
        self._filtered_subblock_directory = x

    filtered_subblock_directory = property(get_filtered_subblock_directory, set_filtered_subblock_directory)


def is_scene(directory_entry, scene):
    for de in directory_entry.dimension_entries:
        if (de.dimension == 'S') and (de.start == scene):
            return True
    else:
        return False

def get_czi_nscenes(fn_czi):
    """ Determine the number of scenes in a czi image """
    with czifile.CziFile(fn_czi) as czi:
        nscenes = czi.shape[1]
    return nscenes
