import pathlib

def symlink_folders(src, target, mkdirs=False):
    """
    Creating/linking a folder on a data drive (src)
    and symlink it to the repository (target)

    Args:
        src: Folder on the data drive
        target: Folder in the repository
        mkdirs: should the src folder be generated?

    Returns:
        None
    """

    src = pathlib.Path(src)
    target = pathlib.Path(target)

    if mkdirs:
        src.mkdir(parents=True, exist_ok=True)
    target.symlink_to(src, target_is_directory=True)


