def make_patch_grid(image_size, patch_size):
    """
    Returns:
        nb_horizontal_patches: number of patch rows
        nb_vertical_patches: number of patch columns
        coords: list of (row, col) tuples, one per patch index
    """
    if not isinstance(image_size, (tuple, list)):
        image_size = (image_size, image_size)

    w, h = image_size
    if h % patch_size != 0 or w % patch_size != 0:
        raise Exception("image size not compatible with patch size")

    nb_vertical_patches = h // patch_size
    nb_horizontal_patches = w // patch_size

    coords = [(i, j) for j in range(nb_vertical_patches) for i in range(nb_horizontal_patches)]
    return nb_vertical_patches, nb_horizontal_patches, coords