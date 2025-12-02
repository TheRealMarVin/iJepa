def make_patch_grid(image_size, patch_size):
    """
    Args:
        image_size: (width, height) in pixels, or a single int for square images.
        patch_size: size of each square patch, in pixels.

    Returns:
        nb_vertical_patches: number of patch rows (height direction)
        nb_horizontal_patches: number of patch columns (width direction)
        coords: flat list of (row, col) tuples in row-major order
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