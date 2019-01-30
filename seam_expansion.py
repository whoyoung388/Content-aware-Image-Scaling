import numpy as np

def search_seam_pixel(origin_img, seam):
    """
    Trace back the seam to find each pixel RGB value on original image.
    You must use shiftted seam to do the search.
    By saying original image, I mean the very first image that being used.
    Not any interval stage image, like has been carved or expanded.
    It will not return correct result if you don't use the original image.

    * Parameters
        @origin_img: 3-d ndarray
        @seam: the 1-d array contains only y-coordinate of the seam path

    * Returns
        @pixels: 1-d array [(R0, G0, B0), (R1, G1, B1), ..., (Rn, Gn, Bn)]
                 where n = len(seam)
    """
    pixels = [origin_img[cur_x, cur_y, :] for cur_x, cur_y in enumerate(seam)]
    return np.array(pixels).astype(np.uint8)

def seam_pixel_plot(color_img, seam, pixels):
    """
    Insert a series of pixels along the path of seam.
    """
    red, green, blue = np.rollaxis(color_img, 2)
    r, g, b = zip(*pixels)
    r_insert = np.array([np.insert(row, seam[i], r[i]) for i, row in enumerate(red)])
    g_insert = np.array([np.insert(row, seam[i], g[i]) for i, row in enumerate(green)])
    b_insert = np.array([np.insert(row, seam[i], b[i]) for i, row in enumerate(blue)])
    restored_img = np.dstack([r_insert, g_insert, b_insert])
    return  restored_img
