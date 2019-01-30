import numpy as np

def rgb2gray(rgb):
    result = np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    return result

def waterfall(upper_seams, current_seam):
    """
    Helper Function for "seam_shiftback"
    For upper_seams = np.array([seam1, seam2, seam3, seam4]) and current_seam = seam5
    The waterfall function will compare from bottom to top, e.g.,
    compare (seam5, seam4), then (seam5, seam3), then (seam5, seam2), then (seam5, seam1)

    For each comparison, the cuurent_seam (seam5) will be corrected base on whom it compairs with.
    So in this case, seam5 will be corrected 4 times until it reaches the top (seam1).
    It has to be done in reversed order.
    """
    assert type(upper_seams) == np.ndarray
    for upper in upper_seams[::-1]:
        current_seam[current_seam >= upper] += 1
    return current_seam

def seam_shiftback(seam_list):
    """
    Intake a list of seam (type=list), and sequentially correct each seam.
    In the helper function "waterfall", it will correct single seam only.
    e.g., waterfall([seam1, seam2, seam3], seam4) will make seam4 being corrected.

    To get a overall shiftback seams, you have to apply waterfall to each seam.
    Return list is a np.array.
    """
    seam_array = np.array(seam_list)
    for i in range(seam_array.shape[0])[::-1]:
        seam_array[i] = waterfall(seam_array[:i], seam_array[i])
    return seam_array

def expansion_shiftback(corrected_seam_list, expand_seam_ind):
    """
    Each time the pixels along the "expand_seam" are replaced by left & right average,
    a.k.a the image is 1-pixel larger, all the seams in seam_list have to be shifted accordingly.

    For a corrected_seam_list = [seam1, seam2, seam3, seam4, ... seamN]
    If I'd like to expand pixels along seam2, then exapnd_seam_ind is 1.
    Each of the rest seams in corrected_seam_list [seam1, seam3, seam4, ...] 
    will be corrected one by one.
    """
    target_seam = np.array([corrected_seam_list[expand_seam_ind],])
    todo_list = np.delete(corrected_seam_list, expand_seam_ind, axis=0)
    out = [waterfall(target_seam, seam) for seam in todo_list]
    if len(out) == 0:
        return corrected_seam_list
    out = np.insert(out, expand_seam_ind, target_seam, axis=0)
    assert out.shape == corrected_seam_list.shape
    return out
