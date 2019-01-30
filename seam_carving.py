import numpy as np
import cv2

from util import rgb2gray

def map_gradient(gray_img):
    """
    Calculate the energy function
    """
    dy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1)
    dx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)
    energy = abs(dx) + abs(dy)
    assert energy.dtype == np.float64
    return energy

def select_parent_min(upper_row, y0):
    """
    * Parameters
        @upper_row
        @y0: if the coordinate on the img is img[x0, y0], then
             upper_row will always be img[x0 - 1, :]
             ---------------
             | | | | | | | |  upper_row
             ---------------
             |0|1|2|3|4|5|6|  y0 is the y coordinate
             ---------------
        e.g., if y0 = 0, parent = below three items with !
                 y0 = 3, parent = below three items with * 
                 y0 = 6, parent = below three items with &
             ---------------
             |!|!|*|*|*|&|&|
             ---------------
             |0|1|2|3|4|5|6|
             ---------------
    * Returns
        Select the min value & it's index from parent,
        A tuple of (min_value, min_y_coordinate)
    """
    parents = upper_row[max(y0-1, 0):y0+2]
    relative_y = parents.argmin()
    val = parents[relative_y]
    y = y0 + relative_y - (y0 != 0)
    return (val, y)

def cumulative_energy(energy):
    """
    Start from the second row (not the first row)
    
    * Parameters
        @energy: the img energy score (gradient) matrix

    * Returns
        @cumul_energies: the cumulative energy matrix
        @cumul_paths: the exacte index of choosen element
    """
    x, y = energy.shape
    cumul_energies = np.empty((x, y))
    cumul_energies[0] = energy[0]
    cumul_paths = np.empty((x, y), np.int32)
    cumul_paths[0] = np.ones(y) * (-1)
    
    for cur_x in range(1, x):
        update_upper_row = cumul_energies[cur_x - 1, :]
        path_energy, path = zip(*[select_parent_min(update_upper_row, cur_y) for cur_y in range(y)])
        cumul_energies[cur_x] = path_energy + energy[cur_x]
        cumul_paths[cur_x] = path
    return cumul_energies, cumul_paths

def search_seam(paths, y_end):
    """
    * Parameters
        @paths: cumulative paths return by cumulative_energy()
        @y_end: the y coordinate of the smallest value 
                in the last row of cumulative energies matrix

        if cumulative matrix.shape is (9, 5) with matrix:
        
        -------------------------
        | 0.  10.   6.  14.   0.|
        -------------------------
        |10.  20.  26.   4.  18.|
        -------------------------
        |10.  24.  22.  20.   4.|
        -------------------------
        |20.  34.  46.  12.  18.|
        -------------------------
        |32.  42.  22.  20.  12.|
        -------------------------
        |36.  44.  26.  16.  14.|
        -------------------------
        |44.  44.  24.  24.  18.|
        -------------------------
        |46.  32.  32.  26.  20.|
        -------------------------
        |32.  38.  36.  28.  20.|  <- the min value is 20.0 with y coor = 4
        -------------------------

        the corresponding cumulative paths matrix is:
        
        [[-1 -1 -1 -1 -1]
         [ 0  0  2  4  4]
         [ 0  0  3  3  3]
         [ 0  0  3  4  4]
         [ 0  0  3  3  3]
         [ 0  2  3  4  4]
         [ 0  2  3  4  4]
         [ 0  2  2  4  4]
         [ 1  1  3  4  4]], type=np.int32 (becasue the value will be passed to array slicing)
         
         which means the seam will be the following path denoted as "x":
         
         [[ o o o o x ]
          [ o o o x o ]
          [ o o o o x ]
          [ o o o x o ]
          [ o o o o x ]
          [ o o o o x ]
          [ o o o o x ]
          [ o o o o x ]
          [ o o o o x ]]

    * Returns
        @seam: ndarray(int64) with shape = the row number of input paths matrix
               Each element is the y coordinate of the pixel to be removed (seam) at that row e.g.
               array([4, 3, 4, 3, 4, 4, 4, 4, 4]) means
               "remove the pixels at (0,4), (1,3), (2,4), (3,3), (4,4), (5,4), (6,4), (7,4), (8,4)"

               Note: return seam is REVERSED, because it's searched backward from the bottom
                     as a result the first found seam belongs to last row of cumulative matrix
    """
    x, y = paths.shape
    seam = [int(y_end)]
    for cur_x in range(1, x)[::-1]:
        upper_seam = paths[cur_x, seam[-1]]
        seam.append(upper_seam)
    assert len(seam) == x
    return seam[::-1]

def remove_seam(color_img, seam):
    """
    * Parameters
        @color_img is a 3-d array with pixel size of (height, width, 3)
        @seam is a 1-d array containing the vertical seam's y coordinates.

    * Returns
        @img with seam removed, with size (height, width - 1, 3)
    """
    return np.array([np.delete(color_img[i], j, axis=0) for i, j in enumerate(seam)])

def img2seam(color_img):
    """
    To encapsulate the process of calculating energy function and blablabla.
    The input color_img is intact without any modification.
    Another words, the seam has been calculated but hasn't been applied to the img.
    Return y_seam, a Python list.
    """
    gray_img = rgb2gray(color_img)
    energies = map_gradient(gray_img)
    c_energies, c_paths = cumulative_energy(energies)
    y_end = c_energies[-1].argmin()
    y_seam = search_seam(c_paths, y_end)
    return y_seam

def cal_multi_seams(color_img, seam_number):
    """
    The function will conduct seam removing for n times where n=seam_number.
    While removing the seams, it will record each seam path.
    
    The returned seam_list is based on "relative" index, e.g.,
    if you apply it directly to original image, the result is incorrect.
    Use seam_shiftback function in util.py to convert the relative sys to absolute.

    * Parameters
        @color_img 3-d ndarray
        @seam_number int: the number of seams needed to be removed
    
    * Returns
        @seam_list is a 1-d Python list. Each element is another Python list.
         seam_list = [[seam1], [seam2], ..., [seamN]], type=list
        @last_img will be the final carved image,
          e.g., if seam_number = 10 and original input image has shape (n, m, 3),
          the returned last_img will have shape (n, m-10, 3)
    """
    seam_list = []
    temp_img = color_img.copy()
    for i in range(seam_number):
        temp_seam = img2seam(temp_img)
        seam_list.append(temp_seam)
        temp_img = remove_seam(temp_img, temp_seam)
    last_img = temp_img.copy()
    return seam_list, last_img

def seam_plot(color_img, seam_list):
    """
    You cannot use this to restore/expand an image.
    It only plots white (or any fixed color) lines.
    Instead, use seam_pixel_plot from seam_expansion.py to achieve the purpose.

    This function is useful for supporting bulk seam_list plot.
    
    Plot multiple seams on given image.
    Be aware that seam_list has to be shifted back already. 
    Do not use the output from cal_multi_seam.
    """
    _img = color_img.copy()
    number_of_seams = len(seam_list)
    rows = color_img.shape[0]
    x_seam = np.array(range(rows))
    seams = [np.array(zip(seam_list[i], x_seam), np.int32) for i in range(number_of_seams)]
    for seam in seams:
        cv2.polylines(_img, [seam], False, (255, 255, 255), 1)
    return _img
