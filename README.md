# Content-aware Image Scaling

According to the description from Adobe Photoshop user guide:

>Content-Aware Scale resizes an image without changing important visual content such as people, buildings, animals, and so forth. While normal scaling affects all pixels uniformly when resizing an image, content-aware scaling mostly affects pixels in areas that don’t have important visual content. `

This repo implemented the Content-aware Scaling features, which allows users to upscale or downscale images to improve a composition, fit a layout, or change the orientation.

### Sample Outputs
![Napoleon](https://media.giphy.com/media/fLpOZ9kNdzICVqGR5d/giphy.gif)

![Lake](https://media.giphy.com/media/tZqWpbZiqn81OTm4db/giphy.gif)


### Requirements
* Python: 2.7
* OpenCV2: 2.4.13.4
* Numpy: 1.15.1
* Matplotlib: 2.1.1


### Usage
The following code snippet will downscale an image from size `(row, col)` to `(row, col-100)`.
```
import matplotlib.pyplot as plt
import seam_carving
import seam_expansion

org = "./[Your Image]"
scale_down_pixel_num = 100

org_img = cv2.imread(org, cv2.IMREAD_COLOR)
org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
y_seams, carved_img = seam_carving.cal_multi_seams(org_img, scale_down_pixel_num)

plt.imshow(org_img);
plt.imshow(carved_img);
```

### Acknowledgments
1. [Seam Carving for Content-Aware Image Resizing](https://perso.crans.org/frenoy/matlab2012/seamcarving.pdf). Shai Avidan, Ariel Shamir, 2007.
2. [Resize images and protect content](https://helpx.adobe.com/nz/photoshop/using/content-aware-scaling.html). Adobe Photoshop User Guide.
