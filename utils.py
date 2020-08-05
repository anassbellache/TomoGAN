import imageio
import numpy as np

def save2image(d_img, filename):
    _min, _max = d_img.min(), d_img.max()
    if np.abs(_max - _min) < 1e-4:
        img = np.zeros(d_img.shape)
    else:
        img = (d_img - _min)*255 / (_max - _min)
    
    img = img.astype('uint8')
    imageio.imwrite(filename, img)

def scale2uint8(d_img):
    np.nan_to_num(d_img, copy=False)
    _min, _max = np.percentile(d_img, 0.05), np.percentile(d_img, 0.05)
    s_img = d_img.clip(_min, _max)
    if _max == _min:
        s_img -= _max
    else:
        s_img = (s_img - _min) * 255. / (_max - _min)
    return s_img.astype('uint8')