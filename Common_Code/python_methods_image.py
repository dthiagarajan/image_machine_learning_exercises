from python_methods_general import *
from skimage.transform import rescale
from PIL import Image
import pickle
import imageio
import glob

common_transpose = lambda x: np.transpose(x, [0, 3, 1, 2])


# read image file
def read_img(filename):
    img = imageio.imread(filename)
    return img


# read  all image from a directory
def read_img_dir(dir, img_type):
    glob_type = "*." + img_type
    file_key = dir + glob_type
    imagesPathList = glob.glob(file_key)

    image_collect = []
    for imagePath in imagesPathList:
        image_collect.append(read_img(imagePath))

    return np.array(image_collect)


# read all image from a list of dirs
def read_img_dirlist(dirlist, img_type):
    img_box = []
    for dir in dirlist:
        all_img_in_dir = read_img_dir(dir, img_type)
        img_box.append(all_img_in_dir)

    return img_box


# assist function of size_filter
def shape_check(img_mat, d_1, d_2, d_3):
    mat_shape = img_mat.shape
    return (mat_shape[1] == d_1 and mat_shape[2] == d_2 and mat_shape[3] == d_3)


# filter the image list so all images have the same size
def size_filter(image_list):
    # getting dimension of l, w and c, dim 0 is the num of images
    standard_l = image_list[0].shape[1]
    standard_w = image_list[0].shape[2]
    standard_c = image_list[0].shape[3]

    sc = lambda img_l: shape_check(img_l, standard_l, standard_w, standard_c)

    return np.array([image_list[i] for i in range(0, len(image_list))
                     if sc(image_list[i])])


# process the images and concatenate into a mat
def process_and_concat(image_list, process_op=None):
    if process_op is None:
        processed_list = image_list
    else:
        processed_list = process_op(image_list)

    return np.concatenate(processed_list, axis=0)


#resize the images
def resize_img_mat(img_mat, scale_rate,
                   whether_alias=True, whether_readable=False):
    # if scale_rate >=1, do nothing
    if scale_rate >= 1:
        return img_mat

    # convert images and rescale as 8 bit int
    if whether_readable:
        resized_img_mat = [rescale(img_mat[i], scale_rate,
                           anti_aliasing=whether_alias,
                           preserve_range=True).astype(np.uint8)
                           for i in range(0, len(img_mat))]
    else:
        resized_img_mat = [rescale(img_mat[i], scale_rate,
                           anti_aliasing=whether_alias,
                           preserve_range=True)
                           for i in range(0, len(img_mat))]

    return np.array(resized_img_mat)


# crop an image
def crop_img(img, start_l, l, start_w, w):
    return img[start_l:start_l+l, start_w:start_w+w, :]


# crop a matrix of images
def crop_img_mat(img_mat, start_l, l, start_w, w):
    return img_mat[:, start_l:start_l+l, start_w:start_w+w, :]


# a function that returns four corners and center
def five_point_crop(img, crop_l, crop_w):
    img_l = img.shape[0]
    img_w = img.shape[1]

    # lambda to get the half
    half = lambda x: int(x / 2)

    up_left = crop_img(img, 0, crop_l, 0, crop_w)
    up_right = crop_img(img, 0, crop_l, img_w-crop_w, crop_w)
    down_left = crop_img(img, img_l-crop_l, crop_l, 0, crop_w)
    down_right = crop_img(img, img_l-crop_l, crop_l, img_w-crop_w, crop_w)
    center = crop_img(img, half(img_l)-half(crop_l), crop_l,
                           half(img_w)-half(crop_w), crop_w)

    img_dictionary = {"ul": up_left, "ur": up_right,
                      "dl": down_left, "dr": down_right,
                      "c": center}

    img_arr = np.array([up_left, up_right, down_left, down_right, center])

    return img_dictionary, img_arr


# a function that returns four corners and center for array of img
def five_point_crop_mat(img_mat, crop_l, crop_w):
    return np.array([five_point_crop(img_mat[i], crop_l, crop_w)[1]
                     for i in range(0, len(img_mat))])


# create a mirror image
def mirror(img):
    return img[:, ::-1, :]


# a function that returns four corners, center, and their mirror image
def five_point_crop_mirror(img, crop_l, crop_w):
    ori_set = five_point_crop(img, crop_l, crop_w)[1]
    mir_set = five_point_crop(mirror(img), crop_l, crop_w)[1]

    whole_set = []
    whole_set.extend(ori_set)
    whole_set.extend(mir_set)

    return np.array(whole_set)


# crop with mirror for an image matrix
def five_point_crop_mirror_mat(img_mat, crop_l, crop_w):
    return np.array([five_point_crop_mirror(img_mat[i], crop_l, crop_w)
                     for i in range(0, len(img_mat))])


# test for rescaling
'''
from PIL import Image
a = read_img_dirlist([pa_2, pa_1], "png")
b = resize_img_mat(a, 0.5, whether_readable=True)
print(a[0])
print("after")
print(b[0])
img = Image.fromarray(b[0], 'RGB')
img.show()
'''

# test for cropping
'''
from PIL import Image
a = read_img_dirlist([pa_2, pa_1], "png")
b = crop_img_mat(a, 1, 200, 1, 200)
print(a[0])
print("after")
print(b[0])
print("value compare")
print(a[0][1][1])
print(b[0][0][0])
img = Image.fromarray(b[0], 'RGB')
img.show()
'''

'''
#unit test for five point crops mirror
from PIL import Image
a = read_img_dirlist([pa_2, pa_1], "png")
mc_a = five_point_crop_mirror_mat(a, 100, 100)
print(mc_a.shape)
'''
