import skimage
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.color import label2rgb
from skimage.measure import label, regionprops
import numpy as np

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimgfrom skimage.color import label2rgbfrom skimage.color import label2rgb


def label_and_region_prop(image):
    np_view = image.to_numpy(copy=False)
    # labeled = label(np_view, connectivity=np_view.ndim)
    return regionprops(np_view, cache=True)

# A

def area(image):
    rp_instance = label_and_region_prop(image)
    areas = [prop.area for prop in rp_instance]
    return np.asarray(areas)


def area_bbox(image):
    rp_instance = label_and_region_prop(image)
    areas = [prop.area_bbox for prop in rp_instance]
    return np.asarray(areas)


def area_convex(image):
    rp_instance = label_and_region_prop(image)
    areas = [prop.area_convex for prop in rp_instance]
    return np.asarray(areas)


def area_filled(image):
    rp_instance = label_and_region_prop(image)
    areas = [prop.area_filled for prop in rp_instance]
    return np.asarray(areas)


def axis_major_length(image):
    rp_instance = label_and_region_prop(image)
    areas = [prop.axis_major_length for prop in rp_instance]
    return np.asarray(areas)


def axis_minor_length(image):
    rp_instance = label_and_region_prop(image)
    areas = [prop.axis_minor_length for prop in rp_instance]
    return np.asarray(areas)


# E
def eccentricity(image):
    rp_instance = label_and_region_prop(image)
    props = [prop.eccentricity for prop in rp_instance]
    return np.asarray(props)


def equivalent_diameter_area(image):
    rp_instance = label_and_region_prop(image)
    props = [prop.equivalent_diameter_area for prop in rp_instance]
    return np.asarray(props)


def euler_number(image):
    rp_instance = label_and_region_prop(image)
    props = [prop.euler_number for prop in rp_instance]
    return np.asarray(props)


def extent(image):
    rp_instance = label_and_region_prop(image)
    props = [prop.extent for prop in rp_instance]
    return np.asarray(props)


# F
def feret_diameter_max(image):
    rp_instance = label_and_region_prop(image)
    props = [prop.feret_diameter_max for prop in rp_instance]
    return np.asarray(props)


# I
# def intensity_max(image):
#     rp_instance = label_and_region_prop(image)
#     props = [prop.intensity_max for prop in rp_instance]
#     return np.asarray(props)


# def intensity_mean(image):
#     rp_instance = label_and_region_prop(image)
#     props = [prop.intensity_mean for prop in rp_instance]
#     return np.asarray(props)


# def intensity_min(image):
#     rp_instance = label_and_region_prop(image)
#     props = [prop.intensity_min for prop in rp_instance]
#     return np.asarray(props)


# def intensity_std(image):
#     rp_instance = label_and_region_prop(image)
#     props = [prop.intensity_std for prop in rp_instance]
#     return np.asarray(props)


# N
def num_pixels(image):
    rp_instance = label_and_region_prop(image)
    props = [prop.num_pixels for prop in rp_instance]
    return np.asarray(props)


# O
def orientation(image):
    rp_instance = label_and_region_prop(image)
    props = [prop.orientation for prop in rp_instance]
    return np.asarray(props)


# P
def perimeter(image):
    rp_instance = label_and_region_prop(image)
    props = [prop.perimeter for prop in rp_instance]
    return np.asarray(props)


def perimeter_crofton(image):
    rp_instance = label_and_region_prop(image)
    props = [prop.perimeter_crofton for prop in rp_instance]
    return np.asarray(props)


# S
def solidity(image):
    rp_instance = label_and_region_prop(image)
    props = [prop.solidity for prop in rp_instance]
    return np.asarray(props)
