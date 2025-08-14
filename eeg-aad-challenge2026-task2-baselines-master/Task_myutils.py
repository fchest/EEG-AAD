from __future__ import print_function
import random
import numpy as np
import math
import task_loader

np.set_printoptions(suppress=True)
import os
import time
import torch
from sklearn.preprocessing import scale
from scipy.interpolate import griddata
from scipy.io import loadmat


def cart2sph(x, y, z):
    """
    Transform Cartesian coordinates to spherical
    :param x: X coordinate
    :param y: Y coordinate
    :param z: Z coordinate
    :return: radius, elevation, azimuth
    """
    x2_y2 = x ** 2 + y ** 2
    r = math.sqrt(x2_y2 + z ** 2)  # r
    elev = math.atan2(z, math.sqrt(x2_y2))  # Elevation
    az = math.atan2(y, x)  # Azimuth
    return r, elev, az


def pol2cart(theta, rho):
    """
    Transform polar coordinates to Cartesian
    :param theta: angle value
    :param rho: radius value
    :return: X, Y
    """
    return rho * math.cos(theta), rho * math.sin(theta)


def makePath(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def monitor(process, multiple, second):
    while True:
        sum = 0
        for ps in process:
            if ps.is_alive():
                sum += 1
        if sum < multiple:
            break
        else:
            time.sleep(second)


def save_load_name(args, name=''):
    name = name if len(name) > 0 else 'default_model'
    return name


def save_model(args, model, name=''):
    name = save_load_name(args, name)
    torch.save(model, f'pre_trained_models/{name}.pt')


def load_model(args, name=''):
    name = save_load_name(args, name)
    model = torch.load(f'pre_trained_models/{name}.pt')
    return model


def getData(name="s1", time_len=1, dataset="MM-AAD"):
    MMAAD_document_path1 = "/media/gaoyoudian/DXG/dataset_v2/pre_aoall"   #replace your data_path for training
    MMAAD_document_path2 = "/media/gaoyoudian/DXG/dataset_v2/pre_avall"    #replace your data_path for validation
    if dataset == 'MM-AAD':
        return task_loader.get_MMAAD_data(name, time_len, MMAAD_document_path1, MMAAD_document_path2)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.

    :param pos: position in 3D Cartesian coordinates
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, math.pi / 2 - elev)


