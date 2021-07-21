import cv2
from time import daylight
import numpy as np
# from numpy.matrixlib import defmatrix
from skimage import transform as tf
from scipy.ndimage import affine_transform

from topocalc.core_c import topo_core
from topocalc.skew import adjust_spacing, skew


def skew_transpose(dem, dx, angle, dy=None):
    """Skew and transpose the dem for the given angle.
    Also calculate the new spacing given the skew.

    Arguments:
        dem {array} -- numpy array of dem elevations
        dx {float} -- grid spacing in E-W direction
        dy {float} -- grid spacing in N-S direction
        angle {float} -- skew angle

    Returns:
        t -- skew and transpose array
        spacing -- new N-S spacing adjusted for azimuth angle
    """

    if dy is None:
        dy = dx

    spacing = adjust_spacing(dy, np.abs(angle))
    t = skew(dem, angle, dx=dx, dy=dy, fill_min=True).transpose()

    return t, spacing


def transpose_skew(dem, dx, angle, dy=None):
    """Transpose, skew then transpose a dem for the
    given angle. Also calculate the new spacing

    Arguments:
        dem {array} -- numpy array of dem elevations
        dx {float} -- grid spacing in E-W direction (columns)
        dy {float} -- grid spacing in N-S direction (rows)
        angle {float} -- skew angle

    Returns:
        t -- skew and transpose array
        spacing -- new E-W spacing adjusted for azimuth angle
    """

    if dy is None:
        dy = dx

    # spacing adjustment is in x direction for transposed array
    spacing = adjust_spacing(dx, np.abs(angle))
    # taking skew of transpose - need to swap dx and dy
    t = skew(dem.transpose(), angle, dx=dy, dy=dx, fill_min=True).transpose()

    return t, spacing


def horizon(azimuth, dem, dx, dy=None):
    """Calculate horizon angles for one direction. Horizon angles
    are based on Dozier and Frew 1990 and are adapted from the
    IPW C code.

    The coordinate system for the azimuth is 0 degrees is South,
    with positive angles through East and negative values
    through West. Azimuth values must be on the -180 -> 0 -> 180
    range.

    Arguments:
        azimuth {float} -- find horizon's along this direction
        dem {np.array2d} -- numpy array of dem elevations
        dx {float} -- grid spacing in E-W direction
        dy {float} -- grid spacing in N-S direction

    Returns:
        hcos {np.array} -- cosines of angles to the horizon
    """

    if dem.ndim != 2:
        raise ValueError('horizon input of dem is not a 2D array')

    if azimuth > 180 or azimuth < -180:
        raise ValueError('azimuth must be between -180 and 180 degrees')

    if dy is None:
        dy = dx

    spacing = dx

    if azimuth == 90:
        # East
        hcos = hor2d_c(dem, dx, fwd=True)

    elif azimuth == -90:
        # West
        hcos = hor2d_c(dem, dx, fwd=False)

    elif azimuth == 0:
        # South
        hcos = hor2d_c(dem.transpose(), dy, fwd=True)
        hcos = hcos.transpose()

    elif np.abs(azimuth) == 180:
        # North
        hcos = hor2d_c(dem.transpose(), dy, fwd=False)
        hcos = hcos.transpose()

    elif azimuth >= -45 and azimuth <= 45:
        # South west through south east
        t, spacing = skew_transpose(dem, dx, azimuth, dy=dy)
        h = hor2d_c(t, spacing, fwd=True)
        hcos = skew(h.transpose(), azimuth, dx=dx, dy=dy, fwd=False)

    elif azimuth <= -135 and azimuth > -180:
        # North west
        a = azimuth + 180
        t, spacing = skew_transpose(dem, dx, a, dy=dy)
        h = hor2d_c(t, spacing, fwd=False)
        hcos = skew(h.transpose(), a, dx=dx, dy=dy, fwd=False)

    elif azimuth >= 135 and azimuth < 180:
        # North East
        a = azimuth - 180
        t, spacing = skew_transpose(dem, dx, a, dy=dy)
        h = hor2d_c(t, spacing, fwd=False)
        hcos = skew(h.transpose(), a, dx=dx, dy=dy, fwd=False)

    elif azimuth > 45 and azimuth < 135:
        # South east through north east
        a = 90 - azimuth
        t, spacing = transpose_skew(dem, dx, a, dy=dy)
        h = hor2d_c(t, spacing, fwd=True)
        hcos = skew(h.transpose(), a, dx=dy, dy=dx, fwd=False).transpose()

    elif azimuth < -45 and azimuth > -135:
        # South west through north west
        a = -90 - azimuth
        t, spacing = transpose_skew(dem, dx, a, dy=dy)
        h = hor2d_c(t, spacing, fwd=False)
        hcos = skew(h.transpose(), a, dx=dy, dy=dx, fwd=False).transpose()

    else:
        ValueError('azimuth not valid')

    # sanity check
    assert hcos.shape == dem.shape

    return hcos

def skew_interp(dem, azimuth, dx, dy, fwd=True):

    # squish the input DEM by the dy/dx difference
    scale_matrix = np.array([[dy/dx,0,0],[0,1,0],[0,0,1]])

    # shear by an angle on the now-square dem array
    # shear takes values of radians counter-clockwise
    shear_matrix = np.array([[1, -np.tan(np.radians(azimuth)), 0],
                                [0, 1, 0],
                                [0, 0, 1]])

    # re-stretch the dem array back to its original dy/dx dimensions
    unscale_matrix = np.array([[dx/dy,0,0],[0,1,0],[0,0,1]])

    scale_tf = tf.AffineTransform(scale=(dx/dy,1))
    shear_tf = tf.AffineTransform(matrix=shear_matrix)
    unscale_tf = tf.AffineTransform(scale=(dy/dx,1))

    # shift sheared matrix to fit within calculated output shape
    # only needed if azimuth angle is positive
    if azimuth > 0:
        if fwd:
            trans_tf = tf.AffineTransform(translation=\
                (dem.shape[1]*abs(np.tan(np.radians(azimuth))),0))
        else:
            # if reversing, need to calculate translate 
            # relative to original (unskewed) size
            orig_x = dem.shape[1]/(1+abs(np.tan(np.radians(azimuth))))
            trans_tf = tf.AffineTransform(translation=\
                (orig_x*abs(np.tan(np.radians(azimuth))),0))
    else:
        trans_tf = tf.AffineTransform()

    # compose the transformations together:
    full_tf = (scale_tf + (shear_tf + (unscale_tf + trans_tf)))

    # openCV version:
    if fwd==True:
        # compose the transformations together:
        full_tf = (scale_tf + (shear_tf + (unscale_tf + trans_tf)))

        # get shape of output array after skewing
        outshape = (round(dem.shape[0]),
                    round(dem.shape[1]*\
                        (1+abs(np.tan(np.radians(azimuth))))))

        skew_t = cv2.warpAffine(dem, full_tf.params[:2],
                                (outshape[1],outshape[0]),
                                flags=cv2.INTER_LINEAR)
        return skew_t

    else:
        # compose the transformations together:
        full_tf = (scale_tf + (shear_tf + (unscale_tf + trans_tf)))

        outshape = (round(dem.shape[0]),
                    round(dem.shape[1]/\
                        (1+abs(np.tan(np.radians(azimuth))))))

        unskew_t = cv2.warpAffine(dem, full_tf.params[:2],
                                  (outshape[1],outshape[0]), 
                                  flags=cv2.WARP_INVERSE_MAP+cv2.INTER_LINEAR)
        return unskew_t

    # skimage version:
    # skew_t = tf.warp(dem, full_tf.inverse, 
    #             order=1, mode='constant', output_shape=outshape)
    # unskew_t = tf.warp(skew_t, full_tf,
    #             order=1, mode='constant',output_shape=dem.shape)

    #print('Just checking this is actually loading...')

    #return skew_t,unskew_t

def skew_transpose_interp(dem, dx, angle, dy=None):
    """Skew and transpose the dem for the given angle.
    Also calculate the new spacing given the skew.

    Arguments:
        dem {array} -- numpy array of dem elevations
        dx {float} -- grid spacing in E-W direction
        dy {float} -- grid spacing in N-S direction
        angle {float} -- skew angle

    Returns:
        t -- skew and transpose array
        spacing -- new N-S spacing adjusted for azimuth angle
    """

    if dy is None:
        dy = dx

    spacing = dy / np.cos(np.abs(angle) * np.arctan(1.) / 45)
    t = skew_interp(dem, angle, dx=dx, dy=dy, fwd=True).transpose()

    return t, spacing

def transpose_skew_interp(dem, dx, angle, dy=None):
    """Transpose, skew then transpose a dem for the
    given angle. Also calculate the new spacing

    Arguments:
        dem {array} -- numpy array of dem elevations
        dx {float} -- grid spacing in E-W direction (columns)
        dy {float} -- grid spacing in N-S direction (rows)
        angle {float} -- skew angle

    Returns:
        t -- skew and transpose array
        spacing -- new E-W spacing adjusted for azimuth angle
    """

    if dy is None:
        dy = dx

    # spacing adjustment is in x direction for transposed array
    spacing = dx / np.cos(np.abs(angle) * np.arctan(1.) / 45)
    # taking skew of transpose - need to swap dx and dy
    t = skew_interp(dem.transpose(), angle, dx=dy, dy=dx, fwd=True).transpose()

    return t, spacing


def horizon_interp(azimuth, dem, dx, dy=None):
    """Calculate horizon angles for one direction, interpolating
    between points to increase fidelity for small angles near 
    cardinal directions.

    The coordinate system for the azimuth is 0 degrees is South,
    with positive angles through East and negative values
    through West. Azimuth values must be on the -180 -> 0 -> 180
    range.

    Arguments:
        azimuth {float} -- find horizon's along this direction
        dem {np.array2d} -- numpy array of dem elevations
        dx {float} -- grid spacing in E-W direction
        dy {float} -- grid spacing in N-S direction

    Returns:
        hcos {np.array} -- cosines of angles to the horizon
    """
    if dem.ndim != 2:
        raise ValueError('horizon input of dem is not a 2D array')

    if azimuth > 180 or azimuth < -180:
        raise ValueError('azimuth must be between -180 and 180 degrees')

    if dy is None:
        dy = dx

    if azimuth == 90:
        # East
        hcos = hor2d_c(dem, dx, fwd=True)

    elif azimuth == -90:
        # West
        hcos = hor2d_c(dem, dx, fwd=False)

    elif azimuth == 0:
        # South
        hcos = hor2d_c(dem.transpose(), dy, fwd=True)
        hcos = hcos.transpose()

    elif np.abs(azimuth) == 180:
        # North
        hcos = hor2d_c(dem.transpose(), dy, fwd=False)
        hcos = hcos.transpose()
    
    elif azimuth >= -45 and azimuth <= 45:
        # South west through south east
        t, spacing = skew_transpose_interp(dem, dx, azimuth, dy=dy)
        h = hor2d_c(t, spacing, fwd=True)
        hcos = skew_interp(h.transpose(), azimuth, dx=dx, dy=dy, fwd=False)

    elif azimuth <= -135 and azimuth > -180:
        # North west
        a = azimuth + 180
        t, spacing = skew_transpose_interp(dem, dx, a, dy=dy)
        h = hor2d_c(t, spacing, fwd=False)
        hcos = skew_interp(h.transpose(), a, dx=dx, dy=dy, fwd=False)

    elif azimuth >= 135 and azimuth < 180:
        # North East
        a = azimuth - 180
        t, spacing = skew_transpose_interp(dem, dx, a, dy=dy)
        h = hor2d_c(t, spacing, fwd=False)
        hcos = skew_interp(h.transpose(), a, dx=dx, dy=dy, fwd=False)

    elif azimuth > 45 and azimuth < 135:
        # South east through north east
        a = 90 - azimuth
        t, spacing = transpose_skew_interp(dem, dx, a, dy=dy)
        h = hor2d_c(t, spacing, fwd=True)
        hcos = skew_interp(h.transpose(), a, dx=dy, dy=dx, fwd=False).transpose()

    elif azimuth < -45 and azimuth > -135:
        # South west through north west
        a = -90 - azimuth
        t, spacing = transpose_skew_interp(dem, dx, a, dy=dy)
        h = hor2d_c(t, spacing, fwd=False)
        hcos = skew_interp(h.transpose(), a, dx=dy, dy=dx, fwd=False).transpose()

    else:
        ValueError('azimuth not valid')

    # sanity check
    assert hcos.shape == dem.shape

    return hcos
    

def hor2d_c(z, spacing, fwd=True):
    """
    Calculate values of cosines of angles to horizons in 2 dimension,
    measured from zenith, from elevation difference and distance.  Let
    G be the horizon angle from horizontal and note that:

        sin G = z / sqrt( z^2 + dis^2);

    This result is the same as cos H, where H measured from zenith.

    Args:
        z: elevation array
        spacing: spacing of array

    Returns:
        hcos: cosines of angles to horizon
    """

    if z.ndim != 2:
        raise ValueError('hor1d input of z is not a 2D array')

    if z.dtype != np.double:
        raise ValueError('hor1d input of z must be a double')

    spacing = np.double(spacing)

    z = np.ascontiguousarray(z)

    h = np.zeros_like(z)

    topo_core.c_hor2d(z, spacing, fwd, h)

    # if not fwd:
    #     h = np.fliplr(h)

    return h


def pyhorizon(dem, dx):
    """Pure python version of the horizon function.

    NOTE: this is fast for small dem's but quite slow
    for larger ones. This is mainly to show that it
    can be done with numpy but requires a bit more to
    remove the for loop over the rows. Also, this just
    calculates the horizon in one direction, need to implement
    the rest of the horizon function for calcuating the
    horizon at an angle.

    Args:
        dem (np.ndarray): dem for the horizon
        dx (float): spacing for the dem

    Returns:
        [tuple]: cosine of the horizon angle and index
            to the horizon.
    """

    # needs to be a float
    if dem.dtype != np.float64:
        dem = dem.astype(np.float64)

    nrows, ncols = dem.shape
    hcos = np.zeros_like(dem)
    horizon_index = np.zeros_like(dem)

    # distance to each point
    # k=-1 because the distance to the point itself is 0
    distance = dx * np.cumsum(np.tri(ncols, ncols, k=-1), axis=0)
    col_index = np.arange(0, ncols)

    for n in range(nrows):
        surface = dem[n, :]

        m = np.repeat(surface.reshape(1, -1), ncols, axis=0)

        # height change
        height = np.tril(m.T - m)

        # slope
        slope = height / distance

        # horizon location
        hor = np.nanargmax(slope[:, :-1], axis=0)
        hor = np.append(hor, ncols-1)
        hidx = hor.astype(int)

        horizon_height_diff = surface[hidx] - surface
        horizon_distance_diff = dx * (hor - col_index)

        new_horizon = horizon_height_diff / \
            np.sqrt(horizon_height_diff**2 + horizon_distance_diff**2)

        new_horizon[new_horizon < 0] = 0
        new_horizon[np.isnan(new_horizon)] = 0

        hcos[n, :] = new_horizon
        horizon_index[n, :] = hidx

    return hcos, horizon_index
