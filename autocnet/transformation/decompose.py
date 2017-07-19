import numpy as np
from scipy.stats import pearsonr


def cart2polar(x, y):
    theta = np.arctan2(y, x)
    return -theta

def index_coords(data, origin=None):
    """Creates x & y coords for the indicies in a numpy array "data".
    "origin" defaults to the center of the image. Specify origin=(0,0)
    to set the origin to the lower left corner of the image."""
    ny, nx = data.shape[:2]
    if origin is None:
        origin_x, origin_y = nx // 2, ny // 2
    else:
        origin_x, origin_y = origin
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x -= origin_x
    y -= origin_y
    return x, y

def reproject_image_into_polar(data, origin=None):
    """Reprojects a 3D numpy array ("data") into a polar coordinate system.
    "origin" is a tuple of (x0, y0) and defaults to the center of the image."""
    ny, nx = data.shape[:2]
    if origin is None:
        origin = (nx//2, ny//2)

    # Determine that the min and max r and theta coords will be...
    x, y = index_coords(data, origin=origin)
    theta = cart2polar(x, y)
    theta[theta < 0] += 2 * np.pi
    return theta

def coupled_decomposition(sdata, ddata, sorigin=(), dorigin=(), M=4, theta_steps=720, theta=None):
    """
    Apply coupled decomposition to two 2d images.

    sdata : ndarray
            (n,m) array of values to decompose

    ddata : ndarray
            (j,k) array of values to decompose

    sorigin : tuple
              in the form (x,y)

    dorigin : tuple
              in the form (x,y)

    """

    soriginx, soriginy = sorigin
    doriginx, doriginy = dorigin

    # Create membership arrays for each input image
    smembership = np.ones(sdata.shape)
    dmembership = np.ones(ddata.shape)

    # Project the image into a polar coordinate system centered on p_{1}
    stheta = reproject_image_into_polar(sdata, origin=(int(soriginx), int(soriginy)))
    dtheta = reproject_image_into_polar(ddata, origin=(int(doriginx), int(doriginy)))

    if theta == None:
        # Compute the mean profiles for each radial slice
        smean = np.empty(theta_steps)
        dmean = np.empty(theta_steps)
        radial_step = 2 * np.pi / theta_steps # 0.5 deg

        thetas = np.arange(0, 2 * np.pi, radial_step)
        for i, t in enumerate(thetas):
            smean[i] = np.nanmean(sdata[(stheta >= t) & (stheta <= t + radial_step)])
            dmean[i] = np.nanmean(ddata[(dtheta >= t) & (dtheta <= t + radial_step)])

        # Rotate the second image around the origin and compute the correlation coeff. for each 0.5 degree rotation.
        maxp = -1
        maxidx = 0
        dsearch=np.empty(theta_steps)
        for j in range(theta_steps):
            dsearch = np.concatenate((dmean[j:], dmean[:j]))
            r, _ = pearsonr(smean, dsearch)
            if r >= maxp:
                maxp = r
                maxidx = j

        # Maximum correlation (theta) defines the angle of rotation for the destination image

        theta = thetas[maxidx]

    # Classify the sub-images based on the decomposition size (M) and theta
    smembership[(stheta >= 0) & (stheta <= np.pi/2)] = 0
    smembership[(stheta >= np.pi/2) & (stheta <= np.pi)] = 1
    smembership[(stheta >=np.pi) & (stheta <= 3 * np.pi/2)] = 2
    smembership[(stheta >= 3 * np.pi/2) & (stheta <= 2 * np.pi)] = 3


    if 0 <= theta <= np.pi / 2:
        order = [0,1,2,3]
    elif np.pi/2 <= theta <= np.pi:
        order = [1,2,3,0]
    elif np.pi <= theta <= 3*np.pi/2:
        order = [2,3,0,1]
    elif 3*np.pi/2 <= theta <= 2*np.pi:
        order = [3,0,1,2]

    def wrap(v):
        return v % (2 * np.pi)

    def classify(start, stop, classid):
        start = wrap(start)
        stop = wrap(stop)
        if start > stop:
            dmembership[(dtheta >=start) & (dtheta <= 2*np.pi)] = classid
            dmembership[(dtheta >=0) & (dtheta <= stop)] = classid
        else:
            dmembership[(dtheta >= start) & (dtheta <= stop)] = classid

    classify(theta, theta + np.pi/2, 0)
    classify(theta + np.pi/2, theta + np.pi, 1)
    classify(theta + np.pi, theta + 3*np.pi/2,2)
    classify(theta + 3*np.pi/2, theta + 2*np.pi, 3)

    return smembership, dmembership
