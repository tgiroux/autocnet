import numpy as np
from scipy.spatial.distance import cdist

from autocnet.matcher.feature import FlannMatcher
from autocnet.matcher.feature_matcher import match
from autocnet.transformation.decompose import coupled_decomposition


def decompose(self, subset=False, k=2, maxiteration=2, size=18,
              buf_dist=3, ndv=None, **kwargs):
    """
    Similar to match, this method first decomposed the image into
    $4^{maxiteration}$ subimages and applys matching between each sub-image.

    This method is potential slower than the standard match due to the
    overhead in matching, but can be significantly more accurate.  The
    increase in accuracy is a function of the total image size.  Suggested
    values for maxiteration are provided below.

    Parameters
    ----------
    k : int
        The number of neighbors to find

    maxiteration : int
                   When using coupled decomposition, the number of recursive
                   divisions to apply.  The total number of resultant
                   sub-images will be 4 ** maxiteration.  Approximate values:

                    | Number of megapixels | maxiteration |
                    |----------------------|--------------|
                    | m < 10               |1-2|
                    | 10 < m < 30          | 3 |
                    | 30 < m < 100         | 4 |
                    | 100 < m < 1000       | 5 |
                    | m > 1000             | 6 |

    size : int
           When using coupled decomposition, the total number of points
           to check in each sub-image to try and find a match.
           Selection of this number is a balance between seeking a
           representative mid-point and computational cost.

    buf_dist : int
               When using coupled decomposition, the distance from the edge of
               the (sub)image a point must be in order to be used as a
               partioning point.  The smaller the distance, the more likely
               percision errors can results in erroneous partitions.

    ndv : float
          The no data value that will be masked when computing the radial
          correlation.
    """

    sdata = self.source.get_array()
    ddata = self.destination.get_array()

    ssize = sdata.shape
    dsize = ddata.shape

    sdata[sdata == ndv] = np.nan
    ddata[ddata == ndv] = np.nan

    matches, _ = self.clean(['ratio'])
    sidx = matches['source_idx']
    didx = matches['destination_idx']

    # Grab all the available candidate keypoints
    skp = self.source.get_keypoints().loc[sidx]
    dkp = self.destination.get_keypoints().loc[didx]

    # Set up the membership arrays
    self.smembership = np.zeros(sdata.shape, dtype=np.int16)
    self.dmembership = np.zeros(ddata.shape, dtype=np.int16)
    self.smembership[:] = -1
    self.dmembership[:] = -1
    pcounter = 0
    for k in range(maxiteration):
        partitions = np.unique(self.smembership)
        npartitions = len(partitions)
        for p in partitions:
            sy_part, sx_part = np.where(self.smembership == p)
            dy_part, dx_part = np.where(self.dmembership == p)

            # Get the source extent
            minsy = np.min(sy_part)
            maxsy = np.max(sy_part) + 1
            minsx = np.min(sx_part)
            maxsx = np.max(sx_part) + 1

            # Get the destination extent
            mindy = np.min(dy_part)
            maxdy = np.max(dy_part) + 1
            mindx = np.min(dx_part)
            maxdx = np.max(dx_part) + 1

            # Clip the sub image from the full images (this is a MBR)
            asub = sdata[minsy:maxsy, minsx:maxsx]
            bsub = ddata[mindy:maxdy, mindx:maxdx]

            # Approximate the mid point of the partition as the mean of the matched keypoints
            sub_skp = skp.query('x >= {} and x <= {} and y >= {} and y <= {}'.format(minsx, maxsx,
                                                                                 minsy, maxsy))
            smx, smy = sub_skp[['x', 'y']].mean()
            mid = np.array([[smx, smy]])
            dists = cdist(mid, sub_skp[['x', 'y']])

            closest = sub_skp.iloc[np.argmin(dists)]
            closest_idx = closest.name
            soriginx, soriginy = closest[['x', 'y']]

            # Grab the corresponding point in the destination
            dest_idx = matches[matches['source_idx'] == closest_idx]['destination_idx']
            dest_pt = dkp.loc[dest_idx]
            doriginx, doriginy = dest_pt[['x', 'y']].values[0]

            # Sub image origin is assumed to be 0,0 (local sub-image space), while match point origins are
            # in the full image space.  Shift the match point orign to be in the sub-image space if needed
            soriginx -= minsx
            soriginy -= minsy
            doriginx -= mindx
            doriginy -= mindy

            # Apply coupled decomposition
            s_submembership, d_submembership = coupled_decomposition(asub, bsub,
                                                                     sorigin=(soriginx, soriginy),
                                                                     dorigin=(doriginx, doriginy))

            # Shift the returned membership counters to a set of unique numbers
            s_submembership += pcounter
            d_submembership += pcounter

            self.smembership[minsy:maxsy,
                        minsx:maxsx] = s_submembership

            sdy = dy_part - min(dy_part)
            sdx = dx_part - min(dx_part)
            self.dmembership[dy_part, dx_part] = d_submembership[sdy, sdx]
            pcounter += 4


def decompose_and_match(self, **kwargs):

    decompose(self, **kwargs)

    # Now match the decomposed segments to one another
    for p in np.unique(self.smembership):
        sy_part, sx_part = np.where(self.smembership == p)
        dy_part, dx_part = np.where(self.dmembership == p)

        # Get the source extent
        minsy = np.min(sy_part)
        maxsy = np.max(sy_part) + 1
        minsx = np.min(sx_part)
        maxsx = np.max(sx_part) + 1

        # Get the destination extent
        mindy = np.min(dy_part)
        maxdy = np.max(dy_part) + 1
        mindx = np.min(dx_part)
        maxdx = np.max(dx_part) + 1

        # Get the indices of the candidate keypoints within those regions / variables are pulled before decomp.
        sidx = skp.query('x >= {} and x <= {} and y >= {} and y <= {}'.format(minsx, maxsx, minsy, maxsy)).index
        didx = dkp.query('x >= {} and x <= {} and y >= {} and y <= {}'.format(mindx, maxdx, mindy, maxdy)).index
        # If the candidates < k, OpenCV throws an error
        if len(sidx) >= k and len(didx) >=k:
            match(self, aidx=sidx, bidx=didx)
            match(self, aidx=didx, bidx=sidx)
