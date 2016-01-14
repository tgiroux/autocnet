import cv2
import numpy as np
import pandas as pd
from autocnet.graph.network import CandidateGraph


FLANN_INDEX_KDTREE = 1  # Algorithm to set centers,
DEFAULT_FLANN_PARAMETERS = dict(algorithm=FLANN_INDEX_KDTREE,
                                trees=3)

class FlannMatcher(object):
    """
    A wrapper to the OpenCV Flann based matcher class that adds
    metadata tracking attributes and methods.  This takes arbitrary
    descriptors and so should be available for use with any
    descriptor data stored as an ndarray.

    Attributes
    ----------
    image_indices : dict
                    with key equal to the train image index (returned by the DMatch object),
                    e.g. an integer array index
                    and value equal to the image identifier, e.g. the name

    image_index_counter : int
                          The current number of images loaded into the matcher
    """

    def __init__(self, flann_parameters=DEFAULT_FLANN_PARAMETERS):
        self._flann_matcher = cv2.FlannBasedMatcher(flann_parameters, {})
        self.image_indices = {}
        self.image_index_counter = 0

    def add(self, descriptor, key):
        """
        Add a set of descriptors to the matcher and add the image
        index key to the image_indices attribute

        Parameters
        ----------
        descriptor : ndarray
                     The descriptor to be added

        key : hashable
              The identifier for this image, e.g. the image name
        """
        self._flann_matcher.add([descriptor])
        self.image_indices[self.image_index_counter] = key
        self.image_index_counter += 1

    def train(self):
        """
        Using the descriptors, generate the KDTree
        """
        self._flann_matcher.train()

    def query(self, descriptor, k=3, self_neighbor=True):
        """

        Parameters
        ----------
        descriptor : ndarray
                     The query descriptor to search for

        k : int
            The number of nearest neighbors to search for

        self_neighbor : bool
                        If the query descriptor is also a member
                        of the KDTree avoid self neighbor, default True.

        Returns
        -------
        matched : dataframe
                  containing matched points with columns containg:
                  matched image name, query index, train index, and
                  descriptor distance
        """
        idx = 0
        if self_neighbor:
            idx = 1
        matches = self._flann_matcher.knnMatch(descriptor, k=k)
        matched = []
        for m in matches:
            for i in m[idx:]:
                matched.append((self.image_indices[i.imgIdx],
                                i.queryIdx,
                                i.trainIdx,
                                i.distance))
        return pd.DataFrame(matched, columns=['matched_to', 'queryIdx',
                                              'trainIdx', 'distance'])
