import warnings

import cv2
import numpy as np
import pandas as pd

FLANN_INDEX_KDTREE = 1  # Algorithm to set centers,
DEFAULT_FLANN_PARAMETERS = dict(algorithm=FLANN_INDEX_KDTREE, trees=3)


def match(self, k=2, overlap=False, **kwargs):
    """
    Given two sets of descriptors, utilize a FLANN (Approximate Nearest
    Neighbor KDTree) matcher to find the k nearest matches.  Nearness is
    the euclidean distance between descriptors.

    The matches are then added as an attribute to the edge object.

    Parameters
    ----------
    k : int
	The number of neighbors to find
    """

    def _add_matches(matches):
        """
        Given a dataframe of matches, either append to an existing
        matches edge attribute or initially populate said attribute.

        Parameters
        ----------
        matches : dataframe
                  A dataframe of matches
        """
        if self.matches is None:
            self.matches = matches
        else:
            df = self.matches
            self.matches = df.append(matches,
                                     ignore_index=True,
                                     verify_integrity=True)

    def mono_matches(a, b, aidx=None, bidx=None):
        """
	    Apply the FLANN match_features

    	Parameters
    	----------
    	a : object
    	    A node object

    	b : object
    	    A node object

    	aidx : iterable
    		An index for the descriptors to subset

    	bidx : iterable
    		An index for the descriptors to subset
    	"""
    	# Subset if requested
        if aidx is not None:
            ad = a.descriptors[aidx]
        else:
            ad = a.descriptors

        if bidx is not None:
            bd = b.descriptors[bidx]
        else:
            bd = b.descriptors

        # Load, train, and match
        fl.add(ad, a['node_id'], index=aidx)
        fl.train()
        matches = fl.query(bd, b['node_id'], k, index=bidx)
        _add_matches(matches)
        fl.clear()

    fl = FlannMatcher()
    
    # Get the correct descriptors
    # TODO: Extract into a helper function
    if 'aidx' in kwargs.keys():
        aidx = kwargs['aidx']
        kwargs.pop('aidx')
    elif overlap:
        # Query the source keypoints for those in the MBR
        source_mbr = self['source_mbr']
        query_result = self.source.keypoints.query()
        aidx = query_result.index
    else:
        aidx = None
    
    if 'bidx' in kwargs.keys():
        bidx = kwargs['bidx']
        kwargs.pop('bidx')
    elif overlap:
        destin_mbr = self['destin_mbr']
        query_result = self.destination.keypoints.query()
        bidx = query_result.index
    else:
        bidx = None

    mono_matches(self.source, self.destination, aidx=aidx, bidx=bidx, **kwargs)
    # Swap the indices since mono_matches is generic and source/destin are
    # swapped
    mono_matches(self.destination, self.source, aidx=bidx, bidx=aidx, **kwargs)

    self.matches.sort_values(by=['distance'])


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
        self.nid_lookup = {}
        self.search_idx = {}
        self.node_counter = 0

    def add(self, descriptor, nid, index=None):
        """
        Add a set of descriptors to the matcher and add the image
        index key to the image_indices attribute

        Parameters
        ----------
        descriptor : ndarray
                     The descriptor to be added

        nid : int
              The node ids
        """
        self._flann_matcher.add([descriptor])
        self.nid_lookup[self.node_counter] = nid
        self.node_counter += 1
        if index is not None:
            self.search_idx = dict((i, j) for i, j in enumerate(index))
        else:
            self.search_idx = dict((i,i) for i in range(len(descriptor)))

    def clear(self):
        """
        Remove all nodes from the tree and resets
        all counters
        """
        self._flann_matcher.clear()
        self.nid_lookup = {}
        self.node_counter = 0
        self.search_idx = {}

    def train(self):
        """
        Using the descriptors, generate the KDTree
        """
        self._flann_matcher.train()

    def query(self, descriptor, query_image, k=3, index=None):
        """

        Parameters
        ----------
        descriptor : ndarray
                     The query descriptor to search for

        query_image : hashable
                      Key of the query image

        k : int
            The number of nearest neighbors to search for

        index : iterable
                An iterable of observation indices to utilize for
                the input descriptors

        Returns
        -------
        matched : dataframe
                  containing matched points with columns containing:
                  matched image name, query index, train index, and
                  descriptor distance
        """

        matches = self._flann_matcher.knnMatch(descriptor, k=k)
        matched = []
        for i, m in enumerate(matches):
            for j in m:
                if index is not None:
                    qid = index[i]
                else:
                    qid = j.queryIdx
                source = query_image
                destination = self.nid_lookup[j.imgIdx]
                if source < destination:
                    matched.append((query_image,
                                    qid,
                                    destination,
                                    self.search_idx[j.trainIdx],
                                    j.distance))
                elif source > destination:
                    matched.append((destination,
                                    self.search_idx[j.trainIdx],
                                    query_image,
                                    qid,
                                    j.distance))
                else:
                    warnings.warn('Likely self neighbor in query!')
        return pd.DataFrame(matched, columns=['source_image', 'source_idx',
                                              'destination_image', 'destination_idx',
                                              'distance']).astype(np.float32)
