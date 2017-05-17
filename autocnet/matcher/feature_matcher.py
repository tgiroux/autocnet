import pandas as pd
from autocnet.matcher.feature import FlannMatcher

def match(self, k=2, **kwargs):
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

    # TODO: This should be converted to a decorator on the class
    # TODO: This entire method should never have access to the class
    self.masks = pd.DataFrame()

    fl = FlannMatcher()
    mono_matches(self.source, self.destination, **kwargs)

    # Since this matches bidirectionally
    if 'aidx' in kwargs.keys():
        if not 'bidx' in kwargs.keys():
            kwargs['bidx'] = None
        kwargs['aidx'], kwargs['bidx'] = kwargs['bidx'], kwargs['aidx']

    mono_matches(self.destination, self.source, **kwargs)

    self.matches.sort_values(by=['distance'])
