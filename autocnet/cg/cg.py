import pandas as pd
import numpy as np
import ogr

from scipy.spatial import ConvexHull
from scipy.spatial import Voronoi
import cv2

from autocnet.utils import utils


def convex_hull_ratio(points, ideal_area):
    """

    Parameters
    ----------
    points : ndarray
             (n, 2) array of point coordinates

    ideal_area : float
                 The total area that could be covered

    Returns
    -------
    ratio : float
            The ratio convex hull volume / ideal_area

    """
    hull = ConvexHull(points)
    return hull.volume / ideal_area


def convex_hull(points):

    """

    Parameters
    ----------
    points : ndarray
             (n, 2) array of point coordinates

    Returns
    -------
    hull : 2-D convex hull
            Provides a convex hull that is used
            to determine coverage

    """

    if isinstance(points, pd.DataFrame) :
        points = pd.DataFrame.as_matrix(points)

    hull = ConvexHull(points)
    return hull


def two_poly_overlap(poly1, poly2):
    """

    Parameters
    ----------
    poly1 : ogr polygon
            Any polygon that shares some kind of overlap
            with poly2

    poly2 : ogr polygon
            Any polygon that shares some kind of overlap
            with poly1

    Returns
    -------
    overlap_percn : float
                    The percentage of image overlap

    overlap_area : float
                   The total area of overalap

    """
    overlap_area_polygon = poly2.Intersection(poly1)
    overlap_area = overlap_area_polygon.GetArea()
    area1 = poly1.GetArea()
    area2 = poly2.GetArea()

    overlap_percn = (overlap_area / (area1 + area2 - overlap_area)) * 100
    return overlap_percn, overlap_area, overlap_area_polygon


def get_area(poly1, poly2):
    """

    Parameters
    ----------
    poly1 : ogr polygon
            General ogr polygon

    poly2 : ogr polygon
            General ogr polygon

    Returns
    -------
    intersection_area : float
                        returns the intersection area
                        of two polygons

    """
    intersection_area = poly1.Intersection(poly2).GetArea()
    return intersection_area


def vor(graph, clean_keys=[], s=30):
        """
        Creates a voronoi diagram for an edge using either the coordinate
        transformation or using the homography between source and destination.

        The coordinate transformation uses the footprint of source and destination to
        calculate an intersection between the two images, then transforms the vertices of
        the intersection back into pixel space.

        If a coordinate transform does not exist, use the homography to project the destination image
        onto the source image, producing an area of intersection.

        The intersection vertices are then scaled by a factor of s (default 30), this accounts for the
        areas of the voronoi that would be missed if the scaled vertices were not included into the
        voronoi calculation.


        Parameters
        ----------
        edge : object
               An edge object

        clean_keys : list
                     Of strings used to apply masks to omit correspondences

        s : int
            offset for the corners of the image

        Returns
        -------
        vor : object
              Scipy Voronoi object

        voronoi_df : dataframe
                     3 column pandas dataframe of x, y, and weights

        """
        num_neighbors = len(graph.nodes()) - 1
        for n in graph.nodes():
            neighbors = len(graph.neighbors(n))
            if neighbors != num_neighbors:
                raise AssertionError('The graph is not complete')

        source = graph.edges(0)[0]
        destination = graph.edges()[0][1]
        edge = graph.edge[source][destination]
        matches, _ = edge.clean(clean_keys=['fundamental'])
        kps = edge.source.get_keypoint_coordinates(index=matches['source_idx'], homogeneous=True)

        intersection_poly = compute_intersection(graph, clean_keys)
        intersection_points = intersection_poly.GetGeometryRef(0).GetPoints()

        centroid = intersection_poly.Centroid().GetPoint()

        points = np.asarray(kps)
        voronoi_np = []

        point_cloud = ogr.Geometry(ogr.wkbMultiPoint)
        for p in points:
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(double(p[0]), double(p[1]))
            point_cloud.AddGeometry(point)

        intersection_cloud = intersection_poly.Intersection(point_cloud)

        for p in intersection_cloud:
            point = p.GetPoint(0)
            voronoi_np.append(point)

        keypoints = np.asarray(voronoi_np)
        voronoi_pd = pd.DataFrame(data=voronoi_np, columns=['x', 'y', 'homogenious'])

        # Based on the keypoints found in the method
        inters = np.empty((len(intersection_points),2))

        for g, (i, j) in enumerate(intersection_points):
            scaledx, scaledy = utils.scale_point((i, j), centroid, s)
            point = np.array([scaledx, scaledy])
            inters[g] = point

        keypoints = np.vstack((keypoints[:,:2], inters))
        vor = Voronoi(keypoints)

        voronoi_df = pd.DataFrame(columns=["x", "y", "weights"])
        voronoi_df["x"] = kps['x']
        voronoi_df["y"] = kps['y']

        poly_array = []
        for i, region in enumerate(vor.regions):
            region_point = vor.points[np.argwhere(vor.point_region==i)]
            if -1 not in region:
                polygon_points = [vor.vertices[i] for i in region]
                if len(polygon_points) != 0:
                    polygon = utils.array_to_poly(polygon_points)
                    intersection = polygon.Intersection(intersection_poly)
                    poly_array = np.append(poly_array, intersection)
                    polygon_area = intersection.GetArea()
                    voronoi_df.loc[(voronoi_df["x"] == region_point[0][0][0]) &
                                   (voronoi_df["y"] == region_point[0][0][1]),
                                   'vor_weights'] = polygon_area

        return vor, voronoi_df


def compute_intersection(graph, clean_keys=[]):
    source_num = graph.edges()[0][0]
    source = graph.node[source_num]
    source_corners = source.geodata.xy_corners
    total_intersect_poly = utils.array_to_poly(source_corners)
    orig_poly = utils.array_to_poly(source_corners)

    for n in graph.nodes_iter():
        if n == source_num:
            continue

        # Define the edge, matches, and destination based on the zero node and the nth node
        edge = graph.edge[source_num][n]
        matches, _ = edge.clean(clean_keys=clean_keys)
        destination = edge.destination
        destination_corners = destination.geodata.xy_corners

        kp1 = edge.source.get_keypoint_coordinates(index=matches['source_idx'], homogeneous=True)
        kp2 = edge.destination.get_keypoint_coordinates(index=matches['destination_idx'], homogeneous=True)

        # If the source image has coordinate transformation data, us the footprint and
        # coordinate transforms as it will produce a more accurate intersection
        if edge.source.geodata.coordinate_transformation.this is not None:
            source_footprint_poly = edge.source.geodata.footprint
            destination_footprint_poly = edge.destination.geodata.footprint

            intersection_poly = destination_footprint_poly.Intersection(source_footprint_poly)

        # Else, use the homography transform to get an intersection of the two images
        else:
            H, mask = cv2.findHomography(kp2.values,
                                         kp1.values,
                                         cv2.RANSAC,
                                         2.0)

            proj_corners = []
            for c in destination_corners:
                x, y, h = utils.reproj_point(H, c)
                x /= h
                y /= h
                h /= h
                proj_corners.append((x, y))

            proj_poly = utils.array_to_poly(proj_corners)

            intersection_poly = orig_poly.Intersection(proj_poly)

        # Intersect the newly calculated intersection with the current total intersection
        # to get the new bound of the overlap area
        total_intersect_poly = intersection_poly.Intersection(total_intersect_poly)

        return total_intersect_poly
