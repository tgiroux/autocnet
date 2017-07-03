import warnings

import pandas as pd
import numpy as np
import networkx as nx
import geopandas as gpd
import ogr

from scipy.spatial import ConvexHull
from scipy.spatial import Voronoi
import shapely.geometry
from shapely.geometry import Polygon, Point
from shapely.affinity import scale
# from shapely.ops import unary_union
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


def geom_mask(keypoints, geom):
    def _in_mbr(r, mbr):
        if (mbr[0] <= r.x <= mbr[2]) and (mbr[1] <= r.y <= mbr[3]):
            return True
        else:
            return False

    mbr = geom.bounds
    initial_mask = keypoints.apply(_in_mbr, axis=1, args=(mbr,))

    return initial_mask


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


def compute_voronoi(keypoints, intersection=None, geometry=False, s=30):
        """
        Creates a voronoi diagram for all edges in a graph, and assigns a given
        weight to each edge. This is based around voronoi polygons generated
        by scipy's voronoi method, to determine if an image has significant coverage.

        Parameters
        ----------
        graph : object
               A networkx graph object

        clean_keys : list
                     Of strings used to apply masks to omit correspondences

        s : int
            Offset for the corners of the image
        """
        vor_keypoints = []

        keypoints.apply(lambda x: vor_keypoints.append((x['x'], x['y'])), axis = 1)

        if intersection is None:
            keypoint_bounds = Polygon(vor_keypoints).bounds
            min_bounding_box = shapely.geometry.box(keypoint_bounds[0], keypoint_bounds[1],
                                                    keypoint_bounds[2], keypoint_bounds[3])

            scaled_coords = np.array(scale(min_bounding_box, s, s).exterior.coords)
        else:
            scaled_coords = np.array(scale(intersection, s, s).exterior.coords)

        vor_keypoints = np.vstack((vor_keypoints, scaled_coords))
        vor = Voronoi(vor_keypoints)

        # For weight computation
        # Should move to its own method
        if geometry:
            voronoi_df = gpd.GeoDataFrame(data = keypoints, columns=['x', 'y', 'weight', 'geometry'])
        else:
            voronoi_df = gpd.GeoDataFrame(data = keypoints, columns=['x', 'y', 'weight'])

        i = 0
        vor_points = np.asarray(vor.points)
        for region in vor.regions:
            region_point = vor_points[np.argwhere(vor.point_region==i)]

            if not -1 in region:
                polygon_points = [vor.vertices[i] for i in region]

                if len(polygon_points) != 0:
                    polygon = Polygon(polygon_points)

                    if intersection is not None:
                        intersection_poly = polygon.intersection(intersection)
                    else:
                        intersection_poly = polygon.intersection(min_bounding_box)

                    voronoi_df.loc[(voronoi_df["x"] == region_point[0][0][0]) &
                                   (voronoi_df["y"] == region_point[0][0][1]),
                                   'weight'] = intersection_poly.area
                    if geometry:
                        voronoi_df.loc[(voronoi_df["x"] == region_point[0][0][0]) &
                                       (voronoi_df["y"] == region_point[0][0][1]),
                                       'geometry'] = intersection_poly
            i += 1

        return voronoi_df


def compute_intersection(graph, source, clean_keys=[]):
    """
    Computes the intersections of images in a graph based on the
    connections between nodes. The method takes every node in a graph,
    sees who is connected to it, then computes an intersection based on
    those connections.

    Parameters
    ----------
    source : int or object
             Node id or Node object to use as the initial reprojection space

    graph : object
            a networkx graph object

    clean_keys : list
                 Of strings used to apply masks to omit correspondences

    Returns
    -------
    intersection : dataframe
                   4 column dataframe of source_node, proj_node, geometry,
                   and overlaps_all

    proj_gdf : dataframe
               2 column dataframe of proj_geom, proj_node

    source_gdf : dataframe
                 2 column dataframe of geometry, source_node
    """
    if type(source) is int:
        source = graph.node[source]

    source_corners = source.geodata.xy_corners
    source_poly = Polygon(source_corners)

    source_gdf = gpd.GeoDataFrame({'geometry': [source_poly], 'source_node': [source['node_id']]})

    proj_list = []
    proj_nodes = []
    # Begin iterating through the nodes in the graph excluding the source node
    for n in graph.nodes_iter():
        if n == source['node_id']:
            continue

        # Define the edge, matches, and destination based on the zero node and the nth node
        destination = graph.node[n]
        destination_corners = destination.geodata.xy_corners

        try:
            edge = graph.edge[source['node_id']][destination['node_id']]

            # If the source image has coordinate transformation data
            if (source.geodata.coordinate_transformation.this is not None) and \
               (destination.geodata.coordinate_transformation.this is not None):
                proj_poly = Polygon(destination.geodata.xy_corners)

            # Else, use the homography transform to get an intersection of the two images
            else:
                # Will still need the check but the helper function will make these calls much easier to understand
                if source['node_id'] > destination['node_id']:
                    kp2 = edge.get_keypoints('source', clean_keys=clean_keys, homogeneous=True)
                    kp1 = edge.get_keypoints('destination', clean_keys=clean_keys, homogeneous=True)
                else:
                    kp2 = edge.get_keypoints('destination', clean_keys=clean_keys, homogeneous=True)
                    kp1 = edge.get_keypoints('source', clean_keys=clean_keys, homogeneous=True)

                H, mask = cv2.findHomography(kp2.values, kp1.values, cv2.RANSAC, 2.0)
                proj_corners = []
                for c in destination_corners:
                    x, y, h = utils.reproj_point(H, c)
                    x /= h
                    y /= h
                    h /= h
                    proj_corners.append((x, y))

                proj_poly = Polygon(proj_corners)
        except:
            continue

        proj_list.append(proj_poly)
        proj_nodes.append(n)

    proj_gdf = gpd.GeoDataFrame({'geometry': proj_list, 'proj_node': proj_nodes})

    intersect_gdf = gpd.overlay(source_gdf, proj_gdf, how='intersection')
    intersect_gdf['overlaps_all'] = intersect_gdf.geometry.apply(lambda z: proj_gdf.geometry.contains(scale(z, .9, .9)).all())
    intersection = intersect_gdf.query('overlaps_all == True')

    if len(intersection) == 0:
        new_poly = unary_union(intersect_gdf.geometry)
        intersection = gpd.GeoDataFrame({'source_node': source['node_id'], 'geometry': new_poly, 'overlaps_all': True})
    return intersection, proj_gdf, source_gdf
