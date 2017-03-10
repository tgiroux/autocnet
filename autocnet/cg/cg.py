import pandas as pd
import numpy as np
import geopandas as gpd
import ogr

from scipy.spatial import ConvexHull
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point
from shapely.affinity import scale
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

        intersection, proj_gdf, source_gdf = compute_intersection(graph.nodes()[0], graph, clean_keys)

        intersection_ind = intersection.query('overlaps_all == True').index.values[0]

        source_node = graph.nodes()[0]
        for e in graph.edges():
            if e[0] != source_node:
                source_node = e[0]
                intersection, proj_gdf, source_gdf = compute_intersection(source_node, graph, clean_keys)
                intersection_ind = intersection.query('overlaps_all == True').index.values[0]

            edge = graph.edge[e[0]][e[1]]
            matches, mask = edge.clean(clean_keys = clean_keys)
            kps = edge.get_keypoints('source', clean_keys = clean_keys, homogeneous = True)

            kps['geometry'] = kps.apply(lambda x: Point(x['x'], x['y']), axis = 1)
            kps.mask = kps['geometry'].apply(lambda x: intersection.geometry.contains(x).all())

            # Creates a mask for displaying the voronoi points
            # Currently erronious and produces NaN values
            # mask[mask] = kps.mask
            # edge.masks = ('voronoi', mask)

            keypoints = []
            kps[kps.mask].apply(lambda x: keypoints.append((x['x'], x['y'])), axis = 1)
            coords = source_gdf.geometry.apply(lambda x:scale(x, s, s).exterior.coords)

            for point in coords:
                keypoints = np.vstack((keypoints, point))

            vor = Voronoi(keypoints)
            voronoi_df = pd.DataFrame(data = kps, columns = ['x', 'y', 'weight'])

            i = 0
            vor_points = np.asarray(vor.points)
            for region in vor.regions:
                region_point = vor_points[np.argwhere(vor.point_region==i)]
                if -1 not in region:
                    polygon_points = [vor.vertices[i] for i in region]
                    if len(polygon_points) != 0:
                        polygon = Polygon(polygon_points)
                        poly_area = polygon.intersection(intersection.geometry[intersection_ind]).area
                        voronoi_df.loc[(voronoi_df["x"] == region_point[0][0][0]) &
                                       (voronoi_df["y"] == region_point[0][0][1]),
                                       'weight'] = poly_area
                i += 1

            edge.weight['vor_weight'] = voronoi_df['weight']


def compute_intersection(source, graph, clean_keys=[]):
    if type(source) is int:
        source = graph.node[source]

    source_corners = source.geodata.xy_corners
    source_poly = Polygon(source_corners)

    source_gdf = gpd.GeoDataFrame({'source_geom': [source_poly], 'source_node': [source.node_id]}).set_geometry('source_geom')

    proj_list = []
    proj_nodes = []
    # Begin iterating through the nodes in the graph excluding the source node
    for n in graph.nodes_iter():
        if n == source.node_id:
            continue

        # Define the edge, matches, and destination based on the zero node and the nth node
        destination = graph.node[n]
        destination_corners = destination.geodata.xy_corners

        edge = graph.edge[source.node_id][destination.node_id]

        # Will still need the check but the helper function will make these calls much easier to understand
        if source.node_id > destination.node_id:
            kp2 = edge.get_keypoints('source', clean_keys = clean_keys, homogeneous = True)
            kp1 = edge.get_keypoints('destination', clean_keys = clean_keys, homogeneous = True)
        else:
            kp2 = edge.get_keypoints('destination', clean_keys = clean_keys, homogeneous = True)
            kp1 = edge.get_keypoints('source', clean_keys = clean_keys, homogeneous = True)

        # If the source image has coordinate transformation data
        if (source.geodata.coordinate_transformation.this != None) and \
        (destination.geodata.coordinate_transformation.this != None):
            proj_poly = Polygon(destination.geodata.xy_corners)

        # Else, use the homography transform to get an intersection of the two images
        else:
            H, mask = cv2.findHomography(kp2.values, kp1.values, cv2.RANSAC, 2.0)
            proj_corners = []
            for c in destination_corners:
                x, y, h = utils.reproj_point(H, c)
                x /= h
                y /= h
                h /= h
                proj_corners.append((x, y))

            proj_poly = Polygon(proj_corners)

        proj_list.append(proj_poly)
        proj_nodes.append(n)

    proj_gdf = gpd.GeoDataFrame({'proj_geom': proj_list, 'proj_node': proj_nodes}).set_geometry('proj_geom')

    intersect_gdf = gpd.overlay(source_gdf, proj_gdf, how='intersection')
    intersect_gdf = intersect_gdf.rename(columns = {'geometry':'intersect_geom'}).set_geometry('intersect_geom')
    intersect_gdf['overlaps_all'] = intersect_gdf.geometry.apply(lambda x:proj_gdf.geometry.contains(scale(x, .9, .9)).all())
    intersection = intersect_gdf.query('overlaps_all == True').set_geometry('intersect_geom')
    return intersection, proj_gdf, source_gdf
