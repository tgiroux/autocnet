import os
import sys
import unittest
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import pandas as pd

from .. import cg
from osgeo import ogr
from unittest.mock import Mock, MagicMock
from plio.io import io_gdal

from autocnet.graph.node import Node
from autocnet.graph.network import CandidateGraph
from autocnet.graph.edge import Edge
from autocnet.utils.utils import array_to_poly


class TestArea(unittest.TestCase):

    def setUp(self):
        seed = np.random.RandomState(12345)
        self.pts = seed.rand(25, 2)

    def test_area_single(self):
        total_area = 1.0
        ratio = cg.convex_hull_ratio(self.pts, total_area)

        self.assertAlmostEqual(0.7566490, ratio, 5)

    def test_overlap(self):
        wkt1 = "POLYGON ((0 40, 40 40, 40 0, 0 0, 0 40))"
        wkt2 = "POLYGON ((20 60, 60 60, 60 20, 20 20, 20 60))"

        poly1 = ogr.CreateGeometryFromWkt(wkt1)
        poly2 = ogr.CreateGeometryFromWkt(wkt2)

        info = cg.two_poly_overlap(poly1, poly2)

        self.assertEqual(info[1], 400)
        self.assertAlmostEqual(info[0], 14.285714285)

    def test_voronoi_homography(self):
        source_keypoint_df = pd.DataFrame({'x': (15, 18, 18, 12, 12), 'y': (6, 10, 15, 15, 10)})
        destination_keypoint_df = pd.DataFrame({'x': (5, 8, 8, 2, 2), 'y': (1, 5, 10, 10, 5)})
        keypoint_matches = [[0, 0, 1, 0],
                            [0, 1, 1, 1],
                            [0, 2, 1, 2],
                            [0, 3, 1, 3],
                            [0, 4, 1, 4]]

        matches_df = pd.DataFrame(data=keypoint_matches, columns=['source_image', 'source_idx',
                                                                  'destination_image', 'destination_idx'])
        # Source and Destination Node Setup
        source_node = MagicMock(spec=Node())
        destination_node = MagicMock(spec=Node())

        source_node.get_keypoint_coordinates = MagicMock(return_value=source_keypoint_df)
        destination_node.get_keypoint_coordinates = MagicMock(return_value=destination_keypoint_df)

        source_geodata = Mock(spec=io_gdal.GeoDataset)
        destination_geodata = Mock(spec=io_gdal.GeoDataset)

        source_node.geodata = source_geodata
        destination_node.geodata = destination_geodata

        source_node.geodata.coordinate_transformation.this = None
        destination_node.geodata.coordinate_transformation.this = None

        source_corners = [(0, 0),
                          (20, 0),
                          (20, 20),
                          (0, 20)]

        destination_corners = [(0, 0),
                               (20, 0),
                               (20, 20),
                               (0, 20)]

        source_node.geodata.xy_corners = source_corners
        destination_node.geodata.xy_corners = destination_corners

        # Edge Setup
        e = Edge(source=source_node, destination=destination_node)

        e.clean = MagicMock(return_value=(matches_df, None))
        e.matches = matches_df

        def side_effect(node, clean_keys, **kwargs):
            if type(node) is str:
                node = node.lower()

            if isinstance(node, Node):
                node = node['node_id']

            if node == "source" or node == "s" or node == source_node['node_id']:
                return e.source.get_keypoint_coordinates().copy(deep=True)
            if node == "destination" or node == "d" or node == destination_node['node_id']:
                return e.destination.get_keypoint_coordinates().copy(deep=True)

        my_dict_source = {'node_id':0}

        def getitem_source(name):
            return my_dict_source[name]

        my_dict_destination = {'node_id':1}

        def getitem_destination(name):
            return my_dict_destination[name]

        source_node.__getitem__.side_effect = getitem_source
        destination_node.__getitem__.side_effect = getitem_destination

        e.get_keypoints = MagicMock(side_effect=side_effect)

        cang = MagicMock(spec=CandidateGraph())

        cang.nodes = MagicMock(return_value=([0, 1]))
        cang.node = [source_node, destination_node]
        cang.nodes_iter = MagicMock(return_value=([0, 1]))
        cang.neighbors = MagicMock(return_value=[1])
        cang.edges = MagicMock(return_value=([(0, 1)]))
        cang.edge = {0: {1: e}, 1: {0: e}}

        cg.vor(cang, clean_keys=['fundamental'])
        self.assertAlmostEqual(e['weights']['vor_weight'][0], 22.5)
        self.assertAlmostEqual(e['weights']['vor_weight'][1], 26.25)
        self.assertAlmostEqual(e['weights']['vor_weight'][2], 37.5)
        self.assertAlmostEqual(e['weights']['vor_weight'][3], 37.5)
        self.assertAlmostEqual(e['weights']['vor_weight'][4], 26.25)

    def test_voronoi_coord(self):
        source_keypoint_df = pd.DataFrame({'x': (15, 18, 18, 12, 12), 'y': (6, 10, 15, 15, 10)})
        destination_keypoint_df = pd.DataFrame({'x': (5, 8, 8, 2, 2), 'y': (1, 5, 10, 10, 5)})

        keypoint_df = pd.DataFrame({'x': (15, 18, 18, 12, 12), 'y': (6, 10, 15, 15, 10)})
        keypoint_matches = [[0, 0, 1, 0],
                            [0, 1, 1, 1],
                            [0, 2, 1, 2],
                            [0, 3, 1, 3],
                            [0, 4, 1, 4]]

        matches_df = pd.DataFrame(data=keypoint_matches, columns=['source_image', 'source_idx',
                                                                  'destination_image', 'destination_idx'])

        source_node = MagicMock(spec=Node())
        destination_node = MagicMock(spec=Node())

        source_node.get_keypoint_coordinates = MagicMock(return_value=source_keypoint_df)
        destination_node.get_keypoint_coordinates = MagicMock(return_value=destination_keypoint_df)

        source_geodata = Mock(spec=io_gdal.GeoDataset)
        destination_geodata = Mock(spec=io_gdal.GeoDataset)

        source_node.geodata = source_geodata
        destination_node.geodata = destination_geodata

        e = Edge(source=source_node, destination = destination_node)

        e.clean = MagicMock(return_value=(matches_df, None))
        e.matches = matches_df

        source_node = MagicMock(spec=Node())
        destination_node = MagicMock(spec=Node())

        cang = MagicMock(spec=CandidateGraph())

        cang.nodes = MagicMock(return_value=([0, 1]))
        cang.node = [source_node, destination_node]
        cang.nodes_iter = MagicMock(return_value=([0, 1]))
        cang.neighbors = MagicMock(return_value=([0]))
        cang.edges = MagicMock(return_value=([(0, 1)]))
        cang.edge = {0: {1: e}, 1: {0: e}}

        source_node.get_keypoint_coordinates = MagicMock(return_value=keypoint_df)
        destination_node.get_keypoint_coordinates = MagicMock(return_value=keypoint_df)
        source_node['node_id'] = 0
        destination_node['node_id'] = 1

        def side_effect(node, clean_keys, **kwargs):
            if type(node) is str:
                node = node.lower()

            if isinstance(node, Node):
                node = node['node_id']

            if node == "source" or node == "s" or node == source_node['node_id']:
                return e.source.get_keypoint_coordinates()
            if node == "destination" or node == "d" or node == destination_node['node_id']:
                return e.destination.get_keypoint_coordinates()

        e.get_keypoints = MagicMock(side_effect=side_effect)

        e.source = source_node
        e.destination = destination_node

        source_geodata = Mock(spec=io_gdal.GeoDataset)
        destination_geodata = Mock(spec=io_gdal.GeoDataset)

        e.source.geodata = source_geodata
        e.destination.geodata = destination_geodata

        source_corners = [(0, 0),
                          (20, 0),
                          (20, 20),
                          (0, 20)]

        destination_corners = [(10, 5),
                               (30, 5),
                               (30, 25),
                               (10, 25)]

        source_xy_extent = [(0, 20), (0, 20)]

        destination_xy_extent = [(10, 30), (5, 25)]

        source_poly = array_to_poly(source_corners)
        destination_poly = array_to_poly(destination_corners)

        vals = {(10, 5): (10, 5), (20, 5): (20, 5), (20, 20): (20, 20), (10, 20): (10, 20)}

        def latlon_to_pixel(i, j):
            return vals[(i, j)]

        my_dict_source = {'node_id': 0}

        def getitem_source(name):
            return my_dict_source[name]

        my_dict_destination = {'node_id': 1}

        def getitem_destination(name):
            return my_dict_destination[name]

        source_node.__getitem__.side_effect = getitem_source
        destination_node.__getitem__.side_effect = getitem_destination

        e.source.geodata.latlon_to_pixel = MagicMock(side_effect=latlon_to_pixel)
        e.destination.geodata.latlon_to_pixel = MagicMock(side_effect=latlon_to_pixel)

        e.source.geodata.footprint = source_poly
        e.source.geodata.xy_corners = source_corners
        e.source.geodata.xy_extent = source_xy_extent
        e.destination.geodata.footprint = destination_poly
        e.destination.geodata.xy_corners = destination_corners
        e.destination.geodata.xy_extent = destination_xy_extent
        cg.vor(cang, clean_keys=['fundamental'])
        self.assertAlmostEqual(e['weights']['vor_weight'][0], 22.5)
        self.assertAlmostEqual(e['weights']['vor_weight'][1], 26.25)
        self.assertAlmostEqual(e['weights']['vor_weight'][2], 37.5)
        self.assertAlmostEqual(e['weights']['vor_weight'][3], 37.5)
        self.assertAlmostEqual(e['weights']['vor_weight'][4], 26.25)
