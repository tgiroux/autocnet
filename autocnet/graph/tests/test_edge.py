import unittest
from unittest.mock import Mock, MagicMock

import ogr
import numpy as np
import pandas as pd
from plio.io import io_gdal

from autocnet.matcher import cpu_outlier_detector as od
from autocnet.examples import get_path
from autocnet.graph.network import CandidateGraph
from autocnet.utils.utils import array_to_poly

from .. import edge
from .. import node


class TestEdge(unittest.TestCase):

    def setUp(self):
        source = Mock(node.Node)
        destination = Mock(node.Node)
        self.edge = edge.Edge(source=source, destination=destination)

    def test_masks(self):
        self.assertIsInstance(self.edge.masks, pd.DataFrame)

    def test_edge_overlap(self):
        e = edge.Edge()
        e.weight = {}
        source = Mock(spec = node.Node)
        destination = Mock(spec = node.Node)
        e.destination = destination
        e.source = source
        geodata_s = Mock(spec = io_gdal.GeoDataset)
        geodata_d = Mock(spec = io_gdal.GeoDataset)
        source.geodata = geodata_s
        destination.geodata = geodata_d

        wkt1 = "POLYGON ((0 40, 40 40, 40 0, 0 0, 0 40))"
        wkt2 = "POLYGON ((20 60, 60 60, 60 20, 20 20, 20 60))"

        poly1 = ogr.CreateGeometryFromWkt(wkt1)
        poly2 = ogr.CreateGeometryFromWkt(wkt2)

        source.geodata.footprint = poly1
        destination.geodata.footprint = poly2

        e.overlap()
        self.assertEqual(e['weights']['overlap_area'], 400)
        self.assertAlmostEqual(e['weights']['overlap_percn'], 14.285714285)

    def test_coverage(self):
        adjacency = get_path('two_image_adjacency.json')
        basepath = get_path('Apollo15')
        cg = CandidateGraph.from_adjacency(adjacency, basepath=basepath)
        keypoint_df = pd.DataFrame({'x': (15, 18, 18, 12, 12), 'y': (5, 10, 15, 15, 10)})
        keypoint_matches = [[0, 0, 1, 0],
                            [0, 1, 1, 1],
                            [0, 2, 1, 2],
                            [0, 3, 1, 3],
                            [0, 4, 1, 4]]

        matches_df = pd.DataFrame(keypoint_matches, columns = ['source_image', 'source_idx', 'destination_image', 'destination_idx'])
        e = edge.Edge()
        source_node = MagicMock(spec = node.Node())
        destination_node = MagicMock(spec = node.Node())

        source_node.get_keypoint_coordinates = MagicMock(return_value=keypoint_df)
        destination_node.get_keypoint_coordinates = MagicMock(return_value=keypoint_df)

        e.source = source_node
        e.destination = destination_node

        source_geodata = Mock(spec = io_gdal.GeoDataset)
        destination_geodata = Mock(spec = io_gdal.GeoDataset)

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

        e.source.geodata.latlon_corners = source_corners
        e.destination.geodata.latlon_corners = destination_corners

        vals = {(15, 5): (15, 5), (18, 10): (18, 10), (18, 15): (18, 15), (12, 15): (12, 15), (12, 10): (12, 10)}

        def pixel_to_latlon(i, j):
            return vals[(i, j)]

        e.source.geodata.pixel_to_latlon = MagicMock(side_effect = pixel_to_latlon)
        e.destination.geodata.pixel_to_latlon = MagicMock(side_effect = pixel_to_latlon)

        e.matches = matches_df

        #self.assertRaises(AttributeError, cg.edge[0][1].coverage)
        self.assertEqual(e.coverage(), 0.3)

    def test_voronoi_transform(self):
        keypoint_df = pd.DataFrame({'x': (15, 18, 18, 12, 12), 'y': (5, 10, 15, 15, 10)})
        keypoint_matches = [[0, 0, 1, 0],
                            [0, 1, 1, 1],
                            [0, 2, 1, 2],
                            [0, 3, 1, 3],
                            [0, 4, 1, 4]]

        matches_df = pd.DataFrame(data=keypoint_matches, columns=['source_image', 'source_idx',
                                                                  'destination_image', 'destination_idx'])
        e = edge.Edge()

        e.clean = MagicMock(return_value=(matches_df, None))
        e.matches = matches_df

        source_node = MagicMock(spec=node.Node())
        destination_node = MagicMock(spec=node.Node())

        source_node.get_keypoint_coordinates = MagicMock(return_value=keypoint_df)
        destination_node.get_keypoint_coordinates = MagicMock(return_value=keypoint_df)

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

        source_poly = array_to_poly(source_corners)
        destination_poly = array_to_poly(destination_corners)

        def latlon_to_pixel(i, j):
            return vals[(i, j)]

        e.source.geodata.latlon_to_pixel = MagicMock(side_effect=latlon_to_pixel)
        e.destination.geodata.latlon_to_pixel = MagicMock(side_effect=latlon_to_pixel)

        e.source.geodata.footprint = source_poly
        e.source.geodata.xy_corners = source_corners
        e.destination.geodata.footprint = destination_poly
        e.destination.geodata.xy_corners = destination_corners

        vals = {(10, 5): (10, 5), (20, 5): (20, 5), (20, 20): (20, 20), (10, 20): (10, 20)}

        weights = pd.DataFrame({"vor_weights": (19, 28, 37.5, 37.5, 28)})

        e.compute_weights(clean_keys=[])

        k = 0
        for i in e.matches['vor_weights']:
            self.assertAlmostEquals(i, weights['vor_weights'][k])
            k += 1

    def test_voronoi_homography(self):
        source_keypoint_df = pd.DataFrame({'x': (15, 18, 18, 12, 12), 'y': (5, 10, 15, 15, 10)})
        destination_keypoint_df = pd.DataFrame({'x': (5, 8, 8, 2, 2), 'y': (0, 5, 10, 10, 5)})
        keypoint_matches = [[0, 0, 1, 0],
                            [0, 1, 1, 1],
                            [0, 2, 1, 2],
                            [0, 3, 1, 3],
                            [0, 4, 1, 4]]

        matches_df = pd.DataFrame(data = keypoint_matches, columns=['source_image', 'source_idx',
                                                                    'destination_image', 'destination_idx'])
        e = edge.Edge()

        e.clean = MagicMock(return_value=(matches_df, None))
        e.matches = matches_df

        source_node = MagicMock(spec=node.Node())
        destination_node = MagicMock(spec=node.Node())

        source_node.get_keypoint_coordinates = MagicMock(return_value=source_keypoint_df)
        destination_node.get_keypoint_coordinates = MagicMock(return_value=destination_keypoint_df)

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

        destination_corners = [(0, 0),
                               (20, 0),
                               (20, 20),
                               (0, 20)]

        e.source.geodata.coordinate_transformation.this = None
        e.destination.geodata.coordinate_transformation.this = None

        e.source.geodata.xy_corners = source_corners
        e.destination.geodata.xy_corners = destination_corners

        weights = pd.DataFrame({"vor_weights": (19, 28, 37.5, 37.5, 28)})

        e.compute_weights(clean_keys=[])

        k = 0
        for i in e.matches['vor_weights']:
            self.assertAlmostEquals(i, weights['vor_weights'][k])
            k += 1

    def test_eq(self):
        edge1 = edge.Edge()
        edge2 = edge.Edge()
        edge3 = edge.Edge()

        # Test edges w/ different keys are not equal, ones with same keys are
        edge1.__dict__["key"] = 1
        edge2.__dict__["key"] = 1
        edge3.__dict__["not_key"] = 1

        self.assertTrue(edge1 == edge2)
        self.assertFalse(edge1 == edge3)

        # Test edges with same keys, but diff df values
        edge1.__dict__["key"] = pd.DataFrame({'x': (0, 1, 2, 3, 4)})
        edge2.__dict__["key"] = pd.DataFrame({'x': (0, 1, 2, 3, 4)})
        edge3.__dict__["key"] = pd.DataFrame({'x': (0, 1, 2, 3, 5)})

        self.assertTrue(edge1 == edge2)
        self.assertFalse(edge1 == edge3)

        # Test edges with same keys, but diff np array vals
        # edge.__eq__ calls ndarray.all(), which checks that
        # all values in an array eval to true
        edge1.__dict__["key"] = np.array([True, True, True], dtype=np.bool)
        edge2.__dict__["key"] = np.array([True, True, True], dtype=np.bool)
        edge3.__dict__["key"] = np.array([True, True, False], dtype=np.bool)

        self.assertTrue(edge1 == edge2)
        self.assertFalse(edge1 == edge3)

    def test_repr(self):
        src = node.Node()
        dst = node.Node()
        masks = pd.DataFrame()

        e = edge.Edge()
        e.source = src
        e.destination = dst

        expected = """
        Source Image Index: {}
        Destination Image Index: {}
        Available Masks: {}
        """.format(src, dst, masks)

        self.assertEqual(expected, e.__repr__())

    def test_ratio_check(self):
        # Matches is init to None
        e = edge.Edge()

        # If there are matches...
        keypoint_matches = [[0, 0, 1, 4, 5],
                            [0, 1, 1, 3, 5],
                            [0, 2, 1, 2, 5],
                            [0, 3, 1, 1, 5],
                            [0, 4, 1, 0, 5]]

        matches_df = pd.DataFrame(data=keypoint_matches, columns=['source_image', 'source_idx',
                                                                  'destination_image', 'destination_idx', 'distance'])
        e.matches = matches_df
        expected = list(od.distance_ratio(matches_df))
        e.ratio_check()
        self.assertEqual(expected, list(e.masks["ratio"]))
