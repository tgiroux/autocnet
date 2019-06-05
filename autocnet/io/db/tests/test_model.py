import pytest
from autocnet.io.db import model
from autocnet import Session, engine

@pytest.fixture
def tables():
    return engine.table_names()

@pytest.fixture
def session(tables, request):
    session = Session()

    def cleanup():
        for t in reversed(tables):
            session.execute(t.delete())
        session.commit()

    request.addfinalizer(cleanup)
    
    return session

def test_keypoints_exists(tables):
    assert model.Keypoints.__tablename__ in tables

def test_edges_exists(tables):
    assert model.Edges.__tablename__ in tables

def test_costs_exists(tables):
    assert model.Costs.__tablename__ in tables

def test_matches_exists(tables):
    assert model.Matches.__tablename__ in tables

def test_cameras_exists(tables):
    assert model.Cameras.__tablename__ in tables

def test_images_exists(tables):
    assert model.Images.__tablename__ in tables

def test_create_image_default(session):
    i = model.Images()
    session.add(i)
    session.commit()
    res_i = session.query(model.Images).first()
    print(i.id)
    print(res_i.id)

def test_image_unique(session):
    serial = 'abcde'
    i = model.Images(serial=serial)
    session.add(i)
    session.commit()
    i2 = model.Images(serial=serial)
    session.add(i2)
    session.commit(i2)
    
def test_overlay_exists(tables):
    assert model.Overlay.__tablename__ in tables

def test_points_exists(tables):
    assert model.Points.__tablename__ in tables

def test_measures_exists(tables):
    assert model.Measures.__tablename__ in tables
