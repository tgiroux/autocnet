from autocnet.examples import get_path
from autocnet.graph.network import CandidateGraph
from autocnet.io.network import load

def test_save_project(tmpdir, candidategraph):
    path = tmpdir.join('prject.proj')
    candidategraph.save(path.strpath)

    candidategraph2 = load(path.strpath)

    assert candidategraph == candidategraph2
