import re
from pathlib import Path
import numpy as np



def enum_edges(ff):
    aa, bb, cc = ff
    yield aa, bb
    yield bb, cc
    yield cc, aa


def normalize(vv):
    norm = np.linalg.norm(vv)
    assert norm != 0
    return vv / norm


def limit_resolution(xx, res=1e-4):
    return int(xx / res) * res


def load_obj(path, mesh_dir = Path("meshes")):
    lines = None
    with open(mesh_dir / path) as handle:
        lines = handle.readlines()
    assert lines is not None

    re_comment = re.compile("^(#|o|vt|s)")
    re_vertex = re.compile("^v\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")
    re_facet_full = re.compile("^f\s+(\d+/\d+)\s+(\d+/\d+)\s+(\d+/\d+)")
    re_facet_simple = re.compile("^f\s+(\d+)\s+(\d+)\s+(\d+)")

    vertices = []
    facets = []
    for line in lines:
        line = line.replace("\n", "")

        # comment
        match = re_comment.match(line)
        if match is not None:
            continue

        # vertices
        match = re_vertex.match(line)
        if match is not None:
            vertex = [float(xx) for xx in match.groups()]
            # print(f"vv {vertex}")
            vertices.append(vertex)
            continue

        # full facets
        match = re_facet_full.match(line)
        if match is not None:
            facet = [int(xx.split("/")[0]) for xx in match.groups()]
            # print(f"ff {facet}")
            facets.append(facet)
            continue

        # simple facets
        match = re_facet_simple.match(line)
        if match is not None:
            facet = [int(xx) for xx in match.groups()]
            # print(f"ff {facet}")
            facets.append(facet)
            continue

        assert match is None
        print(f'!!! "{line}"')

    vertices = np.array(vertices, dtype=float)
    facets = np.array(facets, dtype=int)
    facets -= 1

    return vertices, facets
