#!/usr/bin/env python

import re
import numpy as np
import numpy.linalg as lin

lines = None
with open("dome_small_door.obj") as handle:
    lines = handle.readlines()
assert lines is not None


re_comment = re.compile("^(#|o|vt|s)")
re_vertex = re.compile("^v\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")
re_facet = re.compile("^f\s+(\d+/\d+)\s+(\d+/\d+)\s+(\d+/\d+)")

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

    # facets
    match = re_facet.match(line)
    if match is not None:
        facet = [int(xx.split("/")[0]) for xx in match.groups()]
        # print(f"ff {facet}")
        facets.append(facet)
        continue

    assert match is None
    print(f'!!! "{line}"')

vertices = np.array(vertices, dtype=float)
facets = np.array(facets, dtype=int)
facets -= 1

degrees = np.zeros(vertices.shape[0], dtype=int)
for aa, bb, cc in facets:
    degrees[aa] += 1
    degrees[bb] += 1
    degrees[cc] += 1

degree_to_vertices = {}
for kk, degree in enumerate(degrees):
    if degree not in degree_to_vertices:
        degree_to_vertices[degree] = set()
    degree_to_vertices[degree].add(kk)

print(f"{vertices.shape[0]} vertices")
for deg, verts in degree_to_vertices.items():
    print(f"** {len(verts)} with degree {deg}")

print(f"{facets.shape[0]} facets")


def edges(ff):
    aa, bb, cc = ff
    yield aa, bb
    yield bb, cc
    yield cc, aa


lls = {}
for facet in facets:
    for aa, bb in edges(facet):
        pa = vertices[aa]
        pb = vertices[bb]
        ll = lin.norm(pb - pa)
        ll = int(ll * 1e4) * 1e-4

        if ll not in lls:
            lls[ll] = set()
        edge = (aa, bb) if aa < bb else (bb, aa)
        lls[ll].add(edge)

print(f"{sum(map(len, lls.values()))} edges")
for ll, edges in lls.items():
    print(f"** {len(edges)} with length {ll:0.4f}m")


