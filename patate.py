#!/usr/bin/env python

import numpy as np
import numpy.linalg as lin

from utils import load_obj, enum_edges


def generate_part_listing():
    vertices, facets = load_obj("dome_small_door.obj")

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
    for dd, vvs in degree_to_vertices.items():
        print(f"** {len(vvs)} with degree {dd}")

    print(f"{facets.shape[0]} facets")

    lls = {}
    for facet in facets:
        for aa, bb in enum_edges(facet):
            pa = vertices[aa]
            pb = vertices[bb]
            ll = lin.norm(pb - pa)
            ll = int(ll * 1e4) * 1e-4

            if ll not in lls:
                lls[ll] = set()
            edge = (aa, bb) if aa < bb else (bb, aa)
            lls[ll].add(edge)

    print(f"{sum(map(len, lls.values()))} edges")
    for ll, ees in lls.items():
        print(f"** {len(ees)} with length {ll:0.4f}m")


if __name__ == "__main__":
    generate_part_listing()
