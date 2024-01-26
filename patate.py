#!/usr/bin/env python

import numpy as np
import numpy.linalg as lin

from utils import load_obj, enum_edges, normalize, limit_resolution


def generate_part_listing():
    print("##### generate_part_listing #####")
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
            ll = limit_resolution(ll)

            if ll not in lls:
                lls[ll] = set()
            edge = (aa, bb) if aa < bb else (bb, aa)
            lls[ll].add(edge)

    print(f"{sum(map(len, lls.values()))} edges")
    for ll, ees in lls.items():
        print(f"** {len(ees)} with length {ll:0.4f}m")


def pivot_analysis(vertices, facets, pivot):
    for facet in facets:
        assert pivot in facet

    barycenter = (vertices.sum(axis=0) - vertices[pivot]) / (vertices.shape[0] - 1)
    print(f"pivot {vertices[pivot]}")
    print(f"barycenter {barycenter}")

    normal = normalize(vertices[pivot] - barycenter)
    print(f"normal {normal}")

    edges = set()
    for facet in facets:
        for edge in enum_edges(facet):
            if pivot != edge[0]:
                continue
            edges.add(edge)

    angle_to_edges = {}
    for edge in edges:
        assert edge[0] == pivot
        pa = vertices[edge[0]]
        pb = vertices[edge[1]]
        delta = normalize(pb - pa)
        normal_dot_delta = normal.T @ delta
        angle = np.arccos(normal_dot_delta)
        angle -= np.pi / 2
        angle = limit_resolution(angle)
        if angle not in angle_to_edges:
            angle_to_edges[angle] = set()
        angle_to_edges[angle].add(edge)

    return angle_to_edges


def analyse_pentagon():
    print("##### analyse_pentagon #####")
    vertices, facets = load_obj("dome_pentagon.obj")
    print(f"{vertices.shape[0]} vertices")
    print(f"{facets.shape[0]} facets")

    angle_to_edges = pivot_analysis(vertices, facets, pivot=0)

    print(f"{len(angle_to_edges)} magic angles")
    for magic_angle, magic_edges in angle_to_edges.items():
        print(f"** {magic_angle * 180 / np.pi:.2f}° x{len(magic_edges)}")

    assert len(angle_to_edges) == 1


def analyse_hexagon():
    print("##### analyse_hexagon #####")
    vertices, facets = load_obj("dome_hexagon.obj")
    print(f"{vertices.shape[0]} vertices")
    print(f"{facets.shape[0]} facets")

    angle_to_edges = pivot_analysis(vertices, facets, pivot=5)

    print(f"{len(angle_to_edges)} magic angles")
    for magic_angle, magic_edges in angle_to_edges.items():
        print(f"** {magic_angle * 180 / np.pi:.2f}° x{len(magic_edges)}")

    assert len(angle_to_edges) == 2


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    generate_part_listing()
    analyse_pentagon()
    analyse_hexagon()
