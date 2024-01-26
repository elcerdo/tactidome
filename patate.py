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

    offplane_angle_to_edges = {}
    for edge in edges:
        assert edge[0] == pivot
        pa = vertices[edge[0]]
        pb = vertices[edge[1]]
        delta = normalize(pb - pa)
        normal_dot_delta = normal.T @ delta
        angle = np.arccos(normal_dot_delta)
        angle -= np.pi / 2
        angle = limit_resolution(angle)
        if angle not in offplane_angle_to_edges:
            offplane_angle_to_edges[angle] = set()
        offplane_angle_to_edges[angle].add(edge)

    inplane_angle_to_facets = {}
    for facet in facets:
        while facet[0] != pivot:
            facet = [facet[2], facet[0], facet[1]]
        facet = tuple(facet)
        assert facet[0] == pivot
        pa = vertices[facet[0]]
        pb = vertices[facet[1]]
        pc = vertices[facet[2]]
        aa = normalize(pb - pa)
        bb = normalize(pc - pa)
        aa_dot_nn = aa.T @ bb
        assert aa_dot_nn > 0
        angle = np.arccos(aa_dot_nn)
        angle = limit_resolution(angle)
        if angle not in inplane_angle_to_facets:
            inplane_angle_to_facets[angle] = set()
        inplane_angle_to_facets[angle].add(facet)

    return offplane_angle_to_edges, inplane_angle_to_facets


def analyse_pentagon():
    print("##### analyse_pentagon #####")
    vertices, facets = load_obj("dome_pentagon.obj")
    print(f"{vertices.shape[0]} vertices")
    print(f"{facets.shape[0]} facets")

    offplane_angles, inplane_angles = pivot_analysis(vertices, facets, pivot=0)

    print(f"{len(offplane_angles)} off-plane angles")
    for angle, edges in offplane_angles.items():
        print(f"** {angle * 180 / np.pi:.2f}° x{len(edges)}")

    print(f"{len(inplane_angles)} in-plane angles")
    total_inplane_angle = 0
    for angle, facets in inplane_angles.items():
        print(f"** {angle * 180 / np.pi:.2f}° x{len(facets)}")
        total_inplane_angle += angle * len(facets)
    print(f"total_around_pivot {total_inplane_angle * 180 / np.pi:.2f}°")

    assert len(offplane_angles) == 1
    assert len(inplane_angles) == 1


def analyse_hexagon():
    print("##### analyse_hexagon #####")
    vertices, facets = load_obj("dome_hexagon.obj")
    print(f"{vertices.shape[0]} vertices")
    print(f"{facets.shape[0]} facets")

    offplane_angles, inplane_angles = pivot_analysis(vertices, facets, pivot=5)

    print(f"{len(offplane_angles)} off-plane angles")
    for angle, edges in offplane_angles.items():
        print(f"** {angle * 180 / np.pi:.2f}° x{len(edges)}")

    print(f"{len(inplane_angles)} in-plane angles")
    total_inplane_angle = 0
    for angle, facets in inplane_angles.items():
        print(f"** {angle * 180 / np.pi:.2f}° x{len(facets)}")
        total_inplane_angle += angle * len(facets)
    print(f"total_around_pivot {total_inplane_angle * 180 / np.pi:.2f}°")

    assert len(offplane_angles) == 2
    assert len(inplane_angles) == 2


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    generate_part_listing()
    analyse_pentagon()
    analyse_hexagon()
