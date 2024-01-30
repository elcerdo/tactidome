#!/usr/bin/env python

import numpy as np
import numpy.linalg as lin

from utils import load_obj, enum_edges, normalize, limit_resolution


def generate_part_listing(path, num_hexagons, num_pentagons):
    print("\n##### generate_part_listing #####")
    print(f'path "{path}"')
    vertices, facets = load_obj(path)

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

    length_to_edges = {}
    for facet in facets:
        for aa, bb in enum_edges(facet):
            pa = vertices[aa]
            pb = vertices[bb]
            ll = lin.norm(pb - pa)
            ll = limit_resolution(ll)

            if ll not in length_to_edges:
                length_to_edges[ll] = set()
            edge = (aa, bb) if aa < bb else (bb, aa)
            length_to_edges[ll].add(edge)

    print(f"{sum(map(len, length_to_edges.values()))} edges")
    for length, edges in length_to_edges.items():
        print(f"** {len(edges)} with length {length:0.4f}m")

    iter_length_edges = iter(length_to_edges.items())
    aa = next(iter_length_edges)
    bb = next(iter_length_edges)
    if aa[0] >= bb[0]:
        cc = aa
        aa = bb
        bb = cc
    short_length, short_edges = aa
    long_length, long_edges = bb
    assert short_length < long_length

    assert num_hexagons + num_pentagons == vertices.shape[0]

    return (
        num_hexagons,
        num_pentagons,
        len(long_edges),
        len(short_edges),
    )


def pivot_analysis(vertices, facets, pivot):
    for facet in facets:
        assert pivot in facet

    barycenter = (vertices.sum(axis=0) - vertices[pivot]) / (vertices.shape[0] - 1)
    print(f"pivot {vertices[pivot]}")
    print(f"barycenter {barycenter}")

    normal = normalize(vertices[pivot] - barycenter)
    print(f"normal {normal}")

    normal_proj = normal.reshape(3, 1)
    normal_proj = np.eye(3) - normal_proj @ normal_proj.T
    print(f"normal_proj\n{normal_proj}")

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
        pa = normal_proj @ vertices[facet[0]]
        pb = normal_proj @ vertices[facet[1]]
        pc = normal_proj @ vertices[facet[2]]
        aa = normalize(pb - pa)
        bb = normalize(pc - pa)
        aa_dot_nn = aa.T @ bb
        assert aa_dot_nn > 0
        angle = np.arccos(aa_dot_nn)
        if angle not in inplane_angle_to_facets:
            inplane_angle_to_facets[angle] = set()
        inplane_angle_to_facets[angle].add(facet)

    return offplane_angle_to_edges, inplane_angle_to_facets


def analyse_pentagon():
    print("\n##### analyse_pentagon #####")
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

    for angle in offplane_angles:
        assert np.abs(angle - 15.85 * np.pi / 180) < 1e-2
    for angle in inplane_angles:
        assert np.abs(angle - 2 * np.pi / 5) < 1e-2
    assert np.abs(total_inplane_angle - 2 * np.pi) < 1e-2


def analyse_hexagon():
    print("\n##### analyse_hexagon #####")
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

    for angle in offplane_angles:
        assert (np.abs(angle - np.array([15.85, 18]) * np.pi / 180) < 1e-2).any()
    for angle in inplane_angles:
        assert (np.abs(angle - np.array([63.44, 58.58]) * np.pi / 180) < 1e-2).any()
    assert np.abs(total_inplane_angle - 2 * np.pi) < 1e-2


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    analyse_pentagon()
    analyse_hexagon()
    listings = []

    def generate(name, path, aa, bb):
        listings.append((name, generate_part_listing(path, aa, bb)))

    generate("sphere", "dome_full.obj", 30, 12)
    generate("dome", "dome_closed.obj", 20, 6)
    generate("door S", "dome_small_door.obj", 20, 6)
    generate("door L", "dome_large_door.obj", 20, 5)

    # cad table
    header_line = f"{6*' '} | Hex | Pen |   o |  oo |"
    print()
    print(header_line)
    print("=" * len(header_line))
    for name, counts in listings:
        foo = (name, *counts)
        print("{:>6} | {:3d} | {:3d} | {:3d} | {:3d} |".format(*foo))

    # readme table
    header_line = f"| {6*' '} | Hex | Pen | A   | B   |"
    print()
    print(header_line)
    print("|:-------|----:|----:|----:|----:|")

    for name, counts in listings:
        foo = (name, *counts)  # FIXME
        print("| {:<6} | {:3d} | {:3d} | {:3d} | {:3d} |".format(*foo))
