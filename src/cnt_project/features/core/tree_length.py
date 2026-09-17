"""Branch-aware length measurements from binary object masks."""

from __future__ import annotations

import networkx as nx
import numpy as np
from skimage.morphology import skeletonize


def skeleton_graph(
    skeleton: np.ndarray,
) -> nx.Graph:
    """Convert an 8-connected skeleton into a weighted graph."""
    graph = nx.Graph()

    pixels = set(
        map(
            tuple,
            np.argwhere(skeleton),
        )
    )

    for pixel in pixels:
        graph.add_node(pixel)

        for row_offset in (-1, 0, 1):
            for column_offset in (-1, 0, 1):
                neighbour = (
                    pixel[0] + row_offset,
                    pixel[1] + column_offset,
                )

                if (
                    neighbour in pixels
                    and neighbour != pixel
                ):
                    graph.add_edge(
                        pixel,
                        neighbour,
                        weight=np.hypot(
                            row_offset,
                            column_offset,
                        ),
                    )

    return graph


def prune_branches(
    graph: nx.Graph,
    *,
    min_length: float,
) -> nx.Graph:
    """Remove terminal skeleton branches shorter than a threshold."""
    graph = graph.copy()

    while True:
        endpoints = [
            node
            for node in graph
            if graph.degree[node] == 1
        ]

        removed = False

        for endpoint in endpoints:
            path = [endpoint]
            length = 0.0
            previous = None
            current = endpoint

            while (
                graph.degree[current] == 1
                or graph.degree[current] == 2
            ):
                neighbours = [
                    node
                    for node in graph.neighbors(current)
                    if node != previous
                ]

                if not neighbours:
                    break

                next_node = neighbours[0]

                length += graph[
                    current
                ][next_node]["weight"]

                path.append(next_node)
                previous, current = (
                    current,
                    next_node,
                )

                if graph.degree[current] != 2:
                    break

            if length < min_length:
                graph.remove_nodes_from(
                    path[:-1]
                )

                removed = True
                break

        if not removed:
            break

    return graph


def edge_covering_walk(
    graph: nx.Graph,
) -> tuple[
    list[tuple[int, int]],
    float,
]:
    """Return a shortest closed walk covering every graph edge."""
    if graph.number_of_nodes() == 0:
        return [], 0.0

    if graph.number_of_edges() == 0:
        node = next(
            iter(graph.nodes)
        )

        return [node], 0.0

    odd_nodes = [
        node
        for node in graph
        if graph.degree[node] % 2
    ]

    distances = dict(
        nx.all_pairs_dijkstra_path_length(
            graph,
            weight="weight",
        )
    )

    paths = dict(
        nx.all_pairs_dijkstra_path(
            graph,
            weight="weight",
        )
    )

    matching_graph = nx.Graph()
    matching_graph.add_nodes_from(
        odd_nodes
    )

    for index, first in enumerate(
        odd_nodes
    ):
        for second in odd_nodes[
            index + 1:
        ]:
            matching_graph.add_edge(
                first,
                second,
                weight=distances[first][second],
            )

    matching = (
        nx.algorithms.matching
        .min_weight_matching(
            matching_graph,
            weight="weight",
        )
    )

    augmented_graph = nx.MultiGraph(
        graph
    )

    for first, second in matching:
        shortest_path = paths[first][second]

        for start, end in zip(
            shortest_path,
            shortest_path[1:],
        ):
            augmented_graph.add_edge(
                start,
                end,
                weight=graph[
                    start
                ][end]["weight"],
            )

    euler_edges = list(
        nx.eulerian_circuit(
            augmented_graph,
            keys=True,
        )
    )

    walk = [
        euler_edges[0][0]
    ]

    for _, end, _ in euler_edges:
        walk.append(end)

    walk_length = augmented_graph.size(
        weight="weight"
    )

    return walk, float(walk_length)


def tree_metrics(
    graph: nx.Graph,
) -> tuple[
    float,
    float,
    float,
    list[tuple[int, int]],
    list[list[tuple[int, int]]],
]:
    """Calculate branch sum, diameter, and traversal length."""
    tree_length = float(
        graph.size(weight="weight")
    )

    diameter_length = 0.0
    diameter_path = []
    traversal_length = 0.0
    traversal_paths = []

    for component_nodes in (
        nx.connected_components(graph)
    ):
        component = graph.subgraph(
            component_nodes
        ).copy()

        endpoints = [
            node
            for node in component
            if component.degree[node] == 1
        ]

        if not endpoints:
            endpoints = list(
                component.nodes
            )

        component_diameter_length = 0.0
        component_diameter_path = [
            endpoints[0]
        ]

        for start in endpoints:
            lengths, paths = (
                nx.single_source_dijkstra(
                    component,
                    start,
                    weight="weight",
                )
            )

            for end in endpoints:
                if (
                    lengths[end]
                    > component_diameter_length
                ):
                    component_diameter_length = float(
                        lengths[end]
                    )

                    component_diameter_path = (
                        paths[end]
                    )

        (
            traversal_path,
            component_traversal_length,
        ) = edge_covering_walk(
            component
        )

        traversal_length += (
            component_traversal_length
        )

        traversal_paths.append(
            traversal_path
        )

        if (
            component_diameter_length
            > diameter_length
        ):
            diameter_length = (
                component_diameter_length
            )

            diameter_path = (
                component_diameter_path
            )

    return (
        tree_length,
        diameter_length,
        traversal_length,
        diameter_path,
        traversal_paths,
    )


def branching_skeleton_length(
    mask: np.ndarray,
    *,
    min_branch_length: float | None = None,
) -> dict:
    """
    Calculate branch-aware skeleton measurements.

    ``tree_length`` is the sum of all retained skeleton edges,
    counting every edge once.
    """
    mask = np.asarray(
        mask,
        dtype=bool,
    )

    if mask.ndim != 2:
        raise ValueError(
            "mask must be two-dimensional; "
            f"received shape {mask.shape}."
        )

    if not mask.any():
        raise ValueError(
            "mask must contain foreground pixels."
        )

    skeleton = skeletonize(
        mask
    )

    graph = skeleton_graph(
        skeleton
    )

    if min_branch_length is not None:
        graph = prune_branches(
            graph,
            min_length=min_branch_length,
        )

    (
        tree_length,
        diameter,
        traversal_length,
        diameter_path,
        traversal_paths,
    ) = tree_metrics(
        graph
    )

    tree = np.zeros_like(
        mask,
        dtype=bool,
    )

    for row, column in graph.nodes:
        tree[
            row,
            column,
        ] = True

    return {
        "skeleton": skeleton,
        "tree": tree,
        "graph": graph,
        "tree_length": tree_length,
        "diameter": diameter,
        "traversal_length": traversal_length,
        "diameter_path": diameter_path,
        "traversal_paths": traversal_paths,
    }