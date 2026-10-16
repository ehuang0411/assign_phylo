"""Helper functions for HW3"""
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib.axes import Axes

"""Helper functions for Q4"""
from Bio.Align import substitution_matrices

class Node:
    def __init__(
        self,
        name: str,
        left: "Node",
        left_distance: float,
        right: "Node",
        right_distance: float,
        confidence: float = None,
    ):
        """A node in a binary tree produced by neighbor joining algorithm.

        Parameters
        ----------
        name: str
            Name of the node.
        left: Node
            Left child.
        left_distance: float
            The distance to the left child.
        right: Node
            Right child.
        right_distance: float
            The distance to the right child.
        confidence: float
            The confidence level of the split determined by the bootstrap method.
            Only used if you implement Bonus Problem 1.

        Notes
        -----
        The current public API needs to remain as it is, i.e., don't change the
        names of the properties in the template, as the tests expect this kind
        of structure. However, feel free to add any methods/properties/attributes
        that you might need in your tree construction.

        """
        self.name = name
        self.left = left
        self.left_distance = left_distance
        self.right = right
        self.right_distance = right_distance
        self.confidence = confidence


def neighbor_joining(distances: np.ndarray, labels: list) -> Node:
    """The Neighbor-Joining algorithm.

    For the same results as in the later test dendrograms;
    add new nodes to the end of the list/matrix and
    in case of ties, use np.argmin to choose the joining pair.

    Parameters
    ----------
    distances: np.ndarray
        A 2d square, symmetric distance matrix containing distances between
        data points. The diagonal entries should always be zero; d(x, x) = 0.
    labels: list
        A list of labels corresponding to entries in the distances matrix.
        Use them to set names of nodes.

    Returns
    -------
    Node
        A root node of the neighbor joining tree.

    """
    D = np.asarray(distances, dtype=float).copy()
    labels = list(labels)

    # One Node object per original taxon
    nodes = [Node(name, None, 0.0, None, 0.0) for name in labels]

    internal_count = 0

    while len(nodes) > 2:
        # These three quantities must always agree
        m = len(nodes)
        assert m == D.shape[0] == D.shape[1], (
            f"Inconsistent state: nodes={m}, D shape={D.shape}"
        )

        # Total divergence of each active node from all others
        row_sums = D.sum(axis=1)

        # Neighbour-joining criterion
        Q = (m - 2) * D - row_sums[:, None] - row_sums[None, :]
        np.fill_diagonal(Q, np.inf)

        # Choose the pair with the smallest Q value
        flat_index = np.argmin(Q)
        i, j = np.unravel_index(flat_index, Q.shape)
        i, j = int(i), int(j)

        d_ij = D[i, j]

        # Branch lengths from the joined pair to the new internal node
        left_distance = 0.5 * d_ij + (row_sums[i] - row_sums[j]) / (2 * (m - 2))
        right_distance = d_ij - left_distance

        new_node = Node(
            name=f"Node_{internal_count}",
            left=nodes[i],
            left_distance=float(left_distance),
            right=nodes[j],
            right_distance=float(right_distance),
        )
        internal_count += 1

        # Indices that remain after removing i and j
        keep = [k for k in range(m) if k != i and k != j]

        # Distance from the new node to each remaining node
        new_distances = np.array(
            [
                0.5 * (D[i, k] - left_distance) + 0.5 * (D[j, k] - right_distance)
                for k in keep
            ],
            dtype=float,
        )

        # Build the new distance matrix:
        # old remaining nodes first, then the new node last
        n_remaining = len(keep)

        D_new = np.zeros((n_remaining + 1, n_remaining + 1), dtype=float)
        D_new[:n_remaining, :n_remaining] = D[np.ix_(keep, keep)]
        D_new[:n_remaining, n_remaining] = new_distances
        D_new[n_remaining, :n_remaining] = new_distances

        D = D_new

        # Keep the node list consistent with the new distance matrix
        nodes = [nodes[k] for k in keep] + [new_node]
        labels = [labels[k] for k in keep] + [new_node.name]

    # Join the final two nodes with an artificial root
    final_distance = float(D[0, 1])

    root = Node(
        name="root",
        left=nodes[0],
        left_distance=final_distance / 2,
        right=nodes[1],
        right_distance=final_distance / 2,
    )

    return root


def plot_nj_tree(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.
        But make sure your function can work without them, for testing purposes.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree(tree=tree, ax=ax)
    >>> fig.savefig("example.png")

    """

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))

    color = kwargs.get("color", "black")
    linewidth = kwargs.get("linewidth", 1.2)
    fontsize = kwargs.get("fontsize", 9)
    leaf_colors = kwargs.get("leaf_colors", {})

    leaf_counter = 0

    def get_position(node: Node, x: float):
        nonlocal leaf_counter

        if node.left is None and node.right is None:
            y = leaf_counter
            leaf_counter += 1

            ax.text(
                x + 0.02,
                y,
                node.name,
                va="center",
                ha="left",
                fontsize=fontsize,
                color=leaf_colors.get(node.name, "black"),
            )

            return x, y

        left_x, left_y = get_position(node.left, x + node.left_distance)
        right_x, right_y = get_position(node.right, x + node.right_distance)

        y = (left_y + right_y) / 2

        ax.plot([x, left_x], [left_y, left_y], color=color, linewidth=linewidth)
        ax.plot([x, right_x], [right_y, right_y], color=color, linewidth=linewidth)
        ax.plot([x, x], [left_y, right_y], color=color, linewidth=linewidth)

        return x, y

    get_position(tree, 0.0)

    ax.set_xlabel("Hamming genetic distance")
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    return ax


def _find_a_parent_to_node(tree: Node, node: Node) -> tuple:
    """Utility function for reroot_tree"""
    stack = [tree]

    while len(stack) > 0:

        current_node = stack.pop()
        if node.name == current_node.left.name:
            return current_node, "left"
        elif node.name == current_node.right.name:
            return current_node, "right"

        stack += [
            n for n in [current_node.left, current_node.right] if n.left is not None
        ]

    return None



def _remove_child_from_parent(parent_node: Node, child_location: str) -> None:
    """Utility function for reroot_tree"""
    setattr(parent_node, child_location, None)
    setattr(parent_node, f"{child_location}_distance", 0.0)


def reroot_tree(original_tree: Node, outgroup_node: Node) -> Node:
    """A function to create a new root and invert a tree accordingly.

    This function reroots tree with nodes in original format. If you
    added any other relational parameters to your nodes, these parameters
    will not be inverted! You can modify this implementation or create
    additional functions to fix them.

    Parameters
    ----------
    original_tree: Node
        A root node of the original tree.
    outgroup_node: Node
        A Node to set as an outgroup (already included in a tree).
        Find it by it's name and then use it as parameter.

    Returns
    -------
    Node
        Inverted tree with a new root node.
    """
    tree = deepcopy(original_tree)

    parent, child_loc = _find_a_parent_to_node(tree, outgroup_node)
    distance = getattr(parent, f"{child_loc}_distance")
    _remove_child_from_parent(parent, child_loc)

    new_root = Node("new_root", parent, distance / 2, outgroup_node, distance / 2)
    child = parent

    while tree != child:
        parent, child_loc = _find_a_parent_to_node(tree, child)

        distance = getattr(parent, f"{child_loc}_distance")
        _remove_child_from_parent(parent, child_loc)

        empty_side = "left" if child.left is None else "right"
        setattr(child, f"{empty_side}_distance", distance)
        setattr(child, empty_side, parent)

        if tree.name == parent.name:
            break
        child = parent

    other_child_loc = "right" if child_loc == "left" else "left"
    other_child_distance = getattr(parent, f"{other_child_loc}_distance")

    setattr(child, f"{empty_side}_distance", other_child_distance + distance)
    setattr(child, empty_side, getattr(parent, other_child_loc))

    return new_root

def _count_leaves(node: Node) -> int:
    """Count the number of leaves in a subtree."""
    if node.left is None and node.right is None:
        return 1

    return _count_leaves(node.left) + _count_leaves(node.right)


def sort_children_by_leaves(tree: Node) -> None:
    """Sort the children of a tree by their corresponding number of leaves.

    The tree can be changed inplace.

    Parameters
    ----------
    tree: Node
        The root node of the tree.

    """
    if tree.left is None and tree.right is None:
            return
    
    sort_children_by_leaves(tree.left)
    sort_children_by_leaves(tree.right)

    left_leaves = _count_leaves(tree.left)
    right_leaves = _count_leaves(tree.right)

    if left_leaves > right_leaves:
        tree.left, tree.right = tree.right, tree.left
        tree.left_distance, tree.right_distance = (
            tree.right_distance,
            tree.left_distance,
        )


def plot_nj_tree_radial(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram
    with a radial layout.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree_radial(tree=tree, ax=ax)
    >>> fig.savefig("example_radial.png")

    """
    raise NotImplementedError()

# From Assignment, Part 1
def global_alignment(seq1, seq2, scoring_function):
    n = len(seq1)
    m = len(seq2)

    # scores [i][j] is the best score for aligning:
    # seq1[:i] with seq2[:j]
    scores = [[0.0 for _ in range(m + 1)] for _ in range(n + 1)]

    # Traceback matrix:
    # 0 = diagonal, 1 = up, 2 = left.
    traceback = [[0 for _ in range(m + 1)] for _ in range(n + 1)]

    # Initialise first column: align seq1 characters with gaps.
    for i in range(1, n + 1):
        scores[i][0] = (
            scores[i - 1][0] +
            scoring_function(seq1[i - 1], "-")
        )
        traceback[i][0] = 1

    # Initialise first row: align seq2 characters with gaps.
    for j in range(1, m + 1):
        scores[0][j] = (
            scores[0][j - 1] +
            scoring_function("-", seq2[j - 1])
        )
        traceback[0][j] = 2

    # Fill the dynamic-programming matrices.
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = (
                scores[i - 1][j - 1] +
                scoring_function(seq1[i - 1], seq2[j - 1])
            )

            up = (
                scores[i - 1][j] +
                scoring_function(seq1[i - 1], "-")
            )

            left = (
                scores[i][j - 1] +
                scoring_function("-", seq2[j - 1])
            )

            # Deliberate tie-breaking:
            # diagonal preferred over up, and up over left.
            if diagonal >= up and diagonal >= left:
                scores[i][j] = diagonal
                traceback[i][j] = 0

            elif up >= left:
                scores[i][j] = up
                traceback[i][j] = 1

            else:
                scores[i][j] = left
                traceback[i][j] = 2

    # Reconstruct the alignment by tracing from bottom right to top left.
    aligned_seq1 = []
    aligned_seq2 = []

    i = n
    j = m

    while i > 0 or j > 0:
        if i > 0 and j > 0 and traceback[i][j] == 0:
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append(seq2[j - 1])
            i -= 1
            j -= 1

        elif i > 0 and (j == 0 or traceback[i][j] == 1):
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append("-")
            i -= 1

        else:
            aligned_seq1.append("-")
            aligned_seq2.append(seq2[j - 1])
            j -= 1

    aligned_seq1.reverse()
    aligned_seq2.reverse()

    return (
        "".join(aligned_seq1),
        "".join(aligned_seq2),
        float(scores[n][m]),
    )

def scoring_function(aa_i,aa_j):
    score = [-1, 1][aa_i == aa_j]
    return (score)