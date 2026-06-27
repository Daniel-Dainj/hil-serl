import jax
import numpy as np
import scipy.linalg


def apply_flax_compat_shims():
    """Backfill newer pytree helpers expected by Flax on older JAX releases."""
    if not hasattr(jax.tree_util, "register_pytree_with_keys"):
        def _register_pytree_with_keys(nodetype, flatten_with_keys, unflatten_func):
            def _flatten_func(node):
                children_with_keys, aux_data = flatten_with_keys(node)
                children = tuple(child for _, child in children_with_keys)
                return children, aux_data

            return jax.tree_util.register_pytree_node(
                nodetype,
                _flatten_func,
                unflatten_func,
            )

        jax.tree_util.register_pytree_with_keys = _register_pytree_with_keys

    if not hasattr(jax.tree_util, "register_pytree_with_keys_class"):
        def _register_pytree_with_keys_class(cls):
            flatten_with_keys = getattr(cls, "tree_flatten_with_keys", None)
            if flatten_with_keys is None:
                return jax.tree_util.register_pytree_node_class(cls)

            jax.tree_util.register_pytree_with_keys(
                cls,
                flatten_with_keys,
                cls.tree_unflatten,
            )
            return cls

        jax.tree_util.register_pytree_with_keys_class = _register_pytree_with_keys_class

    if not hasattr(jax.tree_util, "GetAttrKey"):
        jax.tree_util.GetAttrKey = lambda name: ("attr", name)

    if not hasattr(jax.tree_util, "DictKey"):
        jax.tree_util.DictKey = lambda key: ("dict", key)

    if not hasattr(jax.tree_util, "SequenceKey"):
        jax.tree_util.SequenceKey = lambda idx: ("seq", idx)

    if not hasattr(jax.tree_util, "FlattenedIndexKey"):
        jax.tree_util.FlattenedIndexKey = lambda idx: ("flat", idx)

    if not hasattr(scipy.linalg, "tril"):
        scipy.linalg.tril = np.tril

    if not hasattr(scipy.linalg, "triu"):
        scipy.linalg.triu = np.triu
