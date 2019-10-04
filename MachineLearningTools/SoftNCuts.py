import numpy as np

from MachineLearningTools import MachineLearningManager as mlm, Neighbourhood as nbh


def brightness_weights(tensor):
    im_shape = tensor.get_shape()

    sigma_I = 0.05

    x_size = im_shape[1].value
    y_size = im_shape[2].value
    z_size = im_shape[3].value

    indices, vals, shape = nbh.gaussian_neighbour((x_size, y_size, z_size))

    rows = indices[:, 0]
    cols = indices[:, 1]
    deps = indices[:, 2]
    weight_size = x_size * y_size * z_size

    bright_image = mlm.K.reshape(tensor, shape=(-1, weight_size))
    bright_image = mlm.K.transpose(bright_image)

    Fi = mlm.K.transpose(mlm.K.tf.nn.embedding_lookup(bright_image, rows))
    Fj = mlm.K.transpose(mlm.K.tf.nn.embedding_lookup(bright_image, cols))
    Fk = mlm.K.transpose(mlm.K.tf.nn.embedding_lookup(bright_image, deps))

    bright_weight = mlm.K.tf.exp(-(Fi - Fj - Fk) ** 2 / sigma_I ** 2) * vals
    bright_weight = mlm.K.transpose(bright_weight)

    return indices, bright_weight, shape


def synchronize_axes(tensor, axes, tensor_dims=None):
    if tensor_dims is None:
        tensor_dims = len(tensor.get_shape().as_list())
    perm_axes = list(axes)
    perm_axes.extend([i for i in range(tensor_dims) if i not in axes])
    perm_tensor = mlm.K.tf.transpose(tensor, perm_axes)

    contract_axis_0_len = mlm.K.tf.shape(perm_tensor)[0]
    contract_axis_len = len(axes)
    diag_slice = mlm.K.tf.range(contract_axis_0_len)
    diag_slice = mlm.K.tf.expand_dims(diag_slice, axis=1)
    diag_slice = mlm.K.tf.tile(diag_slice, mlm.K.tf.stack([1, contract_axis_len]))

    syn_tensor = mlm.K.tf.gather_nd(perm_tensor, diag_slice)
    return syn_tensor


def soft_ncut(image, image_segment, im_weights):
    image_shape = image.get_shape()
    batch_size = image_shape[0].value
    num_class = image_segment.get_shape()[-1].value
    weight_size = image_shape[1].value * image_shape[2].value * image_shape[3].value
    image_segment = mlm.K.tf.transpose(image_segment, [0, 3, 1, 2])
    image_segment = mlm.K.reshape(image_segment, mlm.K.stack([batch_size, num_class, weight_size]))

    W_Ak = sparse_tensor_dense_tensordot(im_weights, image_segment, axes=([2], [2])) # May be ML2D? Convert to 3D1
    W_Ak = mlm.K.tf.transpose(W_Ak, [0, 2, 3, 1])
    W_Ak = synchronize_axes(W_Ak, [0, 1], tensor_dims=4)

    dis_assoc = mlm.K.tf.tensordot(W_Ak, image_segment, axes=[[2], [2]]) # May be ML2D? Convert to 3D1
    dis_assoc = synchronize_axes(dis_assoc, [0, 2], tensor_dims=4)
    dis_assoc = synchronize_axes(dis_assoc, [1, 2], tensor_dims=3)
    dis_assoc = mlm.K.tf.transpose(dis_assoc, [1, 0])
    dis_assoc = mlm.K.tf.identity(dis_assoc, name="dis_assoc")

    sum_W = mlm.K.tf.sparse_reduce_sum(im_weights, axis=2)
    assoc = mlm.K.tf.tensordot(image_segment, sum_W, axes=[2,1])
    assoc = synchronize_axes(assoc, [0,2], tensor_dims=3)
    assoc = mlm.K.tf.identity(assoc, name="assoc")

    eps = 1e-6
    soft_ncut = mlm.K.tf.cast(num_class, mlm.K.tf.float32) - \
        mlm.K.tf.reduce_sum((dis_assoc + eps) / (assoc + eps), axis=1)

    return soft_ncut


def sparse_tensor_dense_tensordot(sp_a, b, axes, name=None):
    r"""Tensor contraction of a and b along specified axes.
    Tensordot (also known as tensor contraction) sums the product of elements
    from `a` and `b` over the indices specified by `a_axes` and `b_axes`.
    The lists `a_axes` and `b_axes` specify those pairs of axes along which to
    contract the tensors. The axis `a_axes[i]` of `a` must have the same dimension
    as axis `b_axes[i]` of `b` for all `i` in `range(0, len(a_axes))`. The lists
    `a_axes` and `b_axes` must have identical length and consist of unique
    integers that specify valid axes for each of the tensors.
    This operation corresponds to `numpy.tensordot(a, b, axes)`.
    Example 1: When `a` and `b` are matrices (order 2), the case `axes = 1`
    is equivalent to matrix multiplication.
    Example 2: When `a` and `b` are matrices (order 2), the case
    `axes = [[1], [0]]` is equivalent to matrix multiplication.
    Example 3: Suppose that \\(a_{ijk}\\) and \\(b_{lmn}\\) represent two
    tensors of order 3. Then, `contract(a, b, [[0], [2]])` is the order 4 tensor
    \\(c_{jklm}\\) whose entry
    corresponding to the indices \\((j,k,l,m)\\) is given by:
    \\( c_{jklm} = \sum_i a_{ijk} b_{lmi} \\).
    In general, `order(c) = order(a) + order(b) - 2*len(axes[0])`.
    Args:
        a: `SparseTensor` of type `float32` or `float64`.
        b: `Tensor` with the same type as `a`.
        axes: Either a scalar `N`, or a list or an `int32` `Tensor` of shape [2, k].
         If axes is a scalar, sum over the last N axes of a and the first N axes
         of b in order.
         If axes is a list or `Tensor` the first and second row contain the set of
         unique integers specifying axes along which the contraction is computed,
         for `a` and `b`, respectively. The number of axes for `a` and `b` must
         be equal.
        name: A name for the operation (optional).
    Returns:
        A `Tensor` with the same type as `a`.
    Raises:
        ValueError: If the shapes of `a`, `b`, and `axes` are incompatible.
        IndexError: If the values in axes exceed the rank of the corresponding
            tensor.

    authors: kojino
    source: https://github.com/tensorflow/tensorflow/issues/9210
    """

    def _tensordot_reshape(a, axes, flipped=False):
        """Helper method to perform transpose and reshape for contraction op.
        This method is helpful in reducing `math_tf.tensordot` to `math_tf.matmul`
        using `tf.transpose` and `tf.reshape`. The method takes a
        tensor and performs the correct transpose and reshape operation for a given
        set of indices. It returns the reshaped tensor as well as a list of indices
        necessary to reshape the tensor again after matrix multiplication.
        Args:
            a: `Tensor`.
            axes: List or `int32` `Tensor` of unique indices specifying valid axes of
             `a`.
            flipped: An optional `bool`. Defaults to `False`. If `True`, the method
                assumes that `a` is the second argument in the contraction operation.
        Returns:
            A tuple `(reshaped_a, free_dims, free_dims_static)` where `reshaped_a` is
            the tensor `a` reshaped to allow contraction via `matmul`, `free_dims` is
            either a list of integers or an `int32` `Tensor`, depending on whether
            the shape of a is fully specified, and free_dims_static is either a list
            of integers and None values, or None, representing the inferred
            static shape of the free dimensions
        """
        if a.get_shape().is_fully_defined() and isinstance(axes, (list, tuple)):
            shape_a = a.get_shape().as_list()
            axes = [i if i >= 0 else i + len(shape_a) for i in axes]
            free = [i for i in range(len(shape_a)) if i not in axes]
            free_dims = [shape_a[i] for i in free]
            prod_free = int(np.prod([shape_a[i] for i in free]))
            prod_axes = int(np.prod([shape_a[i] for i in axes]))
            perm = list(axes) + free if flipped else free + list(axes)
            new_shape = [prod_axes, prod_free] if flipped else [prod_free, prod_axes]
            reshaped_a = mlm.K.tf.reshape(mlm.K.tf.transpose(a, perm), new_shape)
            return reshaped_a, free_dims, free_dims
        else:
            if a.get_shape().ndims is not None and isinstance(axes, (list, tuple)):
                shape_a = a.get_shape().as_list()
                axes = [i if i >= 0 else i + len(shape_a) for i in axes]
                free = [i for i in range(len(shape_a)) if i not in axes]
                free_dims_static = [shape_a[i] for i in free]
            else:
                free_dims_static = None
            shape_a = mlm.K.tf.shape(a)
            rank_a = mlm.K.tf.rank(a)
            axes = mlm.K.tf.convert_to_tensor(axes, dtype=mlm.K.tf.int32, name="axes")
            axes = mlm.K.tf.cast(axes >= 0, mlm.K.tf.int32) * axes + mlm.K.tf.cast(
                axes < 0, mlm.K.tf.int32) * (
                           axes + rank_a)
            free, _ = mlm.K.tf.setdiff1d(mlm.K.tf.range(rank_a), axes)
            free_dims = mlm.K.tf.gather(shape_a, free)
            axes_dims = mlm.K.tf.gather(shape_a, axes)
            prod_free_dims = mlm.K.tf.reduce_prod(free_dims)
            prod_axes_dims = mlm.K.tf.reduce_prod(axes_dims)
            perm = mlm.K.tf.concat([axes_dims, free_dims], 0)
            if flipped:
                perm = mlm.K.tf.concat([axes, free], 0)
                new_shape = mlm.K.tf.stack([prod_axes_dims, prod_free_dims])
            else:
                perm = mlm.K.tf.concat([free, axes], 0)
                new_shape = mlm.K.tf.stack([prod_free_dims, prod_axes_dims])
            reshaped_a = mlm.K.tf.reshape(mlm.K.tf.transpose(a, perm), new_shape)
            return reshaped_a, free_dims, free_dims_static

    def _tensordot_axes(a, axes):
        """Generates two sets of contraction axes for the two tensor arguments."""
        a_shape = a.get_shape()
        if isinstance(axes, mlm.K.tf.compat.integral_types):
            if axes < 0:
                raise ValueError("'axes' must be at least 0.")
            if a_shape.ndims is not None:
                if axes > a_shape.ndims:
                    raise ValueError("'axes' must not be larger than the number of "
                                     "dimensions of tensor %s." % a)
                return (list(range(a_shape.ndims - axes, a_shape.ndims)),
                        list(range(axes)))
            else:
                rank = mlm.K.tf.rank(a)
                return (range(rank - axes, rank, dtype=mlm.K.tf.int32),
                        range(axes, dtype=mlm.K.tf.int32))
        elif isinstance(axes, (list, tuple)):
            if len(axes) != 2:
                raise ValueError("'axes' must be an integer or have length 2.")
            a_axes = axes[0]
            b_axes = axes[1]
            if isinstance(a_axes, mlm.K.tf.compat.integral_types) and \
                    isinstance(b_axes, mlm.K.tf.compat.integral_types):
                a_axes = [a_axes]
                b_axes = [b_axes]
            if len(a_axes) != len(b_axes):
                raise ValueError(
                    "Different number of contraction axes 'a' and 'b', %s != %s." %
                    (len(a_axes), len(b_axes)))
            return a_axes, b_axes
        else:
            axes = mlm.K.tf.convert_to_tensor(axes, name="axes", dtype=mlm.K.tf.int32)
        return axes[0], axes[1]

    def _sparse_tensordot_reshape(a, axes, flipped=False):
        """Helper method to perform transpose and reshape for contraction op.
        This method is helpful in reducing `math_tf.tensordot` to `math_tf.matmul`
        using `tf.transpose` and `tf.reshape`. The method takes a
        tensor and performs the correct transpose and reshape operation for a given
        set of indices. It returns the reshaped tensor as well as a list of indices
        necessary to reshape the tensor again after matrix multiplication.
        Args:
            a: `Tensor`.
            axes: List or `int32` `Tensor` of unique indices specifying valid axes of
             `a`.
            flipped: An optional `bool`. Defaults to `False`. If `True`, the method
                assumes that `a` is the second argument in the contraction operation.
        Returns:
            A tuple `(reshaped_a, free_dims, free_dims_static)` where `reshaped_a` is
            the tensor `a` reshaped to allow contraction via `matmul`, `free_dims` is
            either a list of integers or an `int32` `Tensor`, depending on whether
            the shape of a is fully specified, and free_dims_static is either a list
            of integers and None values, or None, representing the inferred
            static shape of the free dimensions
        """
        if a.get_shape().is_fully_defined() and isinstance(axes, (list, tuple)):
            shape_a = a.get_shape().as_list()
            axes = [i if i >= 0 else i + len(shape_a) for i in axes]
            free = [i for i in range(len(shape_a)) if i not in axes]
            free_dims = [shape_a[i] for i in free]
            prod_free = int(np.prod([shape_a[i] for i in free]))
            prod_axes = int(np.prod([shape_a[i] for i in axes]))
            perm = list(axes) + free if flipped else free + list(axes)
            new_shape = [prod_axes, prod_free] if flipped else [prod_free, prod_axes]
            reshaped_a = mlm.K.tf.sparse_reshape(mlm.K.tf.sparse_transpose(a, perm), new_shape)
            return reshaped_a, free_dims, free_dims
        else:
            if a.get_shape().ndims is not None and isinstance(axes, (list, tuple)):
                shape_a = a.get_shape().as_list()
                axes = [i if i >= 0 else i + len(shape_a) for i in axes]
                free = [i for i in range(len(shape_a)) if i not in axes]
                free_dims_static = [shape_a[i] for i in free]
            else:
                free_dims_static = None
            shape_a = mlm.K.tf.shape(a)
            rank_a = mlm.K.tf.rank(a)
            axes = mlm.K.tf.convert_to_tensor(axes, dtype=mlm.K.tf.int32, name="axes")
            axes = mlm.K.tf.cast(axes >= 0, mlm.K.tf.int32) * axes + mlm.K.tf.cast(
                axes < 0, mlm.K.tf.int32) * (
                           axes + rank_a)
            # print(sess.run(rank_a), sess.run(axes))
            free, _ = mlm.K.tf.setdiff1d(mlm.K.tf.range(rank_a), axes)
            free_dims = mlm.K.tf.gather(shape_a, free)
            axes_dims = mlm.K.tf.gather(shape_a, axes)
            prod_free_dims = mlm.K.tf.reduce_prod(free_dims)
            prod_axes_dims = mlm.K.tf.reduce_prod(axes_dims)
            perm = mlm.K.tf.concat([axes_dims, free_dims], 0)
            if flipped:
                perm = mlm.K.tf.concat([axes, free], 0)
                new_shape = mlm.K.tf.stack([prod_axes_dims, prod_free_dims])
            else:
                perm = mlm.K.tf.concat([free, axes], 0)
                new_shape = mlm.K.tf.stack([prod_free_dims, prod_axes_dims])
            reshaped_a = mlm.K.tf.sparse_reshape(mlm.K.tf.sparse_transpose(a, perm), new_shape)
            return reshaped_a, free_dims, free_dims_static

    def _sparse_tensordot_axes(a, axes):
        """Generates two sets of contraction axes for the two tensor arguments."""
        a_shape = a.get_shape()
        if isinstance(axes, mlm.K.tf.compat.integral_types):
            if axes < 0:
                raise ValueError("'axes' must be at least 0.")
            if a_shape.ndims is not None:
                if axes > a_shape.ndims:
                    raise ValueError("'axes' must not be larger than the number of "
                                     "dimensions of tensor %s." % a)
                return (list(range(a_shape.ndims - axes, a_shape.ndims)),
                        list(range(axes)))
            else:
                rank = mlm.K.tf.rank(a)
                return (range(rank - axes, rank, dtype=mlm.K.tf.int32),
                        range(axes, dtype=mlm.K.tf.int32))
        elif isinstance(axes, (list, tuple)):
            if len(axes) != 2:
                raise ValueError("'axes' must be an integer or have length 2.")
            a_axes = axes[0]
            b_axes = axes[1]
            if isinstance(a_axes, mlm.K.tf.compat.integral_types) and \
                    isinstance(b_axes, mlm.K.tf.compat.integral_types):
                a_axes = [a_axes]
                b_axes = [b_axes]
            if len(a_axes) != len(b_axes):
                raise ValueError(
                    "Different number of contraction axes 'a' and 'b', %s != %s." %
                    (len(a_axes), len(b_axes)))
            return a_axes, b_axes
        else:
            axes = mlm.K.tf.convert_to_tensor(axes, name="axes", dtype=mlm.K.tf.int32)
        return axes[0], axes[1]

    with mlm.K.tf.name_scope(name, "SparseTensorDenseTensordot", [sp_a, b, axes]) as name:
        # a = tf.convert_to_tensor(a, name="a")
        b = mlm.K.tf.convert_to_tensor(b, name="b")
        sp_a_axes, b_axes = _sparse_tensordot_axes(sp_a, axes)
        sp_a_reshape, sp_a_free_dims, sp_a_free_dims_static = _sparse_tensordot_reshape(sp_a, sp_a_axes)
        b_reshape, b_free_dims, b_free_dims_static = _tensordot_reshape(
            b, b_axes, True)
        ab_matmul = mlm.K.tf.sparse_tensor_dense_matmul(sp_a_reshape, b_reshape)
        if isinstance(sp_a_free_dims, list) and isinstance(b_free_dims, list):
            return mlm.K.tf.reshape(ab_matmul, sp_a_free_dims + b_free_dims, name=name)
        else:
            sp_a_free_dims = mlm.K.tf.convert_to_tensor(sp_a_free_dims, dtype=mlm.K.tf.int32)
            b_free_dims = mlm.K.tf.convert_to_tensor(b_free_dims, dtype=mlm.K.tf.int32)
            product = mlm.K.tf.reshape(
                ab_matmul, mlm.K.tf.concat([sp_a_free_dims, b_free_dims], 0), name=name)
            if sp_a_free_dims_static is not None and b_free_dims_static is not None:
                product.set_shape(sp_a_free_dims_static + b_free_dims_static)
            return product

