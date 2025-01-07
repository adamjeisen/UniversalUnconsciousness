import numpy as np
import sys

def convert_h5_string_array(h5_file, dataset):
    """
    Convert an HDF5 string array stored as references to a list of Python strings.
    
    Args:
        h5_file: An open h5py.File object
        dataset: The HDF5 dataset containing string references
        
    Returns:
        List of Python strings if 1D input, or list of lists of Python strings if 2D input
    """
    # Handle 2D arrays
    if len(dataset.shape) > 1:
        strings = []
        for row in dataset:
            row_strings = []
            for ref in row:
                # Get the referenced data
                ref_data = h5_file[ref][()]
                # Convert numeric array to string by mapping to chars
                string = ''.join([chr(char[0]) if isinstance(char, np.ndarray) else chr(char)
                                for char in ref_data])
                row_strings.append(string)
            strings.append(row_strings)
        strings_array = np.array(strings)
        if strings_array.shape[0] == 1:
            return strings_array[0]
        elif strings_array.shape[1] == 1:
            return strings_array[:, 0]
        else:
            return strings_array
        
    # Handle 1D arrays
    else:
        refs = dataset[:]
        strings = []
        for ref in refs:
            # Get the referenced data
            ref_data = h5_file[ref][()]
            # Convert numeric array to string by mapping to chars
            string = ''.join([chr(char[0]) if isinstance(char, np.ndarray) else chr(char)
                            for char in ref_data])
            strings.append(string)
        return np.array(strings)

def convert_char_array(dataset):
    return ''.join([chr(char[0]) if isinstance(char, np.ndarray) else chr(char)
                            for char in dataset[:]])

class TransposedDatasetView(object):
    """
    This class provides a way to transpose a dataset without
    casting it into a numpy array. This way, the dataset in a file need not
    necessarily be integrally read into memory to view it in a different
    transposition.

    .. note::
        The performances depend a lot on the way the dataset was written
        to file. Depending on the chunking strategy, reading a complete 2D slice
        in an unfavorable direction may still require the entire dataset to
        be read from disk.

    :param dataset: h5py dataset
    :param transposition: List of dimension numbers in the wanted order
    """
    def __init__(self, dataset, transposition=None):
        """

        """
        super(TransposedDatasetView, self).__init__()
        self.dataset = dataset
        """original dataset"""

        self.shape = dataset.shape
        """Tuple of array dimensions"""
        self.dtype = dataset.dtype
        """Data-type of the array’s element"""
        self.ndim = len(dataset.shape)
        """Number of array dimensions"""

        size = 0
        if self.ndim:
            size = 1
            for dimsize in self.shape:
                size *= dimsize
        self.size = size
        """Number of elements in the array."""

        self.transposition = list(range(self.ndim))
        """List of dimension indices, in an order depending on the
        specified transposition. By default this is simply
        [0, ..., self.ndim], but it can be changed by specifying a different
        `transposition` parameter at initialization.

        Use :meth:`transpose`, to create a new :class:`TransposedDatasetView`
        with a different :attr:`transposition`.
        """

        if transposition is not None:
            assert len(transposition) == self.ndim
            assert set(transposition) == set(list(range(self.ndim))), \
                "Transposition must be a list containing all dimensions"
            self.transposition = transposition
            self.__sort_shape()

    def __sort_shape(self):
        """Sort shape in the order defined in :attr:`transposition`
        """
        new_shape = tuple(self.shape[dim] for dim in self.transposition)
        self.shape = new_shape

    def __sort_indices(self, indices):
        """Return array indices sorted in the order needed
        to access data in the original non-transposed dataset.

        :param indices: Tuple of ndim indices, in the order needed
            to access the view
        :return: Sorted tuple of indices, to access original data
        """
        assert len(indices) == self.ndim
        sorted_indices = tuple(idx for (_, idx) in
                               sorted(zip(self.transposition, indices)))
        return sorted_indices

    def __getitem__(self, item):
        """Handle fancy indexing with regards to the dimension order as
        specified in :attr:`transposition`

        The supported fancy-indexing syntax is explained at
        http://docs.h5py.org/en/latest/high/dataset.html#fancy-indexing.

        Additional restrictions exist if the data has been transposed:

            - numpy boolean array indexing is not supported
            - ellipsis objects are not supported

        :param item: Index, possibly fancy index (must be supported by h5py)
        :return:
        """
        # no transposition, let the original dataset handle indexing
        if self.transposition == list(range(self.ndim)):
            return self.dataset[item]

        # 1-D slicing -> n-D slicing (n=1)
        if not hasattr(item, "__len__"):
            # first dimension index is given
            item = [item]
            # following dimensions are indexed with : (all elements)
            item += [slice(0, None, 1) for _i in range(self.ndim - 1)]

        # n-dimensional slicing
        if len(item) != self.ndim:
            raise IndexError(
                "N-dim slicing requires a tuple of N indices/slices. " +
                "Needed dimensions: %d" % self.ndim)

        # get list of indices sorted in the original dataset order
        sorted_indices = self.__sort_indices(item)

        output_data_not_transposed = self.dataset[sorted_indices]

        # now we must transpose the output data
        output_dimensions = []
        frozen_dimensions = []
        for i, idx in enumerate(item):
            # slices and sequences
            if not isinstance(idx, int):
                output_dimensions.append(self.transposition[i])
            # regular integer index
            else:
                # whenever a dimension is fixed (indexed by an integer)
                # the number of output dimension is reduced
                frozen_dimensions.append(self.transposition[i])

        # decrement output dimensions that are above frozen dimensions
        for frozen_dim in reversed(sorted(frozen_dimensions)):
            for i, out_dim in enumerate(output_dimensions):
                if out_dim > frozen_dim:
                    output_dimensions[i] -= 1

        assert (len(output_dimensions) + len(frozen_dimensions)) == self.ndim
        assert set(output_dimensions) == set(range(len(output_dimensions)))

        return np.transpose(output_data_not_transposed,
                               axes=output_dimensions)

    def __array__(self, dtype=None):
        """Cast the dataset into a numpy array, and return it.

        If a transposition has been done on this dataset, return
        a transposed view of a numpy array."""
        return np.transpose(np.array(self.dataset, dtype=dtype),
                               self.transposition)

    def transpose(self, transposition=None):
        """Return a re-ordered (dimensions permutated)
        :class:`TransposedDatasetView`.

        The returned object refers to
        the same dataset but with a different :attr:`transposition`.

        :param list[int] transposition: List of dimension numbers in the wanted order
        :return: Transposed TransposedDatasetView
        """
        # by default, reverse the dimensions
        if transposition is None:
            transposition = list(reversed(self.transposition))

        return TransposedDatasetView(self.dataset,
                                     transposition)