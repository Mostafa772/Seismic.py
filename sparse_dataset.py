"""
This module provides structs to represent sparse datasets.
A **sparse vector** in a `dim`-dimensional space consists of two sequences: 

components,which are distinct values in the range [0, `dim`), and their corresponding values of type `T`.
The type `T` is typically expected to be a float type such as `f16`, `f32`, or `f64`.
The components are of type `u16`.

A dataset is a collection of sparse vectors. This module provides two representations
of a sparse dataset:
a mutable [`SparseDatasetMut`] and its immutable counterpart [`SparseDataset`].
Conversion between the two representations is straightforward, as both implement the [`From`] trait.
"""

from distances import dot_product_dense_sparse
from top_k_selector import *
import bisect
# import seismic_py

import heapq
import numpy as np

class SparseDataset:
    """
    Immutable sparse dataset for efficient storage and querying.
    """
    def __init__(self, offsets, components, values, n_vecs, d):
        self.offsets = np.array(offsets, dtype=np.uint32)
        self.components = np.array(components, dtype=np.uint16)
        self.values = np.array(values, dtype=np.float32)
        self.n_vecs = n_vecs
        self.d = d  # Dimensionality of the space

    def get(self, idx):
        """Get components and values for vector at index"""
        start = self.offsets[idx]
        end = self.offsets[idx+1]
        return self.components[start:end], self.values[start:end]

    def search(self, q_components, q_values, k):
        """Find top-k nearest neighbors using dot product similarity"""
        # Convert query to dense vector
        query = np.zeros(self.d, dtype=np.float32)
        query[q_components] = q_values
        
        scores = []
        for i in range(self.n_vecs):
            start = self.offsets[i]
            end = self.offsets[i+1]
            vec_components = self.components[start:end]
            vec_values = self.values[start:end]
            
            # Calculate dot product
            dot = np.sum(vec_values * query[vec_components])
            scores.append((-dot, i))  # Use negative for min-heap
            
        # Get top-k results
        heapq.heapify(scores)
        topk = heapq.nsmallest(k, scores)
        return [(abs(score), idx) for score, idx in topk]

    def iter(self):
        for i in range(self.len()):
            # yield i, (self.components[i], self.values[i])
            yield i, self.get(i)
    
    
    def get_with_offset(self, offset: int, length: int) -> tuple:
        """
        Returns components and values for a vector starting at the specified offset with the given length.
        
        This method is useful when the offset of the required vector is already known, which can speed up access.
        It's particularly helpful for inverted index implementations.
        
        Args:
            offset: The starting offset in the components and values arrays
            length: The number of elements to retrieve
            
        Returns:
            A tuple of (components, values) slices
            
        Raises:
            AssertionError: If offset + length exceeds the size of the components array
        """
        assert offset + length <= len(self.components), "The offset is out of range"
        
        v_components = self.components[offset:offset + length]
        v_values = self.values[offset:offset + length]
        
        return v_components, v_values
    
    
    
    
    def len(self):
        return self.n_vecs
    
    
    def dim(self):
        return self.d
    
    
    def nnz(self):
        return len(self.components)
    
    
    def quantize_f16(self):
        """Convert values to float16 and return a new SparseDataset instance."""
        values_f16 = self.values.astype(np.float16)  # Convert values to float16
        return SparseDataset(self.n_vecs, self.d, self.offsets, self.components, values_f16)
    
    
    def vector_len(self, id: int) -> int:
        if id >= self.n_vecs:
                raise IndexError(f"ID {id} out of range for dataset with {self.n_vecs} vectors")
        return self.offsets[id + 1] - self.offsets[id]
    
    
    def vector_offset(self, id: int) -> int:
        if id >= self.n_vecs:
            raise IndexError(f"ID {id} out of range for dataset with {self.n_vecs} vectors")
        return self.offsets[id]
    
    
    
    def offset_to_id(self, offset: int) -> int:
        # Use binary search to find where this offset would be inserted
        index = bisect.bisect_left(self.offsets, offset)
        
        # Check if we found an exact match within valid range
        if index < len(self.offsets) and self.offsets[index] == offset:
            # Ensure we don't return the final offset (which is total components count)
            if index < self.n_vecs:
                return index
        raise ValueError(f"Offset {offset} does not start any document")
    
class SparseDatasetBuilder:
    """
    Mutable builder for constructing sparse datasets
    """
    def __init__(self):
        self.offsets = [0]
        self.components = []
        self.values = []
        self.n_vecs = 0
        self.d = 0  # Current maximum dimension
        
    def push(self, components, values):
        """Add a new sparse vector to the dataset"""
        if len(components) != len(values):
            raise ValueError("Components and values must have same length")
            
        # Update dimensionality
        if components:
            new_d = max(components) + 1
            self.d = max(self.d, new_d)
            
        # Store components and values
        self.components.extend(components)
        self.values.extend(values)
        
        # Update offsets
        self.offsets.append(self.offsets[-1] + len(components))
        self.n_vecs += 1
        
    def build(self):
        """Create immutable SparseDataset"""
        return SparseDataset(
            offsets=self.offsets,
            components=self.components,
            values=self.values,
            n_vecs=self.n_vecs,
            d=self.d
        )

if __name__ == "__main__":
    # Build dataset
    builder = SparseDatasetBuilder()
    data = [
        ([0, 2, 4], [1.0, 2.0, 3.0]),
        ([1, 3], [4.0, 5.0]),
        ([0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0])
    ]
    
    for components, values in data:
        builder.push(components, values)
        
    dataset = builder.build()
    
    # Perform search
    query_components = [0, 2]
    query_values = [1.0, 1.0]
    knn = dataset.search(query_components, query_values, k=2)
    
    print("Top-K results:", knn) # Should output: [(4.0, 2), (3.0, 0)]
    


