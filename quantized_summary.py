import sys
import numpy as np


class QuantizedSummary:
    def __init__(self, n_summaries, d, offsets, summaries_ids, values, minimums, quants):
        """
            In the Rust implementation it is not allowed to have more than 2^(16-1) summaries 
        """
        self.n_summaries = n_summaries
        self.d = d 
        self.offsets = np.array(offsets, dtype=np.int32) if offsets is not None else np.array([], dtype=np.int32)
        self.summaries_ids = summaries_ids # There cannot be more than 2^16-1 summaries
        self.values = np.array(values, dtype=np.uint8) if values is not None else np.array([], dtype=np.uint8)
        self.minimums = np.array(minimums, dtype=np.float32) if minimums is not None else np.array([], dtype=np.float32)
        self.quants = np.array(quants, dtype=np.float32) if quants is not None else np.array([], dtype=np.float32)
        
    def space_usage_byte(self):
        total = (
            sys.getsizeof(self.n_summaries) +      
            sys.getsizeof(self.d) +
            sys.getsizeof(self.offsets) +
            sys.getsizeof(self.summaries_ids) +
            sys.getsizeof(self.values) +
            sys.getsizeof(self.minimums) +
            sys.getsizeof(self.quants) 
        )
        return total
    
    def matmul_with_query(self, query_components, query_values) -> np.array: 
        accumulator: np.array = np.array([0.0] * self.n_summaries, dtype=np.float32)
       
        for qc, qv in zip(range(query_components), query_values):
            current_offset = self.offsets[qc] 
            next_offset = self.offsets[qc + 1]
                   
    
