import sys
import numpy as np


# class QuantizedSummary:
#     def __init__(self, n_summaries, d, offsets, summaries_ids, values, minimums, quants):
#         """
#             In the Rust implementation it is not allowed to have more than 2^(16-1) summaries 
#         """
#         self.n_summaries = n_summaries
#         self.d = d 
#         self.offsets = np.array(offsets, dtype=np.int32) if offsets is not None else np.array([], dtype=np.int32)
#         self.summaries_ids = summaries_ids # There cannot be more than 2^16-1 summaries
#         self.values = np.array(values, dtype=np.uint8) if values is not None else np.array([], dtype=np.uint8)
#         self.minimums = np.array(minimums, dtype=np.float32) if minimums is not None else np.array([], dtype=np.float32)
#         self.quants = np.array(quants, dtype=np.float32) if quants is not None else np.array([], dtype=np.float32)
        
#     def space_usage_byte(self):
#         total = (
#             sys.getsizeof(self.n_summaries) +      
#             sys.getsizeof(self.d) +
#             sys.getsizeof(self.offsets) +
#             sys.getsizeof(self.summaries_ids) +
#             sys.getsizeof(self.values) +
#             sys.getsizeof(self.minimums) +
#             sys.getsizeof(self.quants) 
#         )
#         return total
    
#     def matmul_with_query(self, query_components, query_values) -> np.array: 
#         accumulator: np.array = np.array([0.0] * self.n_summaries, dtype=np.float32)
       
#         for qc, qv in zip(range(query_components), query_values):
#             current_offset = self.offsets[qc] 
#             next_offset = self.offsets[qc + 1]
            
#             if next_offset - current_offset == 0:
#                 continue
#             current_summaries_id = self.summaries_ids[current_offset:next_offset]
#             current_values = self.values[current_offset:next_offset]
            
#             for (s_id, v) in zip(range(current_summaries_id), current_values):
#                 val = v * self.quants[s_id] + self.minimums[s_id] # why do we use the minimums here ?
         
#                 accumulator[s_id] += val * qv
            
#         return accumulator            
    
#     def new(dataset, original_dim):
#         """ We need the original dim because the summaries for the current posting list may not
#          contain all the components. An alternative is to use an HashMap to map the components. """
#         inverted_pairs = np.array(original_dim, dtype=np.float32)
#         for _ in range(len(original_dim)):
#             inverted_pairs[_] = np.array([], dtype=np.float32)
            
#         n_classes = 256
        
#         minimums = np.array(len(inverted_pairs))
#         quants = np.array(len(inverted_pairs))
        
#         for doc_id, (components, values) in enumerate(dataset):
#             minimum, quant, current_codes = quantize(values, n_classes)
#             minimums.append(minimum)
#             quants.append(quant)
            
            
            

# def quantize()
        
class QuantizedSummary:
    def __init__(self, dataset, original_dim):

        self.n_summaries = len(dataset)
        self.d = original_dim
        self.offsets = [0]
        self.summaries_ids = []
        self.values = []
        self.minimums = []
        self.quants = []

        # Initialize inverted index: list per feature containing (code, doc_id)
        inverted_pairs = [[] for _ in range(original_dim)]

        # Process each document to quantize values and build inverted index
        for doc_id, (components, values) in enumerate(dataset):
            min_val, quant, codes = self.quantize(values)
            self.minimums.append(min_val)
            self.quants.append(quant)

            # Add to inverted index
            for c, code in zip(components, codes):
                inverted_pairs[c].append((code, doc_id))

        # Build flattened arrays from inverted index
        for c in range(original_dim):
            # Add entries for current feature
            for code, doc_id in inverted_pairs[c]:
                self.values.append(code)
                self.summaries_ids.append(doc_id)
            self.offsets.append(len(self.summaries_ids))

    @staticmethod
    def quantize(values, n_classes=256):
        """Quantize values to 8-bit integers using min/max scaling."""
        if not values:
            return (0.0, 0.0, [])
        
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            return (min_val, 0.0, [0]*len(values))
        
        quant = (max_val - min_val) / n_classes
        codes = []
        for v in values:
            code = int((v - min_val) / quant)
            code = code % 256  # Mimic u8 overflow behavior from Rust
            codes.append(code)
        
        return (min_val, quant, codes)

    def matmul_with_query(self, query_components, query_values):
        accumulator = [0.0] * self.n_summaries
        
        for qc, qv in zip(query_components, query_values):
            # Get range for this feature
            start = self.offsets[qc]
            end = self.offsets[qc + 1]
            
            # Process all entries for this feature
            for i in range(start, end):
                doc_id = self.summaries_ids[i]
                code = self.values[i]
                
                # Reconstruct quantized value
                reconstructed = code * self.quants[doc_id] + self.minimums[doc_id]
                accumulator[doc_id] += reconstructed * qv
        
        return accumulator