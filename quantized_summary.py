import sys
import numpy as np
        
class QuantizedSummary:
    def __init__(self, dataset, original_dim):
        self.n_summaries = len(dataset)
        self.d = original_dim
        self.offsets = [0]
        self.summaries_ids = []
        self.values = []
        self.minimums = []
        self.quants = []

        inverted_pairs = [[] for _ in range(original_dim)]

        # Process each document to quantize values and build inverted index
        for doc_id, (components, values) in enumerate(dataset):
            min_val, quant, current_codes = self.quantize(values)
            self.minimums.append(min_val)
            self.quants.append(quant)

            # Add to inverted index
            for c, code in zip(components, current_codes):
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
        """
        Quantize values to 8-bit integers using min/max scaling.
        """
        # Check if values is empty
        if not values:
            return (0.0, 0.0, [])
        # Compute min and max values in the vector
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            return (min_val, 0.0, [0]*len(values))
        
        # Quantization splits the range [min ,max] into n_classes blocks of equal size (max-m)/n_clasess
        quant = (max_val - min_val) / n_classes
        query_values = []
        for v in values:
            code = int((v - min_val) / quant)
            code = code % 256  # To mimic u8 overflow behavior from Rust
            query_values.append(code)
        
        return (min_val, quant, query_values)

    def matmul_with_query(self, query_components, query_values):
        accumulator = [0.0] * self.n_summaries
        
        for qc, qv in zip(query_components, query_values):
            start = self.offsets[qc]
            end = self.offsets[qc + 1]
            
            if start == end:
                continue
            
            for i in range(start, end):
                doc_id = self.summaries_ids[i]
                code = self.values[i]
                
                # Reconstruct quantized value
                reconstructed = code * self.quants[doc_id] + self.minimums[doc_id]
                accumulator[doc_id] += reconstructed * qv
        
        return accumulator