import heapq
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
import numpy as np

from quantized_summary import * 

@dataclass

class PostingList:
    def __init__(self, packed_postings: List[int], block_offsets: List[int], summaries: QuantizedSummary):
        # Corresponds to:
        # struct PostingList { packed_postings, block_offsets, summaries }
        self.packed_postings = packed_postings
        self.block_offsets = block_offsets
        self.summaries = summaries

    @staticmethod
    def pack_offset_len(offset: int, length: int) -> int:
        # impl PostingList::pack_offset_len
        return (offset << 16) | length

    @staticmethod
    def unpack_offset_len(packed: int) -> Tuple[int, int]:
        # impl PostingList::unpack_offset_len
        return (packed >> 16, packed & 0xFFFF)

    def search(self, query: List[float], query_components: List[int], query_values: List[float],
               k: int, heap_factor: float, heap: List[Tuple[float, int]], visited: set[int],
               forward_index: Any) -> None:
        # impl PostingList::search
        blocks_to_evaluate = []
        dots = self.summaries.matmul_with_query(query_components, query_values)
        
        for block_id, dot in enumerate(dots):
            if len(heap) >= k and dot < -heap_factor * heap[0][0]:
                continue
                
            block_start = self.block_offsets[block_id]
            block_end = self.block_offsets[block_id + 1]
            packed_block = self.packed_postings[block_start:block_end]
            
            if len(blocks_to_evaluate) == 1:
                for cur_block in blocks_to_evaluate:
                    self.evaluate_posting_block(query, query_components, query_values,
                                               cur_block, heap, visited, forward_index)
                blocks_to_evaluate.clear()
            
            blocks_to_evaluate.append(packed_block)
        
        for cur_block in blocks_to_evaluate:
            self.evaluate_posting_block(query, query_components, query_values,
                                        cur_block, heap, visited, forward_index)

    def evaluate_posting_block(self, query: List[float], query_term_ids: List[int], 
                              query_values: List[float], packed_block: List[int],
                              heap: List[Tuple[float, int]], visited: set[int],
                              forward_index: Any) -> None:
        # impl PostingList::evaluate_posting_block
        prev_offset, prev_len = self.unpack_offset_len(packed_block[0])
        
        for pack in packed_block[1:]:
            offset, length = self.unpack_offset_len(pack)
            
            if prev_offset not in visited:
                vec_components, vec_values = forward_index.get_vector(prev_offset, prev_len)
                distance = self.calculate_distance(query_term_ids, query_values,
                                                   vec_components, vec_values, query)
                
                visited.add(prev_offset)
                self.push_to_heap(heap, -distance, prev_offset, k)
            
            prev_offset, prev_len = offset, length
        
        if prev_offset not in visited:
            vec_components, vec_values = forward_index.get_vector(prev_offset, prev_len)
            distance = self.calculate_distance(query_term_ids, query_values,
                                              vec_components, vec_values, query)
            visited.add(prev_offset)
            self.push_to_heap(heap, -distance, prev_offset, k)

    def calculate_distance(self, query_ids: List[int], query_vals: List[float],
                          vec_ids: List[int], vec_vals: List[float], 
                          query: List[float]) -> float:
        # Corresponds to threshold-based calculation in Rust
        if len(query_ids) < 10:  # THRESHOLD_BINARY_SEARCH
            return self.dot_product_merged(query_ids, query_vals, vec_ids, vec_vals)
        else:
            return self.dot_product_dense_sparse(query, vec_ids, vec_vals)

    @staticmethod
    def dot_product_merged(q_ids: List[int], q_vals: List[float],
                          v_ids: List[int], v_vals: List[float]) -> float:
        # Implementation of dot_product_with_merge
        i = j = 0
        result = 0.0
        while i < len(q_ids) and j < len(v_ids):
            if q_ids[i] == v_ids[j]:
                result += q_vals[i] * v_vals[j]
                i += 1
                j += 1
            elif q_ids[i] < v_ids[j]:
                i += 1
            else:
                j += 1
        return result

    @staticmethod
    def push_to_heap(heap: List[Tuple[float, int]], score: float, doc_id: int, k: int) -> None:
        if len(heap) < k:
            heapq.heappush(heap, (score, doc_id))
        else:
            heapq.heappushpop(heap, (score, doc_id))

    @classmethod
    def build(cls, dataset: Any, postings: List[Tuple[float, int]], config: Any) -> 'PostingList':
        # impl PostingList::build
        posting_list = [doc_id for (_, doc_id) in postings]
        
        # Blocking implementation
        if config.blocking.strategy == 'fixed':
            block_offsets = cls.fixed_size_blocking(posting_list, config.blocking.block_size)
        else:
            block_offsets = cls.blocking_with_random_kmeans(posting_list, dataset, config)
        
        # Summarization
        summaries = []
        for i in range(len(block_offsets)-1):
            block = posting_list[block_offsets[i]:block_offsets[i+1]]
            if config.summarization.strategy == 'fixed':
                components, values = cls.fixed_size_summary(dataset, block, 
                                                           config.summarization.n_components)
            else:
                components, values = cls.energy_preserving_summary(dataset, block,
                                                                  config.summarization.energy_fraction)
            summaries.append((components, values))
        
        # Quantization (simplified)
        quant_summary = QuantizedSummary(
            n_summaries=len(postings),
            d=dataset.dim(),
            offsets=[],  # Would need proper offset calculation
            summaries_ids=[],  # Omitted for brevity
            values=[],  # Omitted for brevity
            minimums=[],  # Omitted for brevity
            quants=[]  # Omitted for brevity
        )
        
        # Pack postings
        packed = [cls.pack_offset_len(dataset.offset(doc_id), dataset.length(doc_id))
                 for doc_id in posting_list]
        
        return cls(packed, block_offsets, quant_summary)

    @staticmethod
    def fixed_size_blocking(posting_list: List[int], block_size: int) -> List[int]:
        # impl PostingList::fixed_size_blocking
        return list(range(0, len(posting_list), block_size)) + [len(posting_list)]

    @staticmethod
    def fixed_size_summary(dataset: Any, block: List[int], n_components: int) -> Tuple[List[int], List[float]]:
        # impl PostingList::fixed_size_summary
        component_map = {}
        for doc_id in block:
            for c, v in dataset.get_vector(doc_id):
                component_map[c] = max(component_map.get(c, -np.inf), v)
        
        sorted_components = sorted(component_map.items(), key=lambda x: -x[1])[:n_components]
        sorted_components.sort(key=lambda x: x[0])
        return [c for c, _ in sorted_components], [v for _, v in sorted_components]