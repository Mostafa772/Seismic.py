# import heapq
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Optional, Generic, TypeVar, Union, Any
import numpy as np
import time
from enum import Enum
import multiprocessing as mp
from tqdm import tqdm


from sparse_dataset import * 
from heap_faiss import * 
from top_k_selector import * 
from quantized_summary import * 

class PostingList:
    def __init__(self, packed_postings: List[int], block_offsets: List[int], summaries: QuantizedSummary):
       
        self.packed_postings = packed_postings
        self.block_offsets = block_offsets
        self.summaries = summaries

    @staticmethod
    def pack_offset_len(offset: int, length: int) -> int:
        return (offset << 16) | length

    @staticmethod
    def unpack_offset_len(packed: int) -> Tuple[int, int]:
        return (packed >> 16, packed & 0xFFFF)

    def search(self, query: List[float], query_components: List[int], query_values: List[float],
               k: int, heap_factor: float, heap, visited: set[int],
               forward_index: Any) -> None:
        
        blocks_to_evaluate = []
        dots = self.summaries.matmul_with_query(query_components, query_values)
        for block_id, dot in enumerate(dots):
            if heap.len() >= k:
                if dot < -heap_factor * heap[0][0]:
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
                              heap, visited: set[int],
                              forward_index: Any) -> None:

        prev_offset, prev_len = self.unpack_offset_len(packed_block[0])
        
        for pack in packed_block[1:]:
            offset, length = self.unpack_offset_len(pack)
            
            if prev_offset not in visited:
                vec_components, vec_values = forward_index.get_with_offset(prev_offset, prev_len)
                distance = self.calculate_distance(query_term_ids, query_values,
                                                   vec_components, vec_values, query)
                
                visited.add(prev_offset)
                heap.push_with_id(-1.0 * distance, prev_offset)
            
            prev_offset, prev_len = offset, length
        
        if prev_offset not in visited:
            vec_components, vec_values = forward_index.get_with_offset(prev_offset, prev_len)
            distance = self.calculate_distance(query_term_ids, query_values,
                                              vec_components, vec_values, query)
            visited.add(prev_offset)
            heap.push_with_id(-1.0 * distance, prev_offset)

    def calculate_distance(self, query_ids: List[int], query_vals: List[float],
                          vec_ids: List[int], vec_vals: List[float], 
                          query: List[float]) -> float:

        if len(query_ids) < 10:  # THRESHOLD_BINARY_SEARCH
            return self.dot_product_merged(query_ids, query_vals, vec_ids, vec_vals)
        else:
            return self.dot_product_dense_sparse(query, vec_ids, vec_vals)

    @staticmethod
    def dot_product_merged(q_ids: List[int], q_vals: List[float],
                          v_ids: List[int], v_vals: List[float]) -> float:

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


    @classmethod
    def build(cls, dataset: Any, postings: List[Tuple[float, int]], config: Any) -> 'PostingList':
        """
            Gets a posting list already pruned and represents it by using a blocking
            strategy to partition postings into block and a summarization strategy to
            represents the summary of each block.
        """

        posting_list = [doc_id for (_, doc_id) in postings]

        if config.blocking.strategy == BlockingStrategy.DEFAULT:
            block_offsets = cls.fixed_size_blocking(posting_list, config.blocking.block_size)
        else:
            block_offsets = cls.fixed_size_blocking(posting_list, config.blocking.block_size)  # Adjust if other strategies
        
        # Summarization
        summaries = []
        for i in range(len(block_offsets)-1):
            block = posting_list[block_offsets[i]:block_offsets[i+1]]
            if config.summarization.strategy == SummarizationStrategy.DEFAULT:
                components, values = cls.fixed_size_summary(dataset, block, config.summarization.n_components)
            else:
                components, values = cls.energy_preserving_summary(dataset, block, config.summarization.energy_fraction)
            summaries.append((components, values))
        

        
        quant_summary = QuantizedSummary(summaries, dataset.dim())
        packed = [cls.pack_offset_len(dataset.vector_offset(doc_id), dataset.vector_len(doc_id))
                 for doc_id in posting_list]
        
        return cls(packed, block_offsets, quant_summary)

    @staticmethod
    def fixed_size_blocking(posting_list: List[int], block_size: int) -> List[int]:
        return list(range(0, len(posting_list), block_size)) + [len(posting_list)]

    @staticmethod
    def blocking_with_random_kmeans(posting_list: List[int], dataset: Any, config: Any) -> List[int]:
        raise NotImplementedError
    
    @staticmethod
    def fixed_size_summary(dataset: Any, block: List[int], n_components: int) -> Tuple[List[int], List[float]]:

        component_map = {}
        for doc_id in block:
            components, values = dataset.get(doc_id)
            for c, v in zip(components.tolist(), values.tolist()):
                component_map[int(c)] = max(component_map.get(int(c), float("-inf")), float(v))
        sorted_components = sorted(component_map.items(), key=lambda x: -x[1])[:n_components]
        sorted_components.sort(key=lambda x: x[0])
        return [c for c, _ in sorted_components], [v for _, v in sorted_components]
    
    @staticmethod
    def energy_preserving_summary(dataset: Any, block: List[int], energy_fraction: float) -> Tuple[List[int], List[float]]:
        raise NotImplementedError
    
    
# Enums for configuration strategies
class PruningStrategy(Enum):
    FIXED_SIZE = 1
    GLOBAL_THRESHOLD = 2

class BlockingStrategy(Enum):
    # Blocking strategies can be added as needed
    DEFAULT = 1

class SummarizationStrategy(Enum):
    # Summarization strategies can be added as needed
    DEFAULT = 1

@dataclass
class BlockingConfig:
    strategy: BlockingStrategy
    block_size: int = 64  # Default block size for fixed blocking

@dataclass
class SummarizationConfig:
    strategy: SummarizationStrategy
    n_components: int = 100  # For fixed size summarization
    energy_fraction: float = 0.9  # For energy preserving summarization

@dataclass
class Configuration:
    pruning: PruningStrategy = PruningStrategy.FIXED_SIZE
    blocking: BlockingStrategy = BlockingConfig(BlockingStrategy.DEFAULT)
    summarization: SummarizationStrategy = SummarizationConfig(SummarizationStrategy.DEFAULT)
    
    # Parameters for pruning strategies
    n_postings: int = 1000
    max_fraction: float = 1.0
    
    def pruning_strategy(self, pruning: PruningStrategy):
        self.pruning = pruning
        return self
    
    def blocking_strategy(self, blocking: BlockingStrategy):
        self.blocking = blocking
        return self
    
    def summarization_strategy(self, summarization: SummarizationStrategy):
        self.summarization = summarization
        return self
    

class InvertedIndex:
    def __init__(self, forward_index: SparseDataset, posting_lists: List[PostingList], config: Configuration):
        self.forward_index = forward_index
        self.posting_lists = posting_lists
        self.config = config
    
    def print_space_usage_byte(self) -> int:
        print("Space Usage:")
        forward = 0 # We should calculate the memory usage of the forward index
        print(f"\tForward Index: {forward} Bytes")
        
        postings = sum(pl.space_usage_byte() for pl in self.posting_lists)
        print(f"\tPosting Lists: {postings} Bytes")
        print(f"\tTotal: {forward + postings} Bytes")
        
        return forward + postings
    
    def search(self, query_components: List[int], query_values: List[float], 
               k: int, query_cut: int, heap_factor: float) -> List[Tuple[float, int]]:
        
        query = [0.0] * self.dim()
        for i, v in zip(query_components, query_values):
            query[i] = v
        
        heap = HeapFaiss(k)
        visited = set()  # Equivalent Rust's HashSet  
        
        # Sort query terms by score in descending order and take top query_cut
        component_scores = [(comp, val) for comp, val in zip(query_components, query_values)]
        component_scores.sort(key=lambda x: x[1], reverse=True)
        
        for component_id, _ in component_scores[:query_cut]:
            self.posting_lists[component_id].search(query=query, query_components=query_components, query_values=query_values, k=k, 
                                                    heap_factor=heap_factor, heap=heap, visited=visited, forward_index=self.forward_index)

        # Convert results to (score, doc_id) pairs
        return [(abs(dot), self.forward_index.offset_to_id(offset)) 
                for dot, offset in heap.topk()]
    
    @staticmethod
    def build(dataset: SparseDataset, config: Configuration) -> 'InvertedIndex':
        
        start_time = time.time()
        inverted_pairs = [[] for _ in range(dataset.dim())]
        for doc_id, (components, values) in dataset.iter():

            for c, score in zip(components, values):
                inverted_pairs[c].append((score, doc_id))
        
        elapsed = time.time() - start_time
        print(f"{elapsed:.2f} secs")
        
        # Apply the selected pruning strategy
        print("\tPruning postings ", end="")
        start_time = time.time()
        
        if config.pruning == PruningStrategy.FIXED_SIZE:
            InvertedIndex.fixed_pruning(inverted_pairs, config.n_postings)
        elif config.pruning == PruningStrategy.GLOBAL_THRESHOLD:
            InvertedIndex.global_threshold_pruning(inverted_pairs, config.n_postings)
            InvertedIndex.fixed_pruning(inverted_pairs, int(config.n_postings * config.max_fraction))
        
        elapsed = time.time() - start_time
        print(f"{elapsed:.2f} secs")
        
        # Build summaries and blocks for each posting list
        print("\tBuilding summaries ", end="")
        start_time = time.time()
        
        print(f"\tNumber of posting lists: {len(inverted_pairs)}")

        posting_lists = []
        for posting_list in tqdm(inverted_pairs):
            posting_lists.append(PostingList.build(dataset, posting_list, config))
        
        elapsed = time.time() - start_time
        print(f"{elapsed:.2f} secs")
        
        return InvertedIndex(dataset, posting_lists, config)
    
    @staticmethod
    def fixed_pruning(inverted_pairs: List[List[Tuple[float, int]]], n_postings: int):
        for posting_list in inverted_pairs:
            posting_list.sort(key=lambda x: x[0], reverse=True)
            del posting_list[n_postings:]
    
    @staticmethod
    def global_threshold_pruning(inverted_pairs: List[List[Tuple[float, int]]], n_postings: int):
        tot_postings = len(inverted_pairs) * n_postings
        
        # Create list of (score, docid, posting_list_id) tuples
        postings = []
        for id_posting, posting_list in enumerate(inverted_pairs):
            for score, docid in posting_list:
                postings.append((score, docid, id_posting))
            posting_list.clear()
        
        tot_postings = min(tot_postings, len(postings) - 1)
        
        # Sort by score in descending order
        postings.sort(key=lambda x: x[0], reverse=True)
        
        # Take top tot_postings
        for score, docid, id_posting in postings[:tot_postings]:
            inverted_pairs[id_posting].append((score, docid))
    
    def dim(self) -> int:
        return self.forward_index.dim()
    
    def nnz(self) -> int:
        return self.forward_index.nnz()
    
    def len(self) -> int:
        return self.forward_index.len()
    
    def is_empty(self) -> bool:
        return self.forward_index.len() == 0

    
# Create a configuration
config = Configuration()
config.pruning_strategy(PruningStrategy.FIXED_SIZE)
config.n_postings = 1000

# Create a dataset
data = [
        ([0, 2, 4], [1.0, 2.0, 3.0]),
        ([1, 3], [4.0, 5.0]),
        ([0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0])
    ]   
builder = SparseDatasetBuilder()

for components, values in data:
        builder.push(components, values)
        
dataset = builder.build()
        
# Build the index
index = InvertedIndex.build(dataset, config)
# Search (using valid components within dimension range)
try:
    # This would fail because 10 exceeds the dimension (5)
    # results = index.search([1, 10, 4], [0.5, 0.7, 0.2], k=10, query_cut=3, heap_factor=1.0)
    
    # This should work fine
    results = index.search([1, 2, 4], [0.5, 0.7, 0.2], k=10, query_cut=3, heap_factor=1.0)
    # print("Results:", results)
except ValueError as e:
    print("Error:", e)