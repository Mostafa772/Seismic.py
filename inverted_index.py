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

# @dataclass
# # Fixed size summarization implementation
# def fixed_size_summary(dataset: SparseDataset, block: List[int], n_components: int) -> Tuple[List[int], List[float]]:
#     """Compute a fixed size summary for a block of documents"""
#     hash_map = {}
    
#     for doc_id in block:
#         # For each component_id, store the largest value seen so far
#         for c, v in dataset.iter_vector(doc_id):
#             if c in hash_map:
#                 hash_map[c] = max(hash_map[c], v)
#             else:
#                 hash_map[c] = v
    
#     # Convert to list of tuples (component_id, value)
#     components_values = list(hash_map.items())
    
#     # Sort by decreasing scores
#     components_values.sort(key=lambda x: x[1], reverse=True)
    
#     # Take only up to n_components
#     components_values = components_values[:n_components]
    
#     # Sort by component_id to make binary search possible
#     components_values.sort(key=lambda x: x[0])
    
#     # Extract components and values
#     components = [component_id for component_id, _ in components_values]
#     values = [hash_map[c] for c in components]
    
#     return components, values

# # Fixed size blocking implementation
# def fixed_size_blocking(posting_list: List[int], block_size: int) -> List[int]:
#     """Create blocks of fixed size"""
#     # Create offsets to the beginning of each block
#     block_offsets = [i * block_size for i in range(len(posting_list) // block_size)]
    
#     # Add final offset
#     block_offsets.append(len(posting_list))
    
#     return block_offsets


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
               k: int, heap_factor: float, heap, visited: set[int],
               forward_index: Any) -> None:
        # impl PostingList::search
        blocks_to_evaluate = []
        dots = self.summaries.matmul_with_query(query_components, query_values)
        # print(heap.len())
        for block_id, dot in enumerate(dots):
            # print("lenght of the heap: ", len(heap))
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
        # impl PostingList::evaluate_posting_block
        prev_offset, prev_len = self.unpack_offset_len(packed_block[0])
        
        for pack in packed_block[1:]:
            offset, length = self.unpack_offset_len(pack)
            
            if prev_offset not in visited:
                vec_components, vec_values = forward_index.get_with_offset(prev_offset, prev_len)
                distance = self.calculate_distance(query_term_ids, query_values,
                                                   vec_components, vec_values, query)
                
                visited.add(prev_offset)
                heap.push_with_id(-1.0 * distance, prev_offset)
                # self.push_to_heap(heap, -distance, prev_offset, k)
            
            prev_offset, prev_len = offset, length
        
        if prev_offset not in visited:
            vec_components, vec_values = forward_index.get_with_offset(prev_offset, prev_len)
            distance = self.calculate_distance(query_term_ids, query_values,
                                              vec_components, vec_values, query)
            visited.add(prev_offset)
            heap.push_with_id(-1.0 * distance, prev_offset)
            # self.push_to_heap(heap, -distance, prev_offset, k)

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

    # @staticmethod
    # def push_to_heap(heap: List[Tuple[float, int]], score: float, doc_id: int, k: int) -> None:
    #     if len(heap) < k:
    #         heapq.heappush(heap, (score, doc_id))
    #     else:
    #         heapq.heappushpop(heap, (score, doc_id))
    
    @classmethod
    def build(cls, dataset: Any, postings: List[Tuple[float, int]], config: Any) -> 'PostingList':
        """
            Gets a posting list already pruned and represents it by using a blocking
            strategy to partition postings into block and a summarization strategy to
            represents the summary of each block.
        """
        # impl PostingList::build
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
        # Pack postings
        # print(dataset)
        packed = [cls.pack_offset_len(dataset.vector_offset(doc_id), dataset.vector_len(doc_id))
                 for doc_id in posting_list]
        
        return cls(packed, block_offsets, quant_summary)

    @staticmethod
    def fixed_size_blocking(posting_list: List[int], block_size: int) -> List[int]:
        # impl PostingList::fixed_size_blocking
        return list(range(0, len(posting_list), block_size)) + [len(posting_list)]

    @staticmethod
    def blocking_with_random_kmeans(posting_list: List[int], dataset: Any, config: Any) -> List[int]:
        # impl PostingList::blocking_with_random_kmeans
        raise NotImplementedError
    
    @staticmethod
    def fixed_size_summary(dataset: Any, block: List[int], n_components: int) -> Tuple[List[int], List[float]]:
        # impl PostingList::fixed_size_summary
        component_map = {}
        for doc_id in block:
            # print("dataset.get(doc_id): ", dataset.get(doc_id)[0])
            components, values = dataset.get(doc_id)
            for c, v in zip(components.tolist(), values.tolist()):
                component_map[int(c)] = max(component_map.get(int(c), float("-inf")), float(v))
            # try: 
            #     for c, v in dataset.get(doc_id):
            #         component_map[c] = max(component_map.get(c, float("-inf")), v)
            # except:
            #     c, v = dataset.get(doc_id)
            #     component_map[c] = max(component_map.get(c, float("-inf")), v)
        sorted_components = sorted(component_map.items(), key=lambda x: -x[1])[:n_components]
        sorted_components.sort(key=lambda x: x[0])
        return [c for c, _ in sorted_components], [v for _, v in sorted_components]
    
    @staticmethod
    def energy_preserving_summary(dataset: Any, block: List[int], energy_fraction: float) -> Tuple[List[int], List[float]]:
        # impl PostingList::energy_preserving_summary
        raise NotImplementedError
    
    
# Enums for configuration strategies
class PruningStrategy(Enum):
    FIXED_SIZE = 1
    GLOBAL_THRESHOLD = 2

class BlockingStrategy(Enum):
    # Add blocking strategies as needed
    DEFAULT = 1

class SummarizationStrategy(Enum):
    # Add summarization strategies as needed
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
    
# @dataclass
# class Configuration:
#     """Corresponds to Rust's #[derive(...)] pub struct Configuration"""
#     pruning: 'PruningStrategy'
#     blocking: 'BlockingStrategy'
#     summarization: 'SummarizationStrategy'

#     def pruning_strategy(self, pruning: 'PruningStrategy') -> 'Configuration':
#         """Matches impl Configuration::pruning_strategy"""
#         self.pruning = pruning
#         return self

#     # Similar methods for blocking_strategy/summarization_strategy would be added

# @dataclass
# class PruningStrategy:
#     """Example implementation for pruning strategies"""
#     strategy_type: str  # 'fixed' or 'global_threshold'
#     n_postings: int
#     max_fraction: float = 1.0



class InvertedIndex:
    def __init__(self, forward_index: SparseDataset, posting_lists: List[PostingList], config: Configuration):
        self.forward_index = forward_index
        self.posting_lists = posting_lists
        self.config = config
    
    def print_space_usage_byte(self) -> int:
        print("Space Usage:")
        forward = 0  # In a real implementation, calculate actual memory usage
        print(f"\tForward Index: {forward} Bytes")
        
        postings = sum(pl.space_usage_byte() for pl in self.posting_lists)
        print(f"\tPosting Lists: {postings} Bytes")
        print(f"\tTotal: {forward + postings} Bytes")
        
        return forward + postings
    
    def search(self, query_components: List[int], query_values: List[float], 
               k: int, query_cut: int, heap_factor: float) -> List[Tuple[float, int]]:
        
        query = [0.0] * self.dim()
        # print("query: ", query)
        # print("query_components: ", query_components)
        for i, v in zip(query_components, query_values):
            # print("i: ", i)
            query[i] = v
        
        heap = HeapFaiss(k)
        visited = set()  # Equivalent to HashSet in Rust
        
        # Sort query terms by score in descending order and take top query_cut
        component_scores = [(comp, val) for comp, val in zip(query_components, query_values)]
        component_scores.sort(key=lambda x: x[1], reverse=True)
        
        for component_id, _ in component_scores[:query_cut]:
            self.posting_lists[component_id].search(query=query, query_components=query_components, query_values=query_values, k=k, 
                                                    heap_factor=heap_factor, heap=heap, visited=visited, forward_index=self.forward_index)
            # self.posting_lists[component_id].search(
            #     query,
            #     query_components,
            #     query_values,
            #     k,
            #     heap_factor,
            #     heap,
            #     visited,
            #     self.forward_index
            # )
        
        # Convert results to (score, doc_id) pairs
        return [(abs(dot), self.forward_index.offset_to_id(offset)) 
                for dot, offset in heap.topk()]
    
    @staticmethod
    def build(dataset: SparseDataset, config: Configuration) -> 'InvertedIndex':
        # Distribute pairs (score, doc_id) to corresponding components
        # print("\tDistributing postings")
        start_time = time.time()
        inverted_pairs = [[] for _ in range(dataset.dim())]
        # print("dataset.dim(): ", dataset.dim())
        # print("dataset.get(0): ", dataset.get(0))
        for doc_id, (components, values) in dataset.iter():

            # print(f"doc_id: {doc_id}, components: {components}, values: {values}")
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

    
# print("What the heck ")   
# Create a configuration
config = Configuration()
config.pruning_strategy(PruningStrategy.FIXED_SIZE)
config.n_postings = 1000

# Create a dataset
# dataset = SparseDataset(dim=100)
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
    # This would fail because 5 exceeds the dimension (5)
    # results = index.search([1, 5, 4], [0.5, 0.7, 0.2], k=10, query_cut=3, heap_factor=1.0)
    
    # This should work fine
    results = index.search([1, 2, 4], [0.5, 0.7, 0.2], k=10, query_cut=3, heap_factor=1.0)
    # print("Results:", results)
except ValueError as e:
    print("Error:", e)
    
    
    
    
    
    
    
    
    
    
    
    
    
"""
Add proper vector quantization

Implement efficient sparse vector operations

Add disk-based persistence

Include proper error handling

Optimize critical paths with NumPy/C extensions

Add parallel processing with multiprocessing

Implement proper k-means clustering

Add memory usage tracking similar to SpaceUsage trait
"""












# class InvertedIndex:
#     """Python equivalent of pub struct InvertedIndex<T>"""
#     THRESHOLD_BINARY_SEARCH = 10  # Corresponds to Rust const

#     def __init__(self, forward_index: Any, posting_lists: List[Any], config: Configuration):
#         # Mirrors Rust struct fields
#         self.forward_index = forward_index
#         self.posting_lists = posting_lists
#         self.config = config

#     def space_usage_byte(self) -> int:
#         """Implements SpaceUsage trait equivalent"""
#         forward = self.forward_index.space_usage_byte()
#         postings = sum(pl.space_usage_byte() for pl in self.posting_lists)
#         return forward + postings

#     def print_space_usage_byte(self) -> int:
#         """Corresponds to Rust's print_space_usage_byte method"""
#         print("Space Usage:")
#         forward = self.forward_index.space_usage_byte()
#         print(f"\tForward Index: {forward} Bytes")
#         postings = sum(pl.space_usage_byte() for pl in self.posting_lists)
#         print(f"\tPosting Lists: {postings} Bytes")
#         print(f"\tTotal: {forward + postings} Bytes")
#         return forward + postings

#     def search(self, query_components: List[int], query_values: List[float],
#                k: int, query_cut: int, heap_factor: float) -> List[Tuple[float, int]]:
#         """Python version of Rust's search method"""
#         # Create dense query vector (corresponds to Rust's vec![0.0; self.dim()])
#         query = np.zeros(self.dim())
#         for i, v in zip(query_components, query_values):
#             query[i] = v

#         heap = HeapFaiss(k=k)
#         visited = set()

#         # Sort query terms by descending value (matches Rust's sorted_unstable_by)
#         sorted_terms = sorted(zip(query_components, query_values),
#                             key=lambda x: -x[1])[:query_cut]

#         for component_id, _ in sorted_terms:
#             if component_id >= len(self.posting_lists):
#                 continue
#             # Corresponds to PostingList::search call
#             self.posting_lists[component_id].search(
#                 query, query_components, query_values,
#                 k, heap_factor, heap, visited, self.forward_index
#             )

#         return [(abs(score), self.forward_index.offset_to_id(doc_id)) for score, doc_id in heap.topk()]
#         # Process results similar to Rust's heap.topk()
#         # return sorted([(abs(score), self.forward_index.offset_to_id(doc_id))
#         #              for score, doc_id in heap], reverse=True)[:k]

#     @classmethod
#     def build(cls, dataset: Any, config: Configuration) -> 'InvertedIndex':
#         """Python equivalent of Rust's build method"""
#         print("\tDistributing postings ", end="")
#         start = time.time()

#         # Create inverted pairs structure (matches Rust's Vec<Vec<(T, usize)>>)
#         inverted_pairs = [[] for _ in range(dataset.dim())]
#         for doc_id, (components, values) in enumerate(dataset.iter()):
#             for c, score in zip(components, values):
#                 inverted_pairs[c].append((score, doc_id))

#         print(f"{time.time()-start:.1f} secs")

#         # Apply pruning strategy (matches Rust's match config.pruning)
#         print("\tPruning postings ", end="")
#         start = time.time()
#         if config.pruning.strategy_type == 'fixed':
#             cls.fixed_pruning(inverted_pairs, config.pruning.n_postings)
#         elif config.pruning.strategy_type == 'global_threshold':
#             cls.global_threshold_pruning(inverted_pairs, config.pruning.n_postings)
#             cls.fixed_pruning(inverted_pairs, 
#                             int(config.pruning.n_postings * config.pruning.max_fraction))

#         print(f"{time.time()-start:.1f} secs")

#         # Build posting lists (simplified parallel processing)
#         print("\tBuilding summaries ", end="")
#         start = time.time()
#         posting_lists = [PostingList.build(dataset, plist, config) 
#                         for plist in inverted_pairs]
#         print(f"{time.time()-start:.1f} secs")

#         return cls(dataset, posting_lists, config)

#     @staticmethod
#     def fixed_pruning(inverted_pairs: List[List[Tuple[float, int]]], n_postings: int):
#         """Python version of Rust's fixed_pruning"""
#         for plist in inverted_pairs:
#             plist.sort(key=lambda x: -x[0])
#             del plist[n_postings:]

#     @staticmethod
#     def global_threshold_pruning(inverted_pairs: List[List[Tuple[float, int]]], n_postings: int):
#         """Matches Rust's global_threshold_pruning implementation"""
#         all_postings = []
#         for c, plist in enumerate(inverted_pairs):
#             all_postings.extend((score, doc_id, c) for score, doc_id in plist)
#             plist.clear()

#         all_postings.sort(key=lambda x: -x[0])
#         cutoff = min(n_postings * len(inverted_pairs), len(all_postings))
        
#         for score, doc_id, c in all_postings[:cutoff]:
#             inverted_pairs[c].append((score, doc_id))

#     # Remaining methods directly map to Rust implementations
#     def dim(self) -> int: return self.forward_index.dim()
#     def nnz(self) -> int: return self.forward_index.nnz()
#     def len(self) -> int: return self.forward_index.len()
#     def is_empty(self) -> bool: return self.forward_index.len() == 0






                # print("postings: ", postings)
        # print("posting_list: ", posting_list)
        # # Blocking implementation
        # print("config blocking: ", config.blocking)
        # if config.blocking == 'BlockingStrategy.DEFAULT':
        #     block_offsets = cls.fixed_size_blocking(posting_list, config.blocking.block_size)
        # else:
        #     block_offsets = cls.fixed_size_blocking(posting_list, config.blocking.block_size)
        
        # # Summarization
        # summaries = []
        # for i in range(len(block_offsets)-1):
        #     block = posting_list[block_offsets[i]:block_offsets[i+1]]
        #     if config.summarization.strategy == 'Summarization.DEFAULT':
        #         components, values = cls.fixed_size_summary(dataset, block, 
        #                                                    config.summarization.n_components)
        #     else:
        #         components, values = cls.energy_preserving_summary(dataset, block,
        #                                                           config.summarization.energy_fraction)
        #     summaries.append((components, values))
        # Quantization (simplified)
        # quant_summary = QuantizedSummary(
        #     n_summaries=len(postings),
        #     d=dataset.dim(),
        #     offsets=[],  # Would need proper offset calculation
        #     summaries_ids=[],  # Omitted for brevity
        #     values=[],  # Omitted for brevity
        #     minimums=[],  # Omitted for brevity
        #     quants=[]  # Omitted for brevity
        # )