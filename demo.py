from sparse_dataset import *
from inverted_index import *
# from unittest import 


config = Configuration()
config.pruning_strategy(PruningStrategy.FIXED_SIZE)
config.n_postings = 1000

data = [
    ([0, 2, 4], [1.0, 2.0, 3.0]),  # First tuple
    ([1, 3], [4.0, 5.0]),         # Second tuple
    ([0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0])  # Third tuple
]


builder = SparseDatasetBuilder()

for components, values in data:
        builder.push(components, values)
        
dataset = builder.build()

print("The length of the dataset: ", dataset.len())
assert dataset.len()==3, "The length of the toy dataset should be 3"
print("The dimension of the dataset: ", dataset.dim())
assert dataset.dim()==5, "The dimension of the toy dataset should be 5"
print("The number of the non zero components: ", dataset.nnz())
assert dataset.nnz()==9, "the number of of non zero components should be 9"


# let inverted_index = InvertedIndex::build(dataset, Configuration::default());
inverted_index = InvertedIndex.build(dataset, config)

# let result = inverted_index.search(&vec![0, 1], &vec![1.0, 2.0], 1, 5, 0.7);
result = inverted_index.search([0, 1], [1, 2], k=10, query_cut=3, heap_factor=1.0)
print("The result of the search: ", result)
# assert result[0].0 == 8.0;
# assert_eq!(result[0].1, 1);