from utils import binary_search_branchless

"""
 Computes the dot product between a dense query and a sparse vector.
 Before using this function, the query must be made dense. This is much faster
 than computing the dot product with a "merge" style.

 # Arguments

 * `query` - The dense query vector.
 * `v_components` - The indices of the non-zero components in the vector.
 * `v_values` - The values of the non-zero components in the vector.

 # Returns

 The dot product between the query and the vector.

 # Examples
 ```

 use seismic::distances::dot_product_dense_sparse;

 let query = [1.0, 2.0, 3.0, 0.0];
 let v_components = [0, 2, 3];
 let v_values = [1.0, 1.0, 1.5];

 let result = dot_product_dense_sparse(&query, &v_components, &v_values);
 assert_eq!(result, 4.0);
 ```
 """
##############################################################
#### THIS CODE IS SUPPOSED TO USE SIMDs CHECK IF IT DOES #####
##############################################################
def dot_product_dense_sparse(query, v_components, v_values):
    N_LANES = 4

    result = [0.0] * N_LANES
    chunk_iter = zip(v_components, v_values)

    # Process in chunks
    for chunk in zip(*[chunk_iter] * N_LANES):
        for i in range(N_LANES):
            result[i] += query[chunk[i][0]] * chunk[i][1]

    l = len(v_components)
    rem = l % N_LANES

    # Process the remainder
    if rem > 0:
        for i in range(l - rem, l):
            result[0] += query[v_components[i]] * v_values[i]

    return sum(result)


"""
 Computes the dot product between a sparse query and a sparse vector using binary search.
 This function should be used when the query has just a few components.
 Both the query's and vector's terms must be sorted by id.

 # Arguments

 * `query_term_ids` - The ids of the query terms.
 * `query_values` - The values of the query terms.
 * `v_terms_ids` - The ids of the vector terms.
 * `v_values` - The values of the vector terms.

 # Returns

 The dot product between the query and the vector.

 # Examples

 ```
 use seismic::distances::dot_product_with_binary_search;

 let query_term_ids = [1, 2, 7];
 let query_values = [1.0, 1.0, 1.0];
 let v_term_ids = [0, 1, 2, 3, 4];
 let v_values = [0.1, 1.0, 1.0, 1.0, 0.5];

 let result = dot_product_with_binary_search(&query_term_ids, &query_values, &v_term_ids, &v_values);
 assert_eq!(result, 2.0);
 ```
"""
def dot_product_with_binary_search(query_term_ids, query_values, v_term_ids, v_values):
    result = 0.0
    
    for term_id, value in zip(query_term_ids, query_values):
        # Let's use a branchless binary search
        i = binary_search_branchless(data=v_term_ids, target=term_id)

        # SAFETY: result of binary search is always smaller than len(v_term_id) and len(v_values)
        cmp = v_term_ids[i] == term_id
        result += value * v_values[i] if cmp else 0
 
    return result

"""
 Computes the dot product between a query and a vector using merge style.
 This function should be used when the query has just a few components.
 Both the query's and vector's terms must be sorted by id.

 # Arguments

 * `query_term_ids` - The ids of the query terms.
 * `query_values` - The values of the query terms.
 * `v_term_ids` - The ids of the vector terms.
 * `v_values` - The values of the vector terms.

 # Returns

 The dot product between the query and the vector.

 # Examples

 ```
 use seismic::distances::dot_product_with_merge;

 let query_term_ids = [1, 2, 7];
 let query_values = [1.0, 1.0, 1.0];
 let v_term_ids = [0, 1, 2, 3, 4];
 let v_values = [0.1, 1.0, 1.0, 1.0, 0.5];

 let result = dot_product_with_merge(&query_term_ids, &query_values, &v_term_ids, &v_values);
 assert_eq!(result, 2.0);
 ```
"""

def dot_product_with_merge(query_term_ids, query_values, v_term_ids, v_values):
    
    result = 0.0
    i = 0
    for q_id, q_v in zip(query_term_ids, query_values):
        while i < len(v_term_ids) and v_term_ids[i] < q_id :
            i += 1
            if i == len(v_term_ids):
                break
            if v_term_ids[i] == q_id:
                result += v_values[i] * q_v

    return result



