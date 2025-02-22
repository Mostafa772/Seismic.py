
def binary_search_branchless(data, target):
    base = 0
    size = len(data)
    while size > 1:
        mid = size // 2
        cmp = data[base + mid - 1] < target
        base += mid if cmp else 0
        size -= mid

    return base

# Computes the size of the intersection of two unsorted lists of integers.
def intersection(s, groundtruth):
    s_set = set(s)
    size = 0
    for v in groundtruth:
        if v in s_set:
            size += 1
    return size
