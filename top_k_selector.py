"""
A TopKSelector could be either online or offline.
Here we store distances and we report the k smallest distances.
This means that for some metrics (such as dot product) you want to
store negative distances to get the closest vector.

We report the distances and ids of the top k items
The id is just the timestamp (from 0 to number of items inserted so far)
of the item. You must be able to remap those timestamps to the original ids
if needed.
We adopt this strategy for two reasons:
- The value of k is small so we can afford the remapping;
- The number of distances to be checked is large so we want to save the
  time needed to create a vector of original ids and copy them.

An online selector, such as an implementation of a Heap,
updates the data structure after every `push`.
The current top k values can be reported efficiently after every push.

An offline selector (e.g., quickselect) may just collect every pushed
distances without doing anything. Then it can spend more time
(e.g., linear time) in computing the `topk` distances.

An online selector may be faster if a lot of distance are processed
at once.
"""

import heapq
from typing import List, Tuple

class OnlineTopKSelector:
    def __init__(self, k: int):
        """
        Creates a new empty data structure to compute top-`k` distances.

        :param k: The number of smallest distances to store.
        """
        self.k = k
        self.heap = []  # Min-heap to store (-distance, id) pairs
        self.current_id = 0  # Timestamp-like identifier for each pushed distance

    def push(self, distance: float):
        """
        Pushes a new item `distance` with the current timestamp.

        :param distance: The distance value to insert.
        """
        self.push_with_id(distance, self.current_id)
        self.current_id += 1

    def push_with_id(self, distance: float, id: int):
        """
        Pushes a new item `distance` with a specified `id` as its timestamp.

        :param distance: The distance value to insert.
        :param id: The identifier associated with the distance.
        """
        # Insert into the heap as (-distance, id) to use a min-heap for max selection.
        heapq.heappush(self.heap, (-distance, id))
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)  # Remove the largest (negative smallest) element

    def extend(self, distances: List[float]):
        """
        Pushes a list of distances.

        :param distances: A list of distance values to insert.
        """
        for distance in distances:
            self.push(distance)

    def topk(self) -> List[Tuple[float, int]]:
        """
        Returns the top-k distances and their timestamps.

        :return: A sorted (by decreasing distance) list of (distance, id) pairs.
        """
        # Convert distances back to positive and sort by decreasing distance.
        return sorted([(-distance, id) for distance, id in self.heap], reverse=True)