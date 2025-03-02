import unittest
from heapq import heappush, heappushpop, nsmallest

class HeapFaiss:
    def __init__(self, k: int):
        self.k = k  # The maximum number of elements in the heap
        self.distances = []  # List to store distances (will be used as a heap)
        self.ids = []  # List to store ids corresponding to distances
        self.timestamp = 0  # Timestamp (not used in the methods directly)

    def __len__(self) -> int:
        """Returns the number of distances currently in the heap."""
        return len(self.distances)
    
    def len(self) -> int:
        """Returns the number of distances currently in the heap."""
        return len(self.distances)
    
    def add(self, distance: float, id: int):
        """Adds a new distance and its associated id to the heap."""
        self.distances.append(distance)
        self.ids.append(id)
        
        i = len(self.distances) - 1
        i_father = 0
        
        while i > 0:
            i_father = ((i + 1) >> 1) - 1
            if distance <= self.distances[i_father]:
                break
            self.distances[i] = self.distances[i_father]
            self.ids[i] = self.ids[i_father]
            i = i_father
        self.distances[i] = distance
        self.ids[i] = id
        
        
        
        
    def replace_top(self, distance: float, id: int):
        """Replaces the top (largest) distance in the heap."""

        k = len(self.distances)
        i = 0
        i1 = 0 
        i2 = 0
        
        while True:
            i2 = (i + 1) << 1
            i1 = i2 - 1
            if i1 >= k :
                break
            
            if (i2 == k) or (self.distances[i1] >= self.distances[i2]):
                if distance >= self.distances[i1]:
                    break
                self.distances[i] = self.distances[i1]
                self.ids[i] = self.ids[i1]
                i = i1
            else:
                if distance >= self.distances[i2]:
                    break
                self.distances[i] = self.distances[i2]
                self.ids[i] = self.ids[i2]
                i = i2

        self.distances[i] = distance
        self.ids[i] = id
                        
    def push_with_id(self, distance: float, doc_id: int) -> None:
        """Core method for both auto and manual IDs"""
        if len(self.distances) < self.k:
            self.add(distance, doc_id)
        else:
            if distance < self.top():
                self.replace_top(distance, doc_id)
        self.timestamp += 1
        
        
    def top(self) -> float:
        """Returns the largest distance (the root of the heap)."""
        return self.distances[0] if self.distances else None

    def is_empty(self) -> bool:
        """Checks if the heap is empty."""
        return len(self.distances) == 0

    def topk(self):
        """Returns the top-k distances sorted by decreasing distance."""
        pairs = list(zip(self.distances, self.ids))
    
        # Sort pairs by distance (first element of each tuple)
        sorted_pairs = sorted(pairs, key=lambda x: x[0])
        
        return sorted_pairs
    

    

def maintains_heap_property(heap: HeapFaiss) -> bool:
    for i in range(len(heap.distances)):
        left = 2 * i + 1
        right = 2 + i + 2
        if left < len(heap.distances) and heap.distances[i] < heap.distances[left]:
            return False
        if right < len(heap.distances) and heap.distances[i] < heap.distances[right]:
            return False
    return True



# heap = HeapFaiss(3)

# heap.add(2.0, 0)
# heap.add(3.0, 1)
# heap.add(4.0, 2)

# print(heap.top())  # Should output the top-3 distances, sorted by decreasing order

# heap.replace_top(1.0, 3)
# print(heap.top())
# assert maintains_heap_property(heap) > 0, f"It doesn't maintain the heap property"


class OnlineTopKSelector(HeapFaiss):
    def __init__(self, k):
        # Call the parent class's constructor
        super().__init__(k)
        
    def __len__(self):
        super().__len__()
    
    def len(self):
        return super().__len__()

    def push(self, distance):
        """
        Pushes a new item `distance` with the current timestamp.
        If the data structure has less than k distances, the current one is inserted.
        Otherwise, the current one replaces the largest distance stored so far, if it is smaller.

        Parameters:
        - distance (float): The distance value to insert.
        """
        # if len(self.distances) < self.k:    
        #     heappush(self.distances, (distance, self.timestamp))
        # else:
        #     heappushpop(self.distances, (distance, self.timestamp))
        # self.timestamp += 1
        if self.timestamp < self.k:
            self.add(distance=distance, id=self.timestamp)
            self.timestamp += 1
            return
        
        if distance < self.top() :
            self.replace_top(distance=distance, id=self.timestamp)
            
        self.timestamp += 1
        

    def push_with_id(self, distance, id):
        """
        Pushes a new item `distance` with a specified `id`.
        If the data structure has less than k distances, the current one is inserted.
        Otherwise, the current one replaces the largest distance stored so far, if it is smaller.

        Parameters:
        - distance (float): The distance value to insert.
        - id (int): The associated identifier.
        """
        # if len(self.distances) < self.k:
        #     heappush(self.distances, (distance, id))
        # else:
        #     heappushpop(self.distances, (distance, id))
        if self.timestamp < self.k:
            self.add(distance=distance, id=id)
            self.timestamp += 1
            return
        
        if distance < self.top() :
            self.replace_top(distance=distance, id=id)
            
        self.timestamp += 1
      
      
####################################################################################################################################
##  This method is not necessery for the implementation in Python                                                                 ##
##  Check if that's okay with the professor                                                                                      ##
####################################################################################################################################
    
    # def extend(self, distances):
        # return None
        
        
    def topk(self):
        """
        Returns the top-k distances and their timestamps.
        The method returns these top-k distances as a sorted (by decreasing distance) list of pairs.

        Returns:
        - list of tuples: Sorted list of (distance, id) pairs.
        """
        # for x in zip(self.distances, self.ids):
        #     print(x)
        x = list(zip(self.distances, self.ids))
        x.sort()
        # print(x)
        return x[:self.k]
    
#         # Example Usage:
# selector = OnlineTopKSelector(k=3)
# selector.push(2.0)
# selector.push(3.0)
# selector.push(4.0)
# selector.push(1.0)

# print("Top-k:", selector.topk())  # Should return top-3 smallest distances
# assert maintains_heap_property(selector), "it DOESNT maintain the heap property"
    
    
class TestHeapFaiss(unittest.TestCase):
    
    
    def test_default_initialization(self):
        """
        ========================
        1. Initialization Tests
        ========================

        Tests the default initialization of `HeapFaiss`.
        
        This test:
        1. Initializes a `HeapFaiss` instance with a capacity of 10.
        2. Checks if all properties are correctly initialized.
        
        Expected behavior:
        All properties should be correctly set to their initial values.
        """
        heap = HeapFaiss(10)

        self.assertEqual(len(heap.distances), 0)
        self.assertEqual(len(heap.ids), 0)
        self.assertEqual(heap.k, 10)
        self.assertEqual(heap.timestamp, 0)
     
     
     
####################################################################################################################################
##  This method is not necessery for the implementation in Python                                                                 ##
##  Check if that's okay witht the professor                                                                                      ##
####################################################################################################################################

    # def test_non_sequential_push():
         
        """   
        ========================
        2. Push Behavior Tests
        ========================

        Tests the behavior of `HeapFaiss` when distances are pushed in a non-sequential order.

        This test:
        1. Initializes a `HeapFaiss` with a capacity of 5.
        2. Pushes distances in a non-sequential order using the `extend` method.
        3. Retrieves and verifies the top-k results.

        Expected behavior:
        All pushed distances should be present in the results in ascending order with their indices.
        """
        
####################################################################################################################################
##  This method is not necessery for the implementation in Python                                                                 ##
##  Check if that's okay witht the professor                                                                                      ##
####################################################################################################################################

        
        """        
        Tests the `HeapFaiss`'s ability to handle multiple distances using the `extend` method.

        This test:
        1. Initializes a `HeapFaiss` with a capacity of 4.
        2. Pushes an array of four distances using the `extend` method.
        3. Retrieves and verifies the top-k results.

        Expected behavior:
        The distances pushed are `[4.0, 3.0, 2.0, 1.0]`. These are returned in
        ascending order with their corresponding indices.
        """
        
    def test_mutate_after_retrieval() :
        """
        Tests the behavior of `HeapFaiss` when mutating the heap after a retrieval.

        This test:
        1. Initializes a `HeapFaiss` with a capacity of 4.
        2. Extends the heap with an array of regular positive values.
        3. Retrieves the top K values without mutating them.
        4. Pushes a new value into the heap.
        5. Retrieves the top K values again and checks them against an expected result.

        Expected behavior:
        The results after the second retrieval should reflect the newly pushed value and be in the correct order.
        """
        
        
        
        





