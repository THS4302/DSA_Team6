import time
import random

class _Node:
    # private Node class for internal use ADT
    def __init__(self, value: int):
        self.value = value
        self.next = None

class FastSinglyLinkedList:
    def __init__(self):
        self.head = None
        self.index_map = []  # list of Node references

    # O(1) get(i)
    def get(self, i: int) -> int:
        #O(1) access via direct index lookup
        if not (0 <= i < len(self.index_map)):
            raise IndexError("Index out of bounds")
        return self.index_map[i].value

    # O(1) for insertion at any position, finding + pointer swap
    def insert(self, i: int, value: int) -> None:
        #O(1) pointer manipulation + O(1) access to location
        if not (0 <= i <= len(self.index_map)):
            raise IndexError("Index out of bounds")

        new_node = _Node(value)

        if i == 0:
            new_node.next = self.head
            self.head = new_node
        else:
            # direct access to predecessor node in O(1)
            prev = self.index_map[i - 1]
            new_node.next = prev.next
            prev.next = new_node
        
        # keep the map in sync with the list
        self.index_map.insert(i, new_node)

    # O(1) removal at any position
    def remove(self, i: int) -> None:
        #O(1) pointer bypass + O(1) access to location
        if not (0 <= i < len(self.index_map)):
            raise IndexError("Index out of bounds")

        if i == 0:
            old_head = self.head
            self.head = self.head.next
            old_head.next = None # clean up ref
        else:
            prev = self.index_map[i - 1]
            node_to_remove = prev.next
            prev.next = node_to_remove.next
            node_to_remove.next = None # clean up ref

        self.index_map.pop(i)

    def size(self) -> int:
        return len(self.index_map)


def run_comprehensive_tests():
    lst = FastSinglyLinkedList()

    #First requirement: insert at any position
    print("=== TEST 1: Correctness & Order ===")
    lst.insert(0, 100) # [100]
    lst.insert(1, 300) # [100, 300]
    lst.insert(1, 200) # [100, 200, 300]
    lst.insert(2, 400) # [100, 200, 400, 300]
    lst.insert(3, 500) # [100, 200, 400, 500, 300]
    
    results = [lst.get(i) for i in range(lst.size())]
    print(f"List Contents: {results}")
    assert results == [100, 200, 400, 500, 300], "Error: Order is incorrect!"

    #Second requirement: remove at any position
    print("\n=== TEST 2: Removal at any position ===")
    lst.remove(3) # Remove 500
    results = [lst.get(i) for i in range(lst.size())]
    print(f"After removing index 1: {results}")
    assert results == [100, 200, 400, 300], "Error: Removal failed!"

    #Third requirement: get(i)
    print("\n=== TEST 3: O(1) Performance Benchmark ===")
    N = 50000
    print(f"Populating {N} items to test scalability...")
    for i in range(N):
        lst.insert(i, i)

    # Time to get first element
    t_start = time.perf_counter()
    lst.get(0)
    time_first = time.perf_counter() - t_start

    # Time to get a random element in the middle
    mid_idx = random.randint(0,N-1)
    t_mid = time.perf_counter()
    lst.get(mid_idx)
    time_middle=time.perf_counter() - t_mid

    # Time to get the last element
    t_end = time.perf_counter()
    lst.get(N-1)
    time_last = time.perf_counter() - t_end

    #Calc avg
    avg_time = (time_first + time_middle + time_last) / 3

    print(f"Time to get(0) [Start]:          {time_first:.10f}s")
    print(f"Time to get({mid_idx}) [Middle]:          {time_middle:.10f}s")
    print(f"Time to get({N-1}) [End]:    {time_last:.10f}s")
    
    ratio = time_last / time_first if time_first > 0 else 0
    print(f"Access Time Ratio: {ratio:.2f} (Close to 1.0 indicates O(1))")

    #test empty list 
    print("\n=== TEST 4: Empty List Edge Cases ===")
    empty = FastSinglyLinkedList()

    try:
        empty.get(0)
        assert False, "get() on empty list should fail"
    except IndexError:
        print("Correctly caught get() on empty list")

    try:
        empty.remove(0)
        assert False, "remove() on empty list should fail"
    except IndexError:
        print("Correctly caught remove() on empty list")

    #test insert and remove only element 
    print("\n=== TEST 5: Single Element Insert & Remove ===")
    lst2 = FastSinglyLinkedList()

    lst2.insert(0, 42)
    assert lst2.get(0) == 42
    assert lst2.size() == 1

    lst2.remove(0)
    assert lst2.size() == 0
    assert lst2.head is None

    print("Single element insert/remove passed")

    #insert at head repeatedly
    print("\n=== TEST 6: Repeated Head Insertions ===")
    lst3 = FastSinglyLinkedList()

    for i in range(5):
        lst3.insert(0, i)   # Always insert at head

    results = [lst3.get(i) for i in range(lst3.size())]
    print(f"List Contents: {results}")
    assert results == [4, 3, 2, 1, 0]

    #insert at tail repeatedly
    print("\n=== TEST 7: Repeated Tail Insertions ===")
    lst4 = FastSinglyLinkedList()

    for i in range(5):
        lst4.insert(lst4.size(), i)

    results = [lst4.get(i) for i in range(lst4.size())]
    print(f"List Contents: {results}")
    assert results == [0, 1, 2, 3, 4]

    #remove head, middle, tail
    print("\n=== TEST 8: Remove Head, Middle, Tail ===")
    lst5 = FastSinglyLinkedList()

    for i in range(6):
        lst5.insert(i, i)   # [0,1,2,3,4,5]

    lst5.remove(0)         # remove head -> [1,2,3,4,5]
    lst5.remove(2)         # remove middle -> [1,2,4,5]
    lst5.remove(lst5.size()-1)  # remove tail -> [1,2,4]

    results = [lst5.get(i) for i in range(lst5.size())]
    print(f"List Contents: {results}")
    assert results == [1, 2, 4]

    #linked list and index map consistency
    print("\n=== TEST 9: Linked List Consistency Check ===")
    lst6 = FastSinglyLinkedList()

    for i in range(10):
        lst6.insert(i, i * 10)

    #Traverse linked list manually
    values_from_list = []
    curr = lst6.head
    while curr:
        values_from_list.append(curr.value)
        curr = curr.next

    values_from_index = [lst6.get(i) for i in range(lst6.size())]

    print(f"From linked list: {values_from_list}")
    print(f"From index map:   {values_from_index}")

    assert values_from_list == values_from_index

    #stress test with mixed operations
    print("\n=== TEST 10: Mixed Insert/Remove Stress Test ===")
    lst7 = FastSinglyLinkedList()

    for i in range(1000):
        lst7.insert(0, i)

    for _ in range(500):
        lst7.remove(0)

    assert lst7.size() == 500
    print("Stress test passed")






if __name__ == "__main__":
    run_comprehensive_tests()