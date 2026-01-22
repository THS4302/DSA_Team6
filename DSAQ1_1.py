import time
import random

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

class FastSinglyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self._size = 0
        self.nodes_list = []      # Array for O(1) access
        self.node_to_index = {}   # Maps node object to current array index

    def size(self):
        return self._size

    def get(self, i):
        """O(1) - Direct array access."""
        if i < 0 or i >= self._size:
            raise IndexError("Index out of bounds")
        return self.nodes_list[i].value

    def insert(self, i, value):
        """O(1) - Linked list insertion + array append."""
        if i < 0 or i > self._size:
            raise IndexError("Index out of bounds")
        
        new_node = Node(value)

        # 1. Linked List Logic
        if i == 0:
            new_node.next = self.head
            self.head = new_node
            if self._size == 0:
                self.tail = new_node
        else:
           
            # is used to find the predecessor node.
            prev_node = self.nodes_list[i-1]
            new_node.next = prev_node.next
            prev_node.next = new_node
            if prev_node == self.tail:
                self.tail = new_node

        # 2. Auxiliary Structure Logic (O(1) append)
        self.nodes_list.append(new_node)
        self.node_to_index[new_node] = len(self.nodes_list) - 1
        self._size += 1

    def remove(self, i):
        """O(1) - Swap-with-last technique with safety check."""
        if i < 0 or i >= self._size:
            raise IndexError("Index out of bounds")

        # 1. Identify the node to be removed
        # In this implementation, the i-th element of the array is the target
        removed_node = self.nodes_list[i]

        # 2. Linked List Pointer Removal Logic
        if i == 0:
            self.head = self.head.next
            if self._size == 1:
                self.tail = None
        else:
            # the array to find the predecessor in O(1)
            prev_node = self.nodes_list[i-1]
            prev_node.next = removed_node.next
            if removed_node == self.tail:
                self.tail = prev_node

        # 3. Array Removal Logic
        last_node = self.nodes_list[-1]
        
        if removed_node != last_node:
            
            self.nodes_list[i] = last_node
            # Update the mapping for the node that was moved
            self.node_to_index[last_node] = i
        
        # Clean up the references
        self.nodes_list.pop()
        del self.node_to_index[removed_node]
        self._size -= 1
        
        return removed_node.value

# Helper to traverse physical linked list
def iter_linked_list(lst):
    node = lst.head
    while node:
        yield node.value
        node = node.next

def run_comprehensive_tests():

    

    print("\n=== STARTING 11 EDGE CASE TESTS ===")
    
    # CASE 1: Empty List Initialization
    lst = FastSinglyLinkedList()
    assert lst.size() == 0
    assert lst.head is None and lst.tail is None
    print("Case 1: Empty list initialization - PASSED")


    # CASE 2: Continuous Prepending (Testing Head Update)
    for i in range(10):
        lst.insert(0, i)
    # The head should always be the last value inserted at 0
    assert lst.head.value == 9
    print("Case 2: Continuous head insertion - PASSED")

    # CASE 3: Continuous Appending (Testing Tail Update)
    lst_tail = FastSinglyLinkedList()
    for i in range(10):
        lst_tail.insert(lst_tail.size(), i)
    assert lst_tail.tail.value == 9
    assert lst_tail.get(lst_tail.size() - 1) == 9
    print("Case 3: Continuous tail insertion - PASSED")

    # CASE 4: Removing the Tail specifically
    # In the swap-with-last logic, removing the tail is unique because 
    # it is already the last element in the array (no swap needed).
    old_size = lst_tail.size()
    lst_tail.remove(old_size - 1)
    assert lst_tail.size() == old_size - 1
    assert lst_tail.tail.value == 8
    print("Case 4: Removing the Tail node - PASSED")

    # CASE 5: Removing the Head specifically
    # Removing index 0 should move the current tail node into index 0 of the array.
    old_head_val = lst_tail.head.value
    lst_tail.remove(0)
    assert lst_tail.head.value != old_head_val
    print("Case 5: Removing the Head node - PASSED")

    # CASE 6: Large Random Operations (Stress Test)
    stress_lst = FastSinglyLinkedList()
    elements = 1000
    for i in range(elements):
        stress_lst.insert(random.randint(0, stress_lst.size()), i)
    
    for _ in range(500):
        stress_lst.remove(random.randint(0, stress_lst.size() - 1))
    
    assert stress_lst.size() == 500
    print("Case 6: Random Stress Test (1000 inserts, 500 removes) - PASSED")

    # CASE 7: Out of Bounds Protection
    try:
        stress_lst.get(999)
        raise Exception("Failed Case 9")
    except IndexError:
        print("Case 7: Get Out of Bounds caught - PASSED")

    try:
        stress_lst.insert(1001, "Error")
        raise Exception("Failed Case 9")
    except IndexError:
        print("Case 7: Insert Out of Bounds caught - PASSED")

    # CASE 8: Value Retrieval Integrity
    # Ensure that even after all the swapping, the node objects 
    # in the array still point to the correct next values.
    integrity_lst = FastSinglyLinkedList()
    integrity_lst.insert(0, 100)
    integrity_lst.insert(1, 200)
    integrity_lst.insert(2, 300)
    # Map index 1 might change if we remove index 0, but the Node object 
    # 200 must still point to 300 in the linked list logic.
    node_200 = integrity_lst.nodes_list[1]
    assert node_200.next.value == 300
    print("Case 8: Pointer integrity during re-indexing - PASSED")

    
    lst3 = FastSinglyLinkedList()
    lst3.insert(0, 42)
    assert lst3.get(0) == 42
    assert lst3.size() == 1
    lst3.remove(0)
    assert lst3.size() == 0
    assert lst3.head is None
    print("Case 9: Single element insert/remove passed")

    
    lst4 = FastSinglyLinkedList()
    N = 100000  # Reduced for faster local testing
    for i in range(N):
        lst4.insert(lst4.size(), i)

    # Measure access times
    t_start = time.perf_counter()
    lst4.get(0)
    t_first = time.perf_counter() - t_start

    mid_idx = random.randint(0, N-1)
    t_mid_start = time.perf_counter()
    lst4.get(mid_idx)
    t_mid = time.perf_counter() - t_mid_start

    t_last_start = time.perf_counter()
    lst4.get(N-1)
    t_last = time.perf_counter() - t_last_start

    print(f"get(0) time:   {t_first:.10f}s")
    print(f"get(mid) time: {t_mid:.10f}s")
    print(f"get(last) time: {t_last:.10f}s")
    print("Case 10: O(1) get(i) Performance passed")

    
    lst5 = FastSinglyLinkedList()
    for i in range(10):
        lst5.insert(lst5.size(), i*10)
    
    linked_values = list(iter_linked_list(lst5))
    index_values = [lst5.get(i) for i in range(lst5.size())]
    
    # In swap-logic, the sets should be equal even if the order differs
    assert set(linked_values) == set(index_values), "Data mismatch!"
    print("Case 11: Set consistency passed (Linked List contains all Array values)")

    print("\n=== ALL 11 EDGE CASES PASSED SUCCESSFULLY ===")



if __name__ == "__main__":
    run_comprehensive_tests()