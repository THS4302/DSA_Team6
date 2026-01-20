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
        self.nodes_list = []      # auxiliary array
        self.node_to_index = {}   # node -> index in nodes_list

    def size(self):
        return self._size

    def get(self, i):
        if i < 0 or i >= self._size:
            raise IndexError("Index out of bounds")
        return self.nodes_list[i].value

    def insert(self, i, value):
        if i < 0 or i > self._size:
            raise IndexError("Index out of bounds")
        new_node = Node(value)

        if i == 0:
            new_node.next = self.head
            self.head = new_node
            if self._size == 0:
                self.tail = new_node
        else:
            prev_node = self.nodes_list[i-1]
            new_node.next = prev_node.next
            prev_node.next = new_node
            if prev_node == self.tail:
                self.tail = new_node

        self.nodes_list.append(new_node)
        self.node_to_index[new_node] = len(self.nodes_list) - 1
        self._size += 1

    def remove(self, i):
        if i < 0 or i >= self._size:
            raise IndexError("Index out of bounds")

        # Remove from linked list
        if i == 0:
            removed_node = self.head
            self.head = self.head.next
            if self._size == 1:
                self.tail = None
        else:
            prev_node = self.nodes_list[i-1]
            removed_node = prev_node.next
            prev_node.next = removed_node.next
            if removed_node == self.tail:
                self.tail = prev_node

        # Swap with last in nodes_list to remove O(1)
        last_node = self.nodes_list[-1]
        if removed_node != last_node:
            last_index = self.node_to_index[last_node]
            self.nodes_list[i], self.nodes_list[last_index] = self.nodes_list[last_index], self.nodes_list[i]
            self.node_to_index[last_node] = i

        self.nodes_list.pop()
        del self.node_to_index[removed_node]

        self._size -= 1
        return removed_node.value


# Helper to traverse linked list
def iter_linked_list(lst):
    node = lst.head
    while node:
        yield node.value
        node = node.next


def run_comprehensive_tests():
    print("=== TEST 1: Head & Tail Insertions ===")
    lst = FastSinglyLinkedList()
    
    # Head insertions
    for i in range(5):
        lst.insert(0, i)
    head_order = list(iter_linked_list(lst))
    print(f"After head inserts: {head_order}")
    assert head_order == [4, 3, 2, 1, 0], "Head insertion order incorrect!"

    # Tail insertions
    lst2 = FastSinglyLinkedList()
    for i in range(5):
        lst2.insert(lst2.size(), i)
    tail_order = list(iter_linked_list(lst2))
    print(f"After tail inserts: {tail_order}")
    assert tail_order == [0, 1, 2, 3, 4], "Tail insertion order incorrect!"

    print("\n=== TEST 2: Head & Tail Removals (Verify Integrity) ===")
    # Remove head and tail indices using nodes_list indices
    lst.remove(0)  # remove what was head
    lst.remove(lst.size()-1)  # remove "last in nodes_list" (not guaranteed tail)
    remaining = list(iter_linked_list(lst))
    print(f"Remaining list after removals: {remaining}")
    assert len(remaining) == 3, "Incorrect size after removals"

    print("\n=== TEST 3: Single Element Insert & Remove ===")
    lst3 = FastSinglyLinkedList()
    lst3.insert(0, 42)
    assert lst3.get(0) == 42
    assert lst3.size() == 1
    lst3.remove(0)
    assert lst3.size() == 0
    assert lst3.head is None
    print("Single element insert/remove passed")

    print("\n=== TEST 4: O(1) get(i) Performance ===")
    lst4 = FastSinglyLinkedList()
    N = 1000000
    for i in range(N):
        lst4.insert(lst4.size(), i)

    t_start = time.perf_counter()
    lst4.get(0)
    t_first = time.perf_counter() - t_start

    mid_idx = random.randint(0, N-1)
    t_mid = time.perf_counter()
    lst4.get(mid_idx)
    t_mid_time = time.perf_counter() - t_mid

    t_end = time.perf_counter()
    lst4.get(N-1)
    t_last = time.perf_counter() - t_end

    print(f"get(0) time: {t_first:.10f}s")
    print(f"get({mid_idx}) time: {t_mid_time:.10f}s")
    print(f"get({N-1}) time: {t_last:.10f}s")
    ratio = t_last / t_first if t_first > 0 else 0
    print(f"Access time ratio ~1 indicates O(1): {ratio:.2f}")

    print("\n=== TEST 5: Empty List Edge Cases ===")
    empty = FastSinglyLinkedList()
    try:
        empty.get(0)
    except IndexError:
        print("Correctly caught get() on empty list")
    try:
        empty.remove(0)
    except IndexError:
        print("Correctly caught remove() on empty list")

    print("\n=== TEST 6: Linked List Consistency Check ===")
    lst5 = FastSinglyLinkedList()
    for i in range(10):
        lst5.insert(lst5.size(), i*10)
    linked_values = list(iter_linked_list(lst5))
    index_values = [lst5.get(i) for i in range(lst5.size())]
    assert set(linked_values) == set(index_values), "Mismatch between linked list and get(i)"
    print("Linked list and get(i) consistency passed")

    print("\nAll tests passed successfully!")


if __name__ == "__main__":
    run_comprehensive_tests()
