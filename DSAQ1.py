class Node:
    def __init__(self, value: int):
        self.value = value
        self.next = None


class FastSinglyLinkedList:
    def __init__(self):
        self.head = None
        self.index_map = []  # list of Node references

    # O(1) get(i)
    def get(self, i: int) -> int:
        if i < 0 or i >= len(self.index_map):
            raise IndexError("Index out of bounds")
        return self.index_map[i].value

    #O(1) for insertion at any position, finding + pointer swap
    def insert(self, i: int, value: int) -> None:
        if i < 0 or i > len(self.index_map):
            raise IndexError("Index out of bounds")

        new_node = Node(value)

        #validation for position
        if i == 0:
            new_node.next = self.head
            self.head = new_node
        else:
            prev = self.index_map[i - 1]
            new_node.next = prev.next
            prev.next = new_node
        #keep the map in sync with the list
        self.index_map.insert(i, new_node)

    # O(1) removal at any position (amortized)
    def remove(self, i: int) -> None:
        if i < 0 or i >= len(self.index_map):
            raise IndexError("Index out of bounds")

        if i == 0:
            self.head = self.head.next
        else:
            prev = self.index_map[i - 1]
            prev.next = prev.next.next

        self.index_map.pop(i)

    def size(self) -> int:
        return len(self.index_map)


# ------------------ Test Code ------------------
if __name__ == "__main__":
    lst = FastSinglyLinkedList()

    print("--- Phase 1: Adding Elements ---")
    lst.insert(0, 10)
    lst.insert(1, 20)
    lst.insert(2, 30)

    print("Size should be 3:", lst.size())
    print("Index 1 should be 20:", lst.get(1))

    print("\n--- Phase 2: Middle Insertion ---")
    lst.insert(1, 15)

    for i in range(lst.size()):
        print(lst.get(i), end=" -> ")
    print("NULL")

    print("\n--- Phase 3: Removal ---")
    lst.remove(2)

    for i in range(lst.size()):
        print(lst.get(i), end=" -> ")
    print("NULL")
