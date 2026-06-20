# src/cache/structures.py
"""
Core data structures from LeCaR codebase.
DequeDict : O(1) ordered dict (LRU ordering)
HeapDict  : O(log n) min-heap + hashmap (LFU ordering)
"""


class DequeDict:
    """
    Doubly-linked list + hashmap.
    Head = LRU end (oldest). Tail = MRU end (newest).
    O(1) insert, delete, lookup.
    """

    class Node:
        __slots__ = ['key', 'value', 'prev', 'next']
        def __init__(self, key, value):
            self.key   = key
            self.value = value
            self.prev  = None
            self.next  = None

    def __init__(self):
        self.htbl = {}
        self.head = None
        self.tail = None

    def __contains__(self, key): return key in self.htbl
    def __len__(self):           return len(self.htbl)
    def __getitem__(self, key):  return self.htbl[key].value

    def __setitem__(self, key, value):
        if key in self.htbl:
            self._remove(key)
        self._push(key, value)

    def __delitem__(self, key):
        self._remove(key)

    def first(self):
        """Return LRU item (head) without removing."""
        return self.head.value

    def popFirst(self):
        """Remove and return LRU item (head)."""
        node = self.head
        self._remove(node.key)
        return node.value

    def _push(self, key, value):
        """Push to tail (MRU end)."""
        node = self.Node(key, value)
        self.htbl[key] = node
        if self.tail:
            self.tail.next = node
            node.prev      = self.tail
        else:
            self.head = node
        self.tail = node

    def _remove(self, key):
        node = self.htbl.pop(key)
        if node.prev: node.prev.next = node.next
        else:         self.head      = node.next
        if node.next: node.next.prev = node.prev
        else:         self.tail      = node.prev


class HeapDict:
    """
    Min-heap + hashmap.
    O(log n) insert/remove. O(1) min peek.
    """

    class Node:
        __slots__ = ['key', 'value', 'index']
        def __init__(self, key, value):
            self.key   = key
            self.value = value
            self.index = -1
        def __lt__(self, other):
            return self.value < other.value

    def __init__(self):
        self.htbl = {}
        self.heap = []

    def __contains__(self, key): return key in self.htbl
    def __len__(self):           return len(self.heap)
    def __getitem__(self, key):
        return self.htbl[key].value

    def min(self):
        """Return min item without removing. None if empty."""
        return self.heap[0].value if self.heap else None

    def popMin(self):
        """Remove and return min item."""
        node = self.heap[0]
        del self[node.key]
        return node.value

    def __setitem__(self, key, value):
        if key in self.htbl:
            node       = self.htbl[key]
            node.value = value
            self._up(node.index)
            self._down(node.index)
        else:
            node       = self.Node(key, value)
            self.htbl[key] = node
            node.index = len(self.heap)
            self.heap.append(node)
            self._up(node.index)

    def __delitem__(self, key):
        node = self.htbl.pop(key)
        last = self.heap[-1]
        self._swap(node, last)
        self.heap.pop()
        if node is not last:
            self._up(last.index)
            self._down(last.index)

    def _swap(self, a, b):
        a.index, b.index              = b.index, a.index
        self.heap[a.index]            = a
        self.heap[b.index]            = b

    def _up(self, i):
        while i > 0:
            p = (i - 1) // 2
            if self.heap[i] < self.heap[p]:
                self._swap(self.heap[i], self.heap[p])
                i = p
            else:
                break

    def _down(self, i):
        n = len(self.heap)
        while True:
            s = i
            l, r = 2*i+1, 2*i+2
            if l < n and self.heap[l] < self.heap[s]: s = l
            if r < n and self.heap[r] < self.heap[s]: s = r
            if s == i: break
            self._swap(self.heap[i], self.heap[s])
            i = s