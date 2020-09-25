use_cache = True


class Cache:
    # Acts like a hash map. Should be used in cases where the same data may be recomputed.
    # key should be detailed enough to assert that that computed data will be the same as saved data
    def __init__(self):
        self.mem = {}
        self.size = 0

    def search(self, key):
        if use_cache:
            t, key2 = key
            if t in self.mem:
                return key2 in self.mem[t]
            else:
                return False
        else:
            return False

    def store(self, key, val):
        if use_cache:
            t, key2 = key
            if t in self.mem:
                self.mem[t][key2] = val
            else:
                self.mem[t] = {}
                self.mem[t][key2] = val
                self.size += 1
            if t-1 in self.mem:
                self.mem.pop(t-1)

    def get(self, key):
        if use_cache:
            t, key2 = key
            if t-1 in self.mem:
                self.mem.pop(t-1)
            return self.mem[t][key2]


cache = Cache()
