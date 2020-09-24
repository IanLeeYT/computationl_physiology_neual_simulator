use_cache = True


class Cache:
    # Acts like a hash map. Should be used in cases where the same data may be recomputed.
    # key should be detailed enough to assert that that computed data will be the same as saved data
    def __init__(self):
        self.mem = {}

    def search(self, key):
        if use_cache:
            return key in self.mem
        else:
            return False

    def store(self, key, val):
        if use_cache:
            self.mem[key] = val

    def get(self, key):
        if use_cache:
            return self.mem[key]


cache = Cache()
