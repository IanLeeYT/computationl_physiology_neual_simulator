use_cache = True


class Cache:
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
