import collections, random
import numpy as np

class ReplayBuffer:
    def __init__(self, size=int(1e6)):
        self.buf = collections.deque(maxlen=size)

    def add(self, s,a,r,s2,done):
        self.buf.append((s,a,r,s2,done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s,a,r,s2,d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r), np.array(s2), np.array(d))

    def __len__(self):
        return len(self.buf)
