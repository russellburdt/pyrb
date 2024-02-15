
"""
Python implementations and wrappers of Knuth algorithms
"""


def algorithm_u(ns, m):
    """
    - Python 3 implementation of
        Knuth in the Art of Computer Programming, Volume 4, Fascicle 3B, Algorithm U
    - copied from Python 2 implementation of the same at
        https://codereview.stackexchange.com/questions/1526/finding-all-k-subset-partitions
    - the algorithm returns
        all set partitions with a given number of blocks, as a python generator object

    e.g.
        In [1]: gen = algorithm_u(['A', 'B', 'C'], 2)

        In [2]: list(gen)
        Out[2]: [[['A', 'B'], ['C']],
                  [['A'], ['B', 'C']],
                  [['A', 'C'], ['B']]]
    """

    def visit(n, a):
        ps = [[] for i in range(m)]
        for j in range(n):
            ps[a[j + 1]].append(ns[j])
        return ps

    def f(mu, nu, sigma, n, a):
        if mu == 2:
            yield visit(n, a)
        else:
            for v in f(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v
        if nu == mu + 1:
            a[mu] = mu - 1
            yield visit(n, a)
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                yield visit(n, a)
        elif nu > mu + 1:
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = mu - 1
            else:
                a[mu] = mu - 1
            if (a[nu] + sigma) % 2 == 1:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v

    def b(mu, nu, sigma, n, a):
        if nu == mu + 1:
            while a[nu] < mu - 1:
                yield visit(n, a)
                a[nu] = a[nu] + 1
            yield visit(n, a)
            a[mu] = 0
        elif nu > mu + 1:
            if (a[nu] + sigma) % 2 == 1:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] < mu - 1:
                a[nu] = a[nu] + 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = 0
            else:
                a[mu] = 0
        if mu == 2:
            yield visit(n, a)
        else:
            for v in b(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v

    n = len(ns)
    a = [0] * (n + 1)
    for j in range(1, m + 1):
        a[n - m + j] = j - 1
    return f(m, n, 0, n, a)

class algorithm_u_permutations:
    """
    generator for all permutations of all set partitions with a given number of blocks
    e.g.
        In [4]: gen = algorithm_u_permutations(['A', 'B', 'C'], 2)

        In [5]: list(gen)
        Out[5]:
        [(['A', 'B'], ['C']),
         (['C'], ['A', 'B']),
         (['A'], ['B', 'C']),
         (['B', 'C'], ['A']),
         (['A', 'C'], ['B']),
         (['B'], ['A', 'C'])]
    """

    from itertools import permutations

    def __init__(self, ns, m):

        self.au = algorithm_u(ns, m)
        self.perms = self.permutations(next(self.au))

    def __next__(self):

        try:
            return next(self.perms)

        except StopIteration:
            self.perms = self.permutations(next(self.au))
            return next(self.perms)

    def __iter__(self):
        return self
