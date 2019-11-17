
# recursive programming
def fib1(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fib1(n - 1) + fib1(n - 2)

# dynamic programming
def fib2(n):
    fdict = {}
    fdict[0] = 0
    fdict[1] = 1
    if n in [0, 1]:
        return fdict[n]
    for x in range(2, n + 1):
        fdict[x] = fdict[x - 1] + fdict.pop(x - 2)
    return fdict[n]
