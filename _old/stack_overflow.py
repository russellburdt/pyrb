
""" do not use a mutable object as a default value """
def func(x=[]):
    x.append(1)
    return x
print(func())
print(func())
print(func())

""" instead use a sentinel object, and initialize the mutable object accordingly """
def func(x=None):
    if x is None:
        x = []
    x.append(1)
    return x
print(func())
print(func())
print(func())

""" functions can access outer-scope objects iff the object name is not reused locally """
src = 'abc'
def func():
    print(src)
func()

def func():
    print(src)
    src = 'def'
func()
