# helloworld.pyx
print("hello cython!")

cpdef int fib(int n) :
    if n == 0 or n == 1 :
        return 1
    else :
        return fib(n - 2) + fib(n - 1)

def aaa(n):
    return n+1

