def amdahls_law(p, s):
    """Speedup relative to proportion parallel
    Amdahl's Law gives an idealized speedup we
    can expect for an algorithm given the proportion
    that algorithm can be parallelized and the speed
    we gain from that parallelization. The best case
    scenario is that the speedup, `s`, is equal to
    the number of processors available.
    Args:
        p: proportion parallel
        s: speed up for the parallelized proportion
    """
    return 1 / ((1-p) + p/s)


print("\n Amdahl for p=0.3")
print(amdahls_law(0.3, 1))
print(amdahls_law(0.3, 2))
print(amdahls_law(0.3, 3))
print(amdahls_law(0.3, 4))

print("\n Amdahl for p=0.5")
print(amdahls_law(0.5, 1))
print(amdahls_law(0.5, 2))
print(amdahls_law(0.5, 3))
print(amdahls_law(0.5, 4))

print("\n Amdahl for p=0.7")
print(amdahls_law(0.7, 1))
print(amdahls_law(0.7, 2))
print(amdahls_law(0.7, 3))
print(amdahls_law(0.7, 4))

print("\n Amdahl for p=0.9")
print(amdahls_law(0.9, 1))
print(amdahls_law(0.9, 2))
print(amdahls_law(0.9, 3))
print(amdahls_law(0.9, 4))
