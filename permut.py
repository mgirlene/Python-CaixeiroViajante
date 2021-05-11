from itertools import permutations
obj = range(1, 4)

permuts = permutations(obj)

for permutation in permuts:
    print(permutation)