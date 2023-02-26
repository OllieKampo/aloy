from build.example import add, arg_first_where, Mult, simulate_control

from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np

print(add(1, 2))
m = Mult(2)
print(m.multiply(2))
print(m.multiply_vector([1, 2, 3]))
print(m.multiply_tuple((1, 2, 3)))

a = np.array([1, 2, 3])
# print(first_where(a, lambda x: x > 1))
print(arg_first_where(a, lambda x: x > 1))

start_time = time.time()

with ThreadPoolExecutor(max_workers=4) as executor:
    iter_ = executor.map(lambda x: m.function_concurrent_callable(), [None]*4)
    for r in iter_:
        print(r)

print("--- %s seconds ---" % (time.time() - start_time))