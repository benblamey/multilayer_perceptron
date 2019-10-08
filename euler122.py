import math

def num_operations(m, intermediates=None):
    if intermediates is None:
        intermediates = {1}

    biggest_power_of_2 = round(math.log(m))
    print(f'biggest_power_of_2={biggest_power_of_2}')

    new_existing = {2**i for i in range(biggest_power_of_2 + 1)}
    print(f'existing={intermediates}')

    intermediates = new_existing.union(intermediates)

    remainder = m - 2 ** biggest_power_of_2

    print(f'remainder is {remainder}')

    if remainder == 0:
        return intermediates
    else:
        remainder_intermediates = num_operations(remainder, intermediates)
        return remainder_intermediates

if __name__ == '__main__':
    print(num_operations(15))

