# test.py

import random

def write_test_input():
    with open('input.txt', 'w') as f:
        mask_size = 1000
        data_size = 1_000_000

        # Generate random mask and data values (e.g., between -10 and 10)
        mask = [random.randint(-10, 10) for _ in range(mask_size)]
        data = [random.randint(-100, 100) for _ in range(data_size)]

        f.write(f"{mask_size} {data_size}\n")
        f.write(' '.join(map(str, mask)) + '\n')
        f.write(' '.join(map(str, data)) + '\n')

if __name__ == "__main__":
    write_test_input()
