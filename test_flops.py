import time
from decimal import Decimal, getcontext
import numpy as np
import cupy as cp
from typing import Literal, List
from random import randint
from tabulate import tabulate
import matplotlib.pyplot as plt

Device = Literal['gpu', 'cpu']

getcontext().prec = 100  # Set Decimal precision to 100 decimal places

class FLOPS:
    @staticmethod
    def reference_inputs(size: int) -> List[Decimal]:
        print(f'-> Generating reference inputs of size {size}...')
        max_unique_values = 1000
        unique_size = min(size, max_unique_values)  # Determine number of unique values to generate
        inputs = []
        for _ in range(unique_size):
            number_str = '0.' + ''.join(str(randint(0, 9)) for _ in range(getcontext().prec - 1))
            inputs.append(Decimal(number_str))

        # If the desired size is greater than the number of unique values, repeat the array
        if size > unique_size:
            repeat_count = size // unique_size
            remainder = size % unique_size
            inputs = inputs * repeat_count + inputs[:remainder]


        assert(len(inputs) == size)
        print(f'-> Finished generating')

        return inputs

    @staticmethod
    def reference_outputs(inputs: List[Decimal]):
        results = {'mul': [x * x for x in inputs], 'add': [x + x for x in inputs], 'sub': [x - x for x in inputs]}
        return results

    @classmethod
    def measure_flops(cls, dtype, device: Device, input_size: int):
        inputs = cls.reference_inputs(input_size)
        num_flop, time_taken_ns = cls.compute_routines(inputs, dtype, device)
        flops_rate = num_flop / (time_taken_ns * 10**-9)  # Correctly convert ns to seconds

        print(f'Time taken ns = {time_taken_ns} ns')

        return flops_rate

    @staticmethod
    def compute_routines(inputs: List[Decimal], dtype, device: Device, iterations=1) -> (int, float):
        lib = cp if device == 'gpu' else np
        dtype = getattr(lib, dtype)
        print(f'-> Converting inputs to np array')
        array_inputs = lib.array([float(d) for d in inputs], dtype=dtype)
        print(f'-> Finished conversion')

        start_time = time.time_ns()
        for _ in range(iterations):
            _ = array_inputs * array_inputs
            _ = array_inputs + array_inputs
            _ = array_inputs - array_inputs
        end_time = time.time_ns()

        operations_count = 3 * array_inputs.size * iterations
        time_in_ns = end_time - start_time
        return operations_count, time_in_ns

    @classmethod
    def measure_varying_input(cls, dtype, device: Device):
        input_sizes = [10**1, 10**2, 10**3, 10**4, 10**5, 10**6,10**7, 10**8]
        results = []
        for size in input_sizes:
            print(f'Testing for input size: {size}...')
            flops_rate = cls.measure_flops(dtype, device, size)
            results.append([f'n={float(size):.2e}', f'{flops_rate:.2e} FLOPS'])

        headers = ['Input Size', 'FLOPS Rate']
        print(tabulate(results, headers=headers, tablefmt='psql'))


def plot_flops(input_sizes, flops_rates):
    plt.figure(figsize=(10, 5))  # Set the figure size (optional)
    plt.plot(input_sizes, flops_rates, marker='o', linestyle='-', color='b')  # Plot the data
    plt.xlabel('Input Size')  # Label for the x-axis
    plt.ylabel('FLOP/s')  # Label for the y-axis
    plt.title('FLOP/s vs Input Size')  # Title of the plot
    plt.xscale('log')  # Set the x-axis to a logarithmic scale
    plt.yscale('log')  # Set the y-axis to a logarithmic scale
    plt.grid(True)  # Enable grid
    plt.show()  # Display the plot



if __name__ == "__main__":
    dtype = 'float64'
    device: Device = 'gpu'
    FLOPS.measure_varying_input(dtype, device)

