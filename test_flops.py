import time
from decimal import Decimal, getcontext
import numpy as np
import cupy as cp
from typing import Literal, List
from random import randint
from tabulate import tabulate

Device = Literal['gpu', 'cpu']

getcontext().prec = 100  # Set Decimal precision to 100 decimal places

class FLOPS:
    @staticmethod
    def reference_inputs(size: int) -> List[Decimal]:
        """ Generates reference inputs with randomly generated high-precision Decimal values. """
        inputs = []
        for _ in range(size):
            number_str = '0.' + ''.join(str(randint(0, 9)) for _ in range(getcontext().prec - 1))
            inputs.append(Decimal(number_str))
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

        return flops_rate

    @staticmethod
    def compute_routines(inputs: List[Decimal], dtype, device: Device, iterations=10**3) -> (int, float):
        lib = cp if device == 'gpu' else np
        dtype = getattr(lib, dtype)
        array_inputs = lib.array([float(d) for d in inputs], dtype=dtype)

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
        input_sizes = [10**2, 10**3, 10**4, 10**5]
        results = []
        for size in input_sizes:
            print(f'Testing for input size: {size}...')
            flops_rate = cls.measure_flops(dtype, device, size)
            results.append([f'{float(size):.2e}\u2800', f'{flops_rate:.2e} FLOPS'])

        headers = ['Input Size', 'FLOPS Rate']
        print(tabulate(results, headers=headers, tablefmt='psql'))

if __name__ == "__main__":
    dtype = 'float64'
    device: Device = 'gpu'
    FLOPS.measure_varying_input(dtype, device)
