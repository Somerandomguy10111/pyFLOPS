import time
from decimal import Decimal, getcontext

import numpy as np

getcontext().prec = 100
import cupy as cp
from typing import Literal

Device = Literal['gpu','cpu']

#-------------------------------------------

class FLOPS:
    @staticmethod
    def reference_inputs():
        return [
            Decimal('1.123456789123456789123456789'),
            Decimal('2.987654321987654321987654321'),
            Decimal('3.111111111111111111111111111'),
            Decimal('0.333333333333333333333333333')
        ]

    @staticmethod
    def reference_outputs():
        inputs = FLOPS.reference_inputs()
        results = {'mul': [x * x for x in inputs], 'add': [x + x for x in inputs], 'sub': [x - x for x in inputs]}
        return results

    @classmethod
    def measure_flops(cls, dtype, device : Device):
        start_time = time.time()
        num_flop = cls.compute_routines(dtype, device)
        end_time = time.time()
        flops_rate = num_flop / (end_time - start_time)
        print(f'FLOPS rate on {device} using dtype {dtype}: {flops_rate} FLOPS')

        return flops_rate

    @staticmethod
    def compute_routines(dtype, device : Device, iterations=10000) -> int:
        lib = cp if device == 'gpu' else np
        dtype = getattr(lib, dtype)
        inputs = lib.array([float(d) for d in FLOPS.reference_inputs()], dtype=dtype)

        for _ in range(iterations):
            a = inputs * inputs
            b = inputs + inputs
            c = inputs - inputs

        operations_count = 3 * inputs.size * iterations

        return operations_count


if __name__ == "__main__":
    dtype = 'float64'
    device : Device = 'gpu'
    flops = FLOPS.measure_flops(dtype=dtype,device=device)
    ref_results = FLOPS.reference_outputs()
