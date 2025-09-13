import ctypes
from typing import Any, List, Dict
import torch
import os
from pathlib import Path

class Challenge:
    def __init__(self):
        
        self.name="Max Subarray Sum"
        self.atol=1e-05
        self.rtol=1e-05
        self.num_gpus=1
        self.access_tier="free"

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int, window_size: int):
        # Validate input types and shapes
        assert input.shape == (N,)
        assert output.shape == (1,)
        assert input.dtype == torch.int32
        assert output.dtype == torch.int32

        # Computes the maximum sum of any contiguous subarray of length exactly window_size
        # using a sliding window approach.

        # Compute the sum of the first window_size elements (the initial window)
        current_sum = input[:window_size].sum()

        # Initialize max_sum with the sum of the first window
        max_sum = current_sum

        # Slide the window across the array from index window_size to N - 1
        for i in range(window_size, N):
            # Update the current sum by subtracting the element leaving the window
            # and adding the new element entering the window
            current_sum += input[i] - input[i - window_size]
    
            # Update max_sum if the current sum is greater
            max_sum = torch.max(max_sum, current_sum)

        # Store the final result in the output tensor
        output[0] = max_sum

    def get_solve_signature(self) -> Dict[str, Any]:
        return {
            "input": ctypes.POINTER(ctypes.c_int),
            "output": ctypes.POINTER(ctypes.c_int),
            "N": ctypes.c_int,
            "window_size": ctypes.c_int
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.int32
        input = torch.tensor([1, 2, 4, 2, 3], device="cuda", dtype=dtype)
        output = torch.empty(1, device="cuda", dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": 5,
            "window_size": 2
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.int32
        tests = []

        # basic_example
        tests.append({
            "input": torch.tensor([-1, -4, -2, 1], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 4,
            "window_size": 3
        })

        # all_same_value
        tests.append({
            "input": torch.tensor([2]*16, device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 16,
            "window_size": 15
        })

        # all_minus_value
        tests.append({
            "input": torch.tensor([-10]*1000, device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 1000,
            "window_size": 500
        })

        # increasing_sequence
        tests.append({
            "input": torch.randint(-10, 11, (123,), device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 123,
            "window_size": 7
        })

        # medium_size
        tests.append({
            "input": torch.randint(-10, 11, (1000,), device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 1000,
            "window_size": 476
        })

        # large_size
        tests.append({
            "input": torch.randint(-10, 11, (10000,), device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 10000,
            "window_size": 7011
        })

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.int32
        input = torch.randint(-10, 11, (50000,), device="cuda", dtype=dtype)
        output = torch.empty(1, device="cuda", dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": 50000,
            "window_size": 25000
        }
    


def load_cuda_lib():
    # Compile the CUDA code first
    cuda_file = Path("starter/starter.cu")
    lib_file = Path("starter/libmax_subarray_sum.so")
    
    # Compile the CUDA code
    os.system(f"nvcc -o {lib_file} --shared -Xcompiler -fPIC {cuda_file}")
    
    # Load the compiled library
    cuda_lib = ctypes.CDLL(str(lib_file))
    return cuda_lib

def test_max_subarray_sum():
    # Initialize the challenge
    challenge = Challenge()
    
    # Load the CUDA library
    cuda_lib = load_cuda_lib()
    
    # Get the solve function and set its argument types
    solve_func = cuda_lib.solve
    solve_func.argtypes = [
        ctypes.POINTER(ctypes.c_int),    # input
        ctypes.POINTER(ctypes.c_int),    # output
        ctypes.c_int,                    # N
        ctypes.c_int                     # window_size
    ]

    def run_test(test_case):
        input_tensor = test_case["input"]
        output_cuda = test_case["output"].clone()
        output_torch = test_case["output"].clone()
        N = test_case["N"]
        window_size = test_case["window_size"]

        print(f"\nTest case details:")
        print(f"Input shape: {input_tensor.shape}")
        print(f"N: {N}, Window size: {window_size}")
        print(f"First few input values: {input_tensor[:5].cpu().numpy()}")

        # Run reference implementation
        challenge.reference_impl(input_tensor, output_torch, N, window_size)
        print(f"PyTorch result: {output_torch.cpu().numpy()}")

        # Convert CUDA tensor pointers to ctypes
        input_ptr = ctypes.cast(input_tensor.data_ptr(), ctypes.POINTER(ctypes.c_int))
        output_ptr = ctypes.cast(output_cuda.data_ptr(), ctypes.POINTER(ctypes.c_int))

        # Run CUDA implementation
        solve_func(
            input_ptr,
            output_ptr,
            N,
            window_size
        )
        print(f"CUDA result: {output_cuda.cpu().numpy()}")

        try:
            torch.testing.assert_close(
                output_cuda, 
                output_torch, 
                rtol=challenge.rtol, 
                atol=challenge.atol,
                msg="CUDA and PyTorch results don't match"
            )
            return True
        except AssertionError as e:
            print(f"\nError details:")
            print(f"Expected (PyTorch): {output_torch.cpu().numpy()}")
            print(f"Got (CUDA): {output_cuda.cpu().numpy()}")
            print(f"Absolute difference: {torch.abs(output_cuda - output_torch).cpu().numpy()}")
            raise e

    # Run example test
    print("Running example test...")
    example_test = challenge.generate_example_test()
    run_test(example_test)
    print("Example test passed!")

    # Run functional tests
    print("\nRunning functional tests...")
    for i, test_case in enumerate(challenge.generate_functional_test()):
        try:
            run_test(test_case)
            print(f"Functional test {i+1} passed!")
        except AssertionError as e:
            print(f"Functional test {i+1} failed:", e)

    # Run performance test
    print("\nRunning performance test...")
    perf_test = challenge.generate_performance_test()
    try:
        run_test(perf_test)
        print("Performance test passed!")
    except AssertionError as e:
        print("Performance test failed:", e)

if __name__ == "__main__":
    test_max_subarray_sum()