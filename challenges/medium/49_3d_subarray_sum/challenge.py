import ctypes
from typing import Any, List, Dict
import torch
import os
from pathlib import Path

class Challenge:
    def __init__(self):
        self.name="3D Subarray Sum"
        self.atol=1e-05
        self.rtol=1e-05
        self.num_gpus=1
        self.access_tier="free"

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int, M: int, K: int, S_DEP: int, E_DEP: int, S_ROW: int, E_ROW: int, S_COL: int, E_COL: int):
        # Validate input types and shapes
        assert input.shape == (N, M, K)
        assert output.shape == (1,)
        assert input.dtype == torch.int32
        assert output.dtype == torch.int32

        # add all elements of subarray (input[S_DEP..E_DEP][S_ROW..E_ROW][S_COL..E_COL])
        output[0] = torch.sum(input[S_DEP:E_DEP+1, S_ROW:E_ROW+1, S_COL:E_COL+1])

    def get_solve_signature(self) -> Dict[str, Any]:
        return {
            "input": ctypes.POINTER(ctypes.c_int),
            "output": ctypes.POINTER(ctypes.c_int),
            "N": ctypes.c_int,
            "M": ctypes.c_int,
            "K": ctypes.c_int,
            "S_DEP": ctypes.c_int,
            "E_DEP": ctypes.c_int,
            "S_ROW": ctypes.c_int,
            "E_ROW": ctypes.c_int,
            "S_COL": ctypes.c_int,
            "E_COL": ctypes.c_int
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.int32
        input = torch.tensor([[[1, 2, 3], [4, 5, 1]], [[1, 1, 1], [2, 2, 2]]], device="cuda", dtype=dtype)
        output = torch.empty(1, device="cuda", dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": 2,
            "M": 2,
            "K": 3,
            "S_DEP": 0,
            "E_DEP": 1,
            "S_ROW": 0,
            "E_ROW": 0,
            "S_COL": 1,
            "E_COL": 2
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.int32
        tests = []

        # basic_example
        tests.append({
            "input": torch.tensor([[[5, 10], [5, 2], [2, 2]]], device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 1,
            "M": 3,
            "K": 2,
            "S_DEP": 0,
            "E_DEP": 0,
            "S_ROW": 0,
            "E_ROW": 2,
            "S_COL": 1,
            "E_COL": 1
        })

        # all_same_value
        tests.append({
            "input": torch.tensor([[[2]*16] * 20] * 30, device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 30,
            "M": 20,
            "K": 16,
            "S_DEP": 0,
            "E_DEP": 29,
            "S_ROW": 0,
            "E_ROW": 19,
            "S_COL": 0,
            "E_COL": 15
        })

        # increasing_sequence
        tests.append({
            "input": torch.randint(1, 11, (50,50,50), device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 50,
            "M": 50,
            "K": 50,
            "S_DEP": 0,
            "E_DEP": 49,
            "S_ROW": 0,
            "E_ROW": 49,
            "S_COL": 0,
            "E_COL": 49
        })

        # medium_size
        tests.append({
            "input": torch.randint(1, 11, (77,87,57), device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 77,
            "M": 87,
            "K": 57,
            "S_DEP": 0,
            "E_DEP": 76,
            "S_ROW": 0,
            "E_ROW": 37,
            "S_COL": 1,
            "E_COL": 50
        })

        # large_size
        tests.append({
            "input": torch.randint(1, 11, (100,100,100), device="cuda", dtype=dtype),
            "output": torch.empty(1, device="cuda", dtype=dtype),
            "N": 100,
            "M": 100,
            "K": 100,
            "S_DEP": 10,
            "E_DEP": 91,
            "S_ROW": 77,
            "E_ROW": 91,
            "S_COL": 12,
            "E_COL": 81
        })

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.int32
        input = torch.randint(1, 11, (500,500,500), device="cuda", dtype=dtype)
        output = torch.empty(1, device="cuda", dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": 500,
            "M": 500,
            "K": 500,
            "S_DEP": 11,
            "E_DEP": 498,
            "S_ROW": 0,
            "E_ROW": 499,
            "S_COL": 1,
            "E_COL": 489
        }
    

def load_cuda_lib():
    # Compile the CUDA code first
    cuda_file = Path("starter/starter.cu")
    lib_file = Path("starter/lib3dsubarray_sum.so")
    
    # Compile the CUDA code
    os.system(f"nvcc -o {lib_file} --shared -Xcompiler -fPIC {cuda_file}")
    
    # Load the compiled library
    cuda_lib = ctypes.CDLL(str(lib_file))
    return cuda_lib

def test_3d_subarray_sum():
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
        ctypes.c_int,                    # M
        ctypes.c_int,                    # K
        ctypes.c_int,                    # S_DEP
        ctypes.c_int,                    # E_DEP
        ctypes.c_int,                    # S_ROW
        ctypes.c_int,                    # E_ROW
        ctypes.c_int,                    # S_COL
        ctypes.c_int                     # E_COL
    ]

    def run_test(test_case):
        input_tensor = test_case["input"]
        output_cuda = test_case["output"].clone()
        output_torch = test_case["output"].clone()
        
        print(f"\nTest case details:")
        print(f"Input shape: {input_tensor.shape}")
        print(f"Subarray bounds: [{test_case['S_DEP']}:{test_case['E_DEP']+1}, "
              f"{test_case['S_ROW']}:{test_case['E_ROW']+1}, "
              f"{test_case['S_COL']}:{test_case['E_COL']+1}]")

        # Run reference implementation
        challenge.reference_impl(
            input_tensor, output_torch,
            test_case["N"], test_case["M"], test_case["K"],
            test_case["S_DEP"], test_case["E_DEP"],
            test_case["S_ROW"], test_case["E_ROW"],
            test_case["S_COL"], test_case["E_COL"]
        )
        print(f"PyTorch result: {output_torch.cpu().numpy()}")

        # Convert CUDA tensor pointers to ctypes
        input_ptr = ctypes.cast(input_tensor.data_ptr(), ctypes.POINTER(ctypes.c_int))
        output_ptr = ctypes.cast(output_cuda.data_ptr(), ctypes.POINTER(ctypes.c_int))

        # Run CUDA implementation
        solve_func(
            input_ptr, output_ptr,
            test_case["N"], test_case["M"], test_case["K"],
            test_case["S_DEP"], test_case["E_DEP"],
            test_case["S_ROW"], test_case["E_ROW"],
            test_case["S_COL"], test_case["E_COL"]
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
    test_3d_subarray_sum()