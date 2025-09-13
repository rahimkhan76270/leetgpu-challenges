import ctypes
from typing import Any, List, Dict
import torch
import os
from pathlib import Path

class Challenge:
    def __init__(self):
        self.name="Swish-Gated Linear Unit"
        self.atol=1e-05
        self.rtol=1e-05
        self.num_gpus=1
        self.access_tier="free"
        
    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, N: int):
        assert N % 2 == 0
        assert input.shape ==  (N,)
        assert output.shape ==  (N//2,)
        assert input.dtype == output.dtype
        assert input.device == output.device

        x1, x2 = input.chunk(2)
        output.copy_((x1 * torch.sigmoid(x1)) * x2)
        
    def get_solve_signature(self) -> Dict[str, Any]:
        return {
            "input": ctypes.POINTER(ctypes.c_float),
            "output": ctypes.POINTER(ctypes.c_float),
            "N": ctypes.c_int,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 4
        input = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=dtype)
        output = torch.empty(N // 2, device="cuda", dtype=dtype)
        return {
            "input": input,
            "output": output,
            "N": N,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # basic_small
        N = 2
        tests.append({
            "input": torch.tensor([0.5, 1.0], device="cuda", dtype=dtype),
            "output": torch.empty(N // 2, device="cuda", dtype=dtype),
            "N": N
        })

        # all zeros
        N = 42
        tests.append({
            "input": torch.zeros(N, device="cuda", dtype=dtype),
            "output": torch.empty(N // 2, device="cuda", dtype=dtype),
            "N": N
        })

        # negative numbers
        N = 6
        tests.append({
            "input": torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0], device="cuda", dtype=dtype),
            "output": torch.empty(N // 2, device="cuda", dtype=dtype),
            "N": N
        })

        # mixed positive/negative
        N = 4
        tests.append({
            "input": torch.tensor([-0.5, 0.0, - 1.5, 1.0], device="cuda", dtype=dtype),
            "output": torch.empty(N // 2, device="cuda", dtype=dtype),
            "N": N
        })

        # large values
        N = 1024
        tests.append({
            "input": torch.empty(N, device="cuda", dtype=dtype).uniform_(-100.0, 100.0),
            "output": torch.empty(N // 2, device="cuda", dtype=dtype),
            "N": N
        })

        # large N
        N = 2048
        tests.append({
            "input": torch.empty(N, device="cuda", dtype=dtype).uniform_(-50.0, 50.0),
            "output": torch.empty(N // 2, device="cuda", dtype=dtype),
            "N": N
        })

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 100000
        return {
            "input": torch.empty(N, device="cuda", dtype=dtype).uniform_(-100.0, 100.0),
            "output": torch.empty(N // 2, device="cuda", dtype=dtype),
            "N": N
        }
    

def load_cuda_lib():
    # Compile the CUDA code first
    cuda_file = Path("starter/starter.cu")
    lib_file = Path("starter/libswiglu.so")
    # Compile the CUDA code
    os.system(f"nvcc -o {lib_file} --shared -Xcompiler -fPIC {cuda_file}")
    
    # Load the compiled library
    cuda_lib = ctypes.CDLL(str(lib_file))
    return cuda_lib

def test_swiglu():
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
    ]

    def run_test(test_case):
        input_tensor = test_case["input"]
        output_cuda = test_case["output"].clone()
        output_torch = test_case["output"].clone()
        
        print(f"\nTest case details:")
        print(f"Input shape: {input_tensor.shape}")

        # Run reference implementation
        challenge.reference_impl(
            input_tensor, output_torch,
            test_case["N"]
        )
        print(f"PyTorch result: {output_torch.cpu().numpy()}")

        # Convert CUDA tensor pointers to ctypes
        input_ptr = ctypes.cast(input_tensor.data_ptr(), ctypes.POINTER(ctypes.c_int))
        output_ptr = ctypes.cast(output_cuda.data_ptr(), ctypes.POINTER(ctypes.c_int))

        # Run CUDA implementation
        solve_func(
            input_ptr, output_ptr,
            test_case["N"]
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
    test_swiglu()