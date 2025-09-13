import ctypes
from typing import Any, Dict, List
import torch
import os
from pathlib import Path


class Challenge:
    def __init__(self):
        self.name = "Blockwise Online Softmax"
        self.atol = 1e-5
        self.rtol = 1e-5
        self.num_gpus = 1
        self.access_tier = "free"

    def reference_impl(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor):
        """
        Ground-truth implementation using torch.nn.functional.softmax. 
        Assumes both tensors are on the same device.
        """
        assert input_tensor.shape == output_tensor.shape
        assert input_tensor.device == output_tensor.device

        # Row-wise softmax
        output_tensor.copy_(torch.softmax(input_tensor, dim=-1))

    def get_solve_signature(self) -> Dict[str, Any]:
        return {
            "input_ptr": ctypes.POINTER(ctypes.c_float),
            "output_ptr": ctypes.POINTER(ctypes.c_float),
            "rows": ctypes.c_int,
            "cols": ctypes.c_int,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        rows, cols = 2, 4
        input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                                     [2.0, 1.0, 0.0, -1.0]],
                                    device="cuda", dtype=dtype)
        output_tensor = torch.empty_like(input_tensor)
        return {"input": input_tensor, "output": output_tensor, "rows": rows, "cols": cols}

    def generate_functional_tests(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        cases: List[Dict[str, Any]] = []

        # Small random matrix
        rows, cols = 4, 8
        input_tensor = torch.randn((rows, cols), device="cuda", dtype=dtype)
        output_tensor = torch.empty_like(input_tensor)
        cases.append({"input": input_tensor, "output": output_tensor, "rows": rows, "cols": cols})

        # Larger random matrix
        rows, cols = 32, 64
        input_tensor = torch.randn((rows, cols), device="cuda", dtype=dtype)
        output_tensor = torch.empty_like(input_tensor)
        cases.append({"input": input_tensor, "output": output_tensor, "rows": rows, "cols": cols})

        return cases


def load_cuda_lib():
    cuda_file = Path("starter/starter.cu")
    lib_file = Path("starter/libsoftmax.so")

    # Compile CUDA code into a shared library
    os.system(f"nvcc -o {lib_file} --shared -Xcompiler -fPIC {cuda_file}")

    # Load the compiled library
    cuda_lib = ctypes.CDLL(str(lib_file))
    return cuda_lib


def test_softmax():
    challenge = Challenge()
    cuda_lib = load_cuda_lib()

    # Get kernel function
    solve_func = cuda_lib.solve
    solve_func.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
    ]

    def run_test(test_case):
        input_tensor = test_case["input"]
        output_cuda = test_case["output"].clone()
        output_torch = test_case["output"].clone()

        # Run reference implementation
        challenge.reference_impl(input_tensor, output_torch)

        # Convert pointers for CUDA call
        input_ptr = ctypes.cast(input_tensor.data_ptr(), ctypes.POINTER(ctypes.c_float))
        output_ptr = ctypes.cast(output_cuda.data_ptr(), ctypes.POINTER(ctypes.c_float))

        # Launch CUDA kernel
        solve_func(input_ptr, output_ptr, test_case["rows"], test_case["cols"])

        # Compare results
        try:
            torch.testing.assert_close(
                output_cuda, output_torch,
                rtol=challenge.rtol,
                atol=challenge.atol,
                msg="CUDA and PyTorch results mismatch"
            )
            return True
        except AssertionError as e:
            print("\n❌ Mismatch detected!")
            print("Input:\n", input_tensor.cpu().numpy())
            print("Expected:\n", output_torch.cpu().numpy())
            print("Got:\n", output_cuda.cpu().numpy())
            raise e

    # Example test
    print("Running example test...")
    example_test = challenge.generate_example_test()
    run_test(example_test)
    print("Example test passed ✅")

    # Functional tests
    print("\nRunning functional tests...")
    for i, test_case in enumerate(challenge.generate_functional_tests()):
        run_test(test_case)
        print(f"Functional test {i+1} passed ✅")


if __name__ == "__main__":
    test_softmax()
