import ctypes
import os
import torch
from pathlib import Path
from challenge import Challenge

def load_cuda_lib():
    # Compile the CUDA code first
    cuda_file = Path("starter/starter.cu")
    lib_file = Path("starter/libfft.so")
    
    # Compile the CUDA code
    os.system(f"nvcc -o {lib_file} --shared -Xcompiler -fPIC {cuda_file}")
    
    # Load the compiled library
    cuda_lib = ctypes.CDLL(str(lib_file))
    return cuda_lib

def test_fft():
    # Initialize the challenge
    challenge = Challenge()
    
    # Load the CUDA library
    cuda_lib = load_cuda_lib()
    
    # Get the solve function and set its argument types
    solve_func = cuda_lib.solve
    solve_func.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # signal
        ctypes.POINTER(ctypes.c_float),  # spectrum
        ctypes.c_int                     # N
    ]

    def run_test(test_case):
        signal = test_case["signal"]
        spectrum_cuda = test_case["spectrum"].clone()  # Clone to keep original tensor unchanged
        spectrum_torch = test_case["spectrum"].clone()
        N = test_case["N"]

        # Run reference implementation
        challenge.reference_impl(signal, spectrum_torch, N)

        # Convert CUDA tensor pointers to ctypes
        signal_ptr = ctypes.cast(signal.data_ptr(), ctypes.POINTER(ctypes.c_float))
        spectrum_ptr = ctypes.cast(spectrum_cuda.data_ptr(), ctypes.POINTER(ctypes.c_float))

        # Run CUDA implementation
        solve_func(
            signal_ptr,
            spectrum_ptr,
            N
        )

        # Compare results
        torch.testing.assert_close(
            spectrum_cuda, 
            spectrum_torch, 
            rtol=challenge.rtol, 
            atol=challenge.atol,
            msg="CUDA and PyTorch FFT results don't match"
        )
        return True

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
    test_fft()