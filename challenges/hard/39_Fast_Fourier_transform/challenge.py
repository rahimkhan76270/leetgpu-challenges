import ctypes
from typing import Any, List, Dict
import torch
import os
from pathlib import Path


class Challenge:
    def __init__(self):
        self.name="Fast Fourier Transform"
        self.atol=1e-3         
        self.rtol=1e-3
        self.num_gpus=1
        self.access_tier="free"

    def reference_impl(self, signal: torch.Tensor, spectrum: torch.Tensor, N: int):
        """
        Ground-truth implementation using torch.fft. Assumes both tensors are
        on the same device (CPU or CUDA).  Works for any N (power-of-two not
        required, but contestants may optimise for radix-2).

        Args
        ----
        signal     : flattened real/imag interleaved input  (len == 2 × N)
        spectrum   : flattened real/imag interleaved output (len == 2 × N)
        N          : number of complex samples
        """
        assert signal.shape == (2 * N,)
        assert spectrum.shape == (2 * N,)
        assert signal.dtype == spectrum.dtype
        assert signal.device == spectrum.device

        # View as (N, 2) → complex tensor
        sig_ri = signal.view(N, 2)
        sig_c  = torch.complex(sig_ri[:, 0], sig_ri[:, 1])

        # Torch reference FFT
        spec_c = torch.fft.fft(sig_c)

        # Write back as interleaved real/imag
        spec_ri = torch.stack((spec_c.real, spec_c.imag), dim=1).contiguous()
        spectrum.copy_(spec_ri.view(-1))

    def get_solve_signature(self) -> Dict[str, Any]:
        return {
            "signal": ctypes.POINTER(ctypes.c_float),    # in  (2 × N)
            "spectrum": ctypes.POINTER(ctypes.c_float),  # out (2 × N)
            "N": ctypes.c_int
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 4
        # Impulse signal δ[n] = 1 when n=0 else 0 (expected flat spectrum)
        signal = torch.tensor([1.0, 0.0,   0.0, 0.0,
                               0.0, 0.0,   0.0, 0.0],
                              device="cuda", dtype=dtype)
        spectrum = torch.empty(2 * N, device="cuda", dtype=dtype)
        return {"signal": signal, "spectrum": spectrum, "N": N}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        cases: List[Dict[str, Any]] = []

        # 1. Constant signal  (all ones) – DC spike only
        N = 8
        const_sig = torch.ones(2 * N, device="cuda", dtype=dtype)
        const_spec = torch.empty_like(const_sig)
        cases.append({"signal": const_sig, "spectrum": const_spec, "N": N})

        # 2. Single-frequency sinusoid  (real: cos, imag: sin)
        N = 16
        k = 3  # frequency bin
        n = torch.arange(N, device="cuda", dtype=dtype)
        real = torch.cos(2.0 * torch.pi * k * n / N)
        imag = torch.sin(2.0 * torch.pi * k * n / N)
        sinusoid = torch.stack((real, imag), dim=1).contiguous().view(-1)
        sinusoid_spec = torch.empty_like(sinusoid)
        cases.append({"signal": sinusoid, "spectrum": sinusoid_spec, "N": N})

        # 3. Random complex signal, power-of-two length
        N = 256
        rnd = torch.empty(2 * N, device="cuda", dtype=dtype).uniform_(-1.0, 1.0)
        rnd_spec = torch.empty_like(rnd)
        cases.append({"signal": rnd, "spectrum": rnd_spec, "N": N})

        # 4. Random complex signal, non-power-of-two length
        N = 250
        rnd_np2 = torch.empty(2 * N, device="cuda", dtype=dtype).uniform_(-1.0, 1.0)
        rnd_np2_spec = torch.empty_like(rnd_np2)
        cases.append({"signal": rnd_np2, "spectrum": rnd_np2_spec, "N": N})

        # 5. Medium-size signal (performance sanity)
        N = 4096
        med = torch.empty(2 * N, device="cuda", dtype=dtype).normal_(0.0, 0.5)
        med_spec = torch.empty_like(med)
        cases.append({"signal": med, "spectrum": med_spec, "N": N})

        return cases

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 262_144          # 256 K complex samples  (~2 MiB real/imag)
        big_sig = torch.empty(2 * N, device="cuda", dtype=dtype).normal_(0.0, 1.0)
        big_spec = torch.empty_like(big_sig)
        return {"signal": big_sig, "spectrum": big_spec, "N": N}
    



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
        input_tensor = test_case["signal"]
        output_cuda = test_case["spectrum"].clone()
        output_torch = test_case["spectrum"].clone()
        # N = test_case["N"]

        
        print("\nTest case details:")
        print(f"Input shape: {input_tensor.shape}")

        # Run reference implementation
        challenge.reference_impl(
            input_tensor, output_torch,
            test_case["N"]
        )
        print(f"PyTorch result: {output_torch.cpu().numpy()}")

        # Convert CUDA tensor pointers to ctypes
        input_ptr = ctypes.cast(input_tensor.data_ptr(), ctypes.POINTER(ctypes.c_float))
        output_ptr = ctypes.cast(output_cuda.data_ptr(), ctypes.POINTER(ctypes.c_float))

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
            print("\nError details:")
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
    test_fft()



