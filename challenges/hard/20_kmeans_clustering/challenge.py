import ctypes
from typing import Any, List, Dict
import torch
from pathlib import Path
import os

class Challenge:
    def __init__(self):
        self.name="K-Means Clustering"
        self.atol=1e-04
        self.rtol=1e-04
        self.num_gpus=1
        self.access_tier="free"
        
    def reference_impl(self, data_x: torch.Tensor, data_y: torch.Tensor, labels: torch.Tensor, initial_centroid_x: torch.Tensor, initial_centroid_y: torch.Tensor, final_centroid_x: torch.Tensor, final_centroid_y: torch.Tensor, sample_size: int, k: int, max_iterations: int):
        assert data_x.shape == (sample_size,)
        assert data_y.shape == (sample_size,)
        assert initial_centroid_x.shape == (k,)
        assert initial_centroid_y.shape == (k,)
        assert final_centroid_x.shape == (k,)
        assert final_centroid_y.shape == (k,)
        assert labels.shape == (sample_size,)
        final_centroid_x.copy_(initial_centroid_x)
        final_centroid_y.copy_(initial_centroid_y)
        for _ in range(max_iterations):
            expanded_x = data_x.view(-1, 1) - final_centroid_x.view(1, -1)
            expanded_y = data_y.view(-1, 1) - final_centroid_y.view(1, -1)
            distances = expanded_x ** 2 + expanded_y ** 2
            labels.copy_(torch.argmin(distances, dim=1))
            for i in range(k):
                mask = (labels == i)
                if mask.any():
                    final_centroid_x[i] = data_x[mask].mean()
                    final_centroid_y[i] = data_y[mask].mean()
        
    def get_solve_signature(self) -> Dict[str, Any]:
        return {
            "data_x": ctypes.POINTER(ctypes.c_float),
            "data_y": ctypes.POINTER(ctypes.c_float),
            "labels": ctypes.POINTER(ctypes.c_int),
            "initial_centroid_x": ctypes.POINTER(ctypes.c_float),
            "initial_centroid_y": ctypes.POINTER(ctypes.c_float),
            "final_centroid_x": ctypes.POINTER(ctypes.c_float),
            "final_centroid_y": ctypes.POINTER(ctypes.c_float),
            "sample_size": ctypes.c_int,
            "k": ctypes.c_int,
            "max_iterations": ctypes.c_int
        }
    
    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        sample_size, k, max_iterations = 4, 2, 10
        data_x = torch.tensor([1.0, 2.0, 8.0, 9.0], device="cuda", dtype=dtype)
        data_y = torch.tensor([1.0, 2.0, 8.0, 9.0], device="cuda", dtype=dtype)
        labels = torch.empty(sample_size, device="cuda", dtype=torch.int32)
        initial_centroid_x = torch.tensor([1.0, 8.0], device="cuda", dtype=dtype)
        initial_centroid_y = torch.tensor([1.0, 8.0], device="cuda", dtype=dtype)
        final_centroid_x = torch.empty(k, device="cuda", dtype=dtype)
        final_centroid_y = torch.empty(k, device="cuda", dtype=dtype)
        return {
            "data_x": data_x,
            "data_y": data_y,
            "labels": labels,
            "initial_centroid_x": initial_centroid_x,
            "initial_centroid_y": initial_centroid_y,
            "final_centroid_x": final_centroid_x,
            "final_centroid_y": final_centroid_y,
            "sample_size": sample_size,
            "k": k,
            "max_iterations": max_iterations
        }
    
    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        test_cases = []
        # basic_clustering
        data_x = torch.tensor([1.0, 1.5, 1.2, 1.3, 1.1, 5.0, 5.2, 5.1, 5.3, 5.4, 10.1, 10.2, 10.0, 10.3, 10.5], device="cuda", dtype=dtype)
        data_y = torch.tensor([1.0, 1.5, 1.2, 1.3, 1.1, 5.0, 5.2, 5.1, 5.3, 5.4, 10.1, 10.2, 10.0, 10.3, 10.5], device="cuda", dtype=dtype)
        labels = torch.empty(15, device="cuda", dtype=torch.int32)
        initial_centroid_x = torch.tensor([3.4, 7.1, 8.5], device="cuda", dtype=dtype)
        initial_centroid_y = torch.tensor([3.4, 7.1, 8.5], device="cuda", dtype=dtype)
        final_centroid_x = torch.empty(3, device="cuda", dtype=dtype)
        final_centroid_y = torch.empty(3, device="cuda", dtype=dtype)
        test_cases.append({
            "data_x": data_x,
            "data_y": data_y,
            "labels": labels,
            "initial_centroid_x": initial_centroid_x,
            "initial_centroid_y": initial_centroid_y,
            "final_centroid_x": final_centroid_x,
            "final_centroid_y": final_centroid_y,
            "sample_size": 15,
            "k": 3,
            "max_iterations": 20
        })
        # single_cluster
        data_x = torch.tensor([1.0, 1.2, 1.1, 1.3, 1.5, 1.4, 1.6, 1.2, 1.3, 1.1], device="cuda", dtype=dtype)
        data_y = torch.tensor([1.0, 1.2, 1.1, 1.3, 1.5, 1.4, 1.6, 1.2, 1.3, 1.1], device="cuda", dtype=dtype)
        labels = torch.empty(10, device="cuda", dtype=torch.int32)
        initial_centroid_x = torch.tensor([1.0, 5.0, 10.0], device="cuda", dtype=dtype)
        initial_centroid_y = torch.tensor([1.0, 5.0, 10.0], device="cuda", dtype=dtype)
        final_centroid_x = torch.empty(3, device="cuda", dtype=dtype)
        final_centroid_y = torch.empty(3, device="cuda", dtype=dtype)
        test_cases.append({
            "data_x": data_x,
            "data_y": data_y,
            "labels": labels,
            "initial_centroid_x": initial_centroid_x,
            "initial_centroid_y": initial_centroid_y,
            "final_centroid_x": final_centroid_x,
            "final_centroid_y": final_centroid_y,
            "sample_size": 10,
            "k": 3,
            "max_iterations": 10
        })
        # empty_clusters
        data_x = torch.tensor([1.0, 1.5, 1.2, 1.3, 1.1, 1.4, 1.6, 1.2, 1.7, 1.3, 10.0, 10.5, 10.2, 10.3, 10.1, 10.4, 10.6, 10.2, 10.7, 10.3], device="cuda", dtype=dtype)
        data_y = torch.tensor([1.0, 1.5, 1.2, 1.3, 1.1, 1.4, 1.6, 1.2, 1.7, 1.3, 10.0, 10.5, 10.2, 10.3, 10.1, 10.4, 10.6, 10.2, 10.7, 10.3], device="cuda", dtype=dtype)
        labels = torch.empty(20, device="cuda", dtype=torch.int32)
        initial_centroid_x = torch.tensor([1.5, 5.0, 10.5], device="cuda", dtype=dtype)
        initial_centroid_y = torch.tensor([1.5, 5.0, 10.5], device="cuda", dtype=dtype)
        final_centroid_x = torch.empty(3, device="cuda", dtype=dtype)
        final_centroid_y = torch.empty(3, device="cuda", dtype=dtype)
        test_cases.append({
            "data_x": data_x,
            "data_y": data_y,
            "labels": labels,
            "initial_centroid_x": initial_centroid_x,
            "initial_centroid_y": initial_centroid_y,
            "final_centroid_x": final_centroid_x,
            "final_centroid_y": final_centroid_y,
            "sample_size": 20,
            "k": 3,
            "max_iterations": 15
        })
        # max_iterations_limit
        data_x = torch.tensor([1.0, 1.5, 1.2, 1.3, 1.1, 5.0, 5.2, 5.1, 5.3, 5.4, 10.1, 10.2, 10.0, 10.3, 10.5], device="cuda", dtype=dtype)
        data_y = torch.tensor([1.0, 1.5, 1.2, 1.3, 1.1, 5.0, 5.2, 5.1, 5.3, 5.4, 10.1, 10.2, 10.0, 10.3, 10.5], device="cuda", dtype=dtype)
        labels = torch.empty(15, device="cuda", dtype=torch.int32)
        initial_centroid_x = torch.tensor([3.4, 7.1, 8.5], device="cuda", dtype=dtype)
        initial_centroid_y = torch.tensor([3.4, 7.1, 8.5], device="cuda", dtype=dtype)
        final_centroid_x = torch.empty(3, device="cuda", dtype=dtype)
        final_centroid_y = torch.empty(3, device="cuda", dtype=dtype)
        test_cases.append({
            "data_x": data_x,
            "data_y": data_y,
            "labels": labels,
            "initial_centroid_x": initial_centroid_x,
            "initial_centroid_y": initial_centroid_y,
            "final_centroid_x": final_centroid_x,
            "final_centroid_y": final_centroid_y,
            "sample_size": 15,
            "k": 3,
            "max_iterations": 5
        })
        # medium_random
        sample_size = 100
        k = 5
        data_x = torch.empty(sample_size, device="cuda", dtype=dtype).uniform_(0.0, 100.0)
        data_y = torch.empty(sample_size, device="cuda", dtype=dtype).uniform_(0.0, 100.0)
        labels = torch.empty(sample_size, device="cuda", dtype=torch.int32)
        initial_centroid_x = torch.tensor([20.0, 40.0, 60.0, 80.0, 10.0], device="cuda", dtype=dtype)
        initial_centroid_y = torch.tensor([20.0, 40.0, 60.0, 80.0, 50.0], device="cuda", dtype=dtype)
        final_centroid_x = torch.empty(k, device="cuda", dtype=dtype)
        final_centroid_y = torch.empty(k, device="cuda", dtype=dtype)
        test_cases.append({
            "data_x": data_x,
            "data_y": data_y,
            "labels": labels,
            "initial_centroid_x": initial_centroid_x,
            "initial_centroid_y": initial_centroid_y,
            "final_centroid_x": final_centroid_x,
            "final_centroid_y": final_centroid_y,
            "sample_size": sample_size,
            "k": k,
            "max_iterations": 30
        })
        return test_cases
    
    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        sample_size = 10000
        k = 5
        data_x = torch.empty(sample_size, device="cuda", dtype=dtype).uniform_(0.0, 1000.0)
        data_y = torch.empty(sample_size, device="cuda", dtype=dtype).uniform_(0.0, 1000.0)
        labels = torch.empty(sample_size, device="cuda", dtype=torch.int32)
        initial_centroid_x = torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0], device="cuda", dtype=dtype)
        initial_centroid_y = torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0], device="cuda", dtype=dtype)
        final_centroid_x = torch.empty(k, device="cuda", dtype=dtype)
        final_centroid_y = torch.empty(k, device="cuda", dtype=dtype)
        return {
            "data_x": data_x,
            "data_y": data_y,
            "labels": labels,
            "initial_centroid_x": initial_centroid_x,
            "initial_centroid_y": initial_centroid_y,
            "final_centroid_x": final_centroid_x,
            "final_centroid_y": final_centroid_y,
            "sample_size": sample_size,
            "k": k,
            "max_iterations": 30
        } 
    


def load_cuda_lib():
    # Compile the CUDA code first
    cuda_file = Path("starter/starter.cu")
    lib_file = Path("starter/libkmeans.so")
    # Compile the CUDA code
    os.system(f"nvcc -o {lib_file} --shared -Xcompiler -fPIC {cuda_file}")
    
    # Load the compiled library
    cuda_lib = ctypes.CDLL(str(lib_file))
    return cuda_lib

def test_kmeans():
    # Initialize the challenge
    challenge = Challenge()
    
    # Load the CUDA library
    cuda_lib = load_cuda_lib()
    
    # Get the solve function and set its argument types
    solve_func = cuda_lib.solve
    solve_func.argtypes = list(challenge.get_solve_signature().values())

    def run_test(test_case):
        # Extract test case data
        data_x = test_case["data_x"]
        data_y = test_case["data_y"]
        labels = test_case["labels"]
        initial_centroid_x = test_case["initial_centroid_x"]
        initial_centroid_y = test_case["initial_centroid_y"]
        final_centroid_x = test_case["final_centroid_x"]
        final_centroid_y = test_case["final_centroid_y"]
        sample_size = test_case["sample_size"]
        k = test_case["k"]
        max_iterations = test_case["max_iterations"]

        # Create copies for CUDA and PyTorch results
        cuda_final_x = final_centroid_x.clone()
        cuda_final_y = final_centroid_y.clone()
        torch_final_x = final_centroid_x.clone()
        torch_final_y = final_centroid_y.clone()
        cuda_labels = labels.clone()
        torch_labels = labels.clone()

        print("\nTest case details:")
        print(f"Sample size: {sample_size}, k: {k}, max_iterations: {max_iterations}")

        # Run reference implementation
        challenge.reference_impl(
            data_x, data_y, torch_labels,
            initial_centroid_x, initial_centroid_y,
            torch_final_x, torch_final_y,
            sample_size, k, max_iterations
        )

        # Convert CUDA tensor pointers to ctypes
        data_x_ptr = ctypes.cast(data_x.data_ptr(), ctypes.POINTER(ctypes.c_float))
        data_y_ptr = ctypes.cast(data_y.data_ptr(), ctypes.POINTER(ctypes.c_float))
        labels_ptr = ctypes.cast(cuda_labels.data_ptr(), ctypes.POINTER(ctypes.c_int))
        initial_x_ptr = ctypes.cast(initial_centroid_x.data_ptr(), ctypes.POINTER(ctypes.c_float))
        initial_y_ptr = ctypes.cast(initial_centroid_y.data_ptr(), ctypes.POINTER(ctypes.c_float))
        final_x_ptr = ctypes.cast(cuda_final_x.data_ptr(), ctypes.POINTER(ctypes.c_float))
        final_y_ptr = ctypes.cast(cuda_final_y.data_ptr(), ctypes.POINTER(ctypes.c_float))

        # Run CUDA implementation
        solve_func(
            data_x_ptr, data_y_ptr, labels_ptr,
            initial_x_ptr, initial_y_ptr,
            final_x_ptr, final_y_ptr,
            sample_size, k, max_iterations
        )

        try:
            # Check centroids
            torch.testing.assert_close(
                cuda_final_x, 
                torch_final_x,
                rtol=challenge.rtol, 
                atol=challenge.atol,
                msg="CUDA and PyTorch centroid X results don't match"
            )
            torch.testing.assert_close(
                cuda_final_y, 
                torch_final_y,
                rtol=challenge.rtol, 
                atol=challenge.atol,
                msg="CUDA and PyTorch centroid Y results don't match"
            )
            # Check labels
            torch.testing.assert_close(
                cuda_labels,
                torch_labels,
                msg="CUDA and PyTorch labels don't match"
            )
            return True
        except AssertionError as e:
            print("\nError details:")
            print(f"Expected centroids X (PyTorch): {torch_final_x.cpu().numpy()}")
            print(f"Got centroids X (CUDA): {cuda_final_x.cpu().numpy()}")
            print(f"Expected centroids Y (PyTorch): {torch_final_y.cpu().numpy()}")
            print(f"Got centroids Y (CUDA): {cuda_final_y.cpu().numpy()}")
            print(f"Expected labels (PyTorch): {torch_labels.cpu().numpy()}")
            print(f"Got labels (CUDA): {cuda_labels.cpu().numpy()}")
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
            raise e

    # Run performance test
    print("\nRunning performance test...")
    perf_test = challenge.generate_performance_test()
    try:
        run_test(perf_test)
        print("Performance test passed!")
    except AssertionError as e:
        print("Performance test failed:", e)
        raise e

if __name__ == "__main__":
    test_kmeans()