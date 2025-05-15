import numpy as np
from pathlib import Path

import ctypes
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

# Load the CUDA shared library
# cuda_lib = ctypes.CDLL('./hist.so')

# Define argument types
# cuda_lib.histogram_wrapper.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
# cuda_lib.histogram_wrapper.restype = None

def get_kernal_from_file(file_path: Path):
    with open(file_path, "r") as f:
        cuda_code = f.read()
    
    mod = SourceModule(cuda_code)
    kernal = mod.get_function(file_path.stem)
    kernal.prepare("F")
    return kernal

def histogram_kernal_runner(text, c_kernal):
    # initialize data
    lenBuffer = np.int32(len(text))
    buffer = np.frombuffer(text.encode('ascii'), dtype=np.byte)
    
    numBins = np.int32((26 + 3) // 4) 
    histogram = np.zeros(numBins, dtype=np.int32)
    
    # Kernal Launch
    threads = (256, 1, 1)
    blocks = ((int(lenBuffer) + threads[0] - 1)//threads[0], 1, 1)
    shared_mem_size = int(numBins * np.dtype(np.int32).itemsize)
    
    
    # Synchronize before kernel launch
    cuda.Context.synchronize()

    c_kernal.prepare_call(
        cuda.In(buffer), cuda.Out(histogram), lenBuffer, numBins,
        block=threads,
        grid=blocks,
        shared_size=shared_mem_size,
    )

    cuda.Context.synchronize()
    
    return histogram


if __name__ == "__main__":
    with open("long_text.txt", "r") as f:
        text = f.read()
        
    c_kernal = get_kernal_from_file(Path("histogram_kernal.cu"))
    histogram = histogram_kernal_runner(text, c_kernal)
    
    print("Histogram:", histogram)
