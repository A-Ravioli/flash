"""
Script to run distributed tests across multiple GPUs.
"""

import os
import sys
import subprocess
import torch

def get_free_port():
    """Get a free port using socket."""
    import socket
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

def main():
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        print("No GPUs available. Distributed tests require at least one GPU.")
        sys.exit(1)
    
    # Use all available GPUs, up to 4
    world_size = min(num_gpus, 4)
    
    # Get a free port for the master process
    master_port = get_free_port()
    
    # Run the distributed test script on each GPU
    processes = []
    for rank in range(world_size):
        env = os.environ.copy()
        env.update({
            'MASTER_ADDR': 'localhost',
            'MASTER_PORT': str(master_port),
            'WORLD_SIZE': str(world_size),
            'RANK': str(rank),
            'CUDA_VISIBLE_DEVICES': str(rank)
        })
        
        cmd = [
            sys.executable,
            '-m',
            'pytest',
            'tests/test_distributed.py',
            '-v'
        ]
        
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        processes.append(process)
    
    # Wait for all processes to complete
    exit_codes = []
    for rank, process in enumerate(processes):
        stdout, stderr = process.communicate()
        exit_code = process.wait()
        exit_codes.append(exit_code)
        
        print(f"=== Output from rank {rank} ===")
        print(stdout.decode())
        if stderr:
            print(f"=== Error output from rank {rank} ===")
            print(stderr.decode())
    
    # Check if all processes succeeded
    if any(exit_codes):
        print("Some tests failed!")
        sys.exit(1)
    else:
        print("All distributed tests passed!")

if __name__ == '__main__':
    main() 