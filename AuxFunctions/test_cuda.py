import torch

def test_cuda(flag):
    """
    Test if CUDA is available on local machine.
    
    Args:
        flag (bool): yes/no

    """
    if flag==True:
        if torch.cuda.is_available():
            print("CUDA is available. Using GPU.", flush=True)
            print(f"Device count: {torch.cuda.device_count()}", flush=True)
            for i in range(torch.cuda.device_count()):
                print(f"\tDevice {i}: {torch.cuda.get_device_name(i)}", flush=True)
        else:
            print("CUDA is not available. Using CPU.", flush=True)
    else:
        pass