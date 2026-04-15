import torch
import time
from thop import profile

def measure_model(model, inputs, warmup=10, reps=30):
    model.eval()
    model.cuda()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 1. Params & FLOPs
    macs, params = profile(model, inputs=inputs, verbose=False)


    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*inputs)

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(reps):
            _ = model(*inputs)
        torch.cuda.synchronize()
        end = time.time()

    avg_infer_time = (end - start) / reps * 1000  # ms

    # 3. 显存占用
    allocated = torch.cuda.memory_allocated() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2

    print(f"Params: {params / 1e6:.2f} M")
    print(f"FLOPs: {macs / 1e9:.2f} G")
    print(f"Inference time: {avg_infer_time:.2f} ms")
    print(f"Memory allocated: {allocated:.2f} MB")
    print(f"Memory peak: {peak:.2f} MB")
if __name__ == "__main__":
    from st_gcn import sagcn  

    model = sagcn(64, 64)
    x = torch.randn(64, 64, 129, 50).cuda()
    condition = torch.randn(64, 9, 512).cuda()
    
    measure_model(model, inputs=(x, condition))