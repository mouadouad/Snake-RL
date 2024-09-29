import torch
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
print(torch.cuda.memory_allocated())

a = torch.rand(10000, 10000).to('cuda')
print(torch.cuda.memory_allocated())  # This should show some GPU memory usage

