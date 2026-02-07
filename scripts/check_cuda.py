import sys
try:
    import torch
except Exception as e:
    print('torch_import_error', e)
    sys.exit(2)

print('torch_version', torch.__version__)
print('cuda_available', torch.cuda.is_available())
try:
    print('device_count', torch.cuda.device_count())
    if torch.cuda.is_available():
        print('device_name_0', torch.cuda.get_device_name(0))
except Exception as e:
    print('cuda_probe_error', e)
