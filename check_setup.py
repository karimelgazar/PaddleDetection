import paddle

print("paddle.__version__ =", paddle.__version__)
print("paddle.version.cuda() =", paddle.version.cuda())
print("compiled_with_cuda =", paddle.is_compiled_with_cuda())
device_count = paddle.device.cuda.device_count()
print("cuda_device_count =", device_count)
assert paddle.is_compiled_with_cuda(), "Paddle is not a CUDA build"
assert device_count > 0, "No CUDA GPU visible to Paddle"
print("set_device =", paddle.set_device("gpu:0"))
