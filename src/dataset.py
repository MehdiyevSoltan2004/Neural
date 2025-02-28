import dlib
print(f"CUDA поддерживается в dlib: {dlib.DLIB_USE_CUDA}")
print(f"Количество доступных GPU: {dlib.cuda.get_num_devices()}")
