from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cutlassconv',
    ext_modules=[
        CUDAExtension('cutlassconv_cuda',
                      ['cutlassconv.cpp',
                       'cutlassconv_kernel.cu',],
                      include_dirs=['/niti/pytorch/cutlass-extension/include','/niti/pytorch/cutlass-extension/util/include'])],
    cmdclass={
        'build_ext': BuildExtension
    })
