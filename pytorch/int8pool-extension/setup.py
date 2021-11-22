from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='int8pool',
    ext_modules=[
        CUDAExtension('int8pool_cuda',
                      ['int8pool.cpp',
                       'int8pool_kernel.cu',],
                      library_dirs=['/usr/lib/x86_64-linux-gnu'],
                      libraries=['cudnn'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
