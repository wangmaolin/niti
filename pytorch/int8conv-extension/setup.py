from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='int8conv',
    ext_modules=[
        CUDAExtension('int8conv_cuda',
                      ['int8conv.cpp',
                       'int8conv_kernel.cu',],
                      library_dirs=['/usr/lib/x86_64-linux-gnu'],
                      libraries=['cudnn'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
