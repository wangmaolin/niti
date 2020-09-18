from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='int_im2col',
    ext_modules=[
        CUDAExtension('int_im2col_cuda',
                      ['int_im2col.cpp',
                       'int_im2col_kernel.cu'],
                      include_dirs=['./'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
