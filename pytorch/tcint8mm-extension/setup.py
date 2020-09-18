from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='int8mm',
    ext_modules=[
        CUDAExtension('int8mm_cuda',
                      ['int8mm.cpp',
                       'int8mm_kernel.cu',],
                      include_dirs=['/usr/local/cuda/samples/common/inc'],
                      extra_compile_args={'cxx': ['-g'],
                                          'nvcc': ['-lcublasLt']}
                      )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
