# allow us to import cpp functions into python

import glob 
import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
include_dirs = [os.path.join(ROOT_DIR, "include")]
sources = glob.glob("*.cpp") + glob.glob("*.cu") #list of cpp and cu files in dir

setup(
    name='rendering', 
    version='42.0',
    author='stanleyedward',
    author_email='114278820+stanleyedward@users.noreply.github.com',
    description='forward model for instant ngp',
    long_description='forward model for instantngp nerf rendering using CUDA',
    ext_modules=[
        CUDAExtension(
            name='rendering',
            include_dirs=include_dirs,
            sources=sources,
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": ["-O2"]},
            # extra_link_flags=['-Wl,--no-as-needed', '-lm']
            )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)