from setuptools import find_packages, setup
import torch
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

from distutils.command.build import build
from setuptools.command.install import install

from setuptools.command.develop import develop

import os
import subprocess
import platform
BASEPATH = os.path.dirname(os.path.abspath(__file__))

compile_args = []
link_args = []

if platform.system() != 'Darwin':  # add openmp
    compile_args.append('-fopenmp')
    link_args.append('-lgomp')
    
ext_modules=[CppExtension('extlib', 
                          ['gln/mods/torchext/src/extlib.cpp'],
                          extra_compile_args=compile_args,
                          extra_link_args=link_args)]

# build cuda lib
if torch.cuda.is_available():
    ext_modules.append(CUDAExtension('extlib_cuda',
                                    ['gln/mods/torchext/src/extlib_cuda.cpp', 'gln/mods/torchext/src/extlib_cuda_kernels.cu'],
                                    extra_compile_args=compile_args,
                                    extra_link_args=link_args))

class custom_develop(develop):
    def run(self):
        original_cwd = os.getcwd()

        folders = [
            os.path.join(BASEPATH, 'gln/mods/mol_gnn/mg_clib'),
        ]
        for folder in folders:
            os.chdir(folder)
            subprocess.check_call(['make'])

        os.chdir(original_cwd)

        super().run()

setup(name='gln',
      packages=find_packages(include=('gln', 'gln.*')),
      ext_modules=ext_modules,
      install_requires=[
          # Native extensions are compiled against PyTorch's C++ ABI.  Keep
          # this in lockstep with pyproject.toml's isolated-build dependency.
          'torch==2.12.1',
          'rich>=13',
      ],
      cmdclass={
          'develop': custom_develop,
          'build_ext': BuildExtension,
        }
)
