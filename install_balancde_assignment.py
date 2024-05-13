from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
setup(
    name='balanced_assignment',
    ext_modules=[
        CppExtension(
            name='balanced_assignment',
            sources=['./csrc/balanced_assignment.cpp'])
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    })