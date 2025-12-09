from setuptools import setup, find_packages


setup(
    name="nncore",
    version="0.1",
    url="https://github.com/Phi-C/nncore",
    author="Xingjian Chen",
    author_email="chenxj@163.com",
    description="A light-weight core package.",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "mysync=nncore.sync:main",
            "get_nvtx_kernel=nncore.get_nvtx_kernel:main",
        ]
    },
)
