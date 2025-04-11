import setuptools

setuptools.setup(
    name="transformer",
    version="0.1",
    author="Ignacio Quintanilla",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch",
        "pandas",
        "numpy",
        "huggingface_hub",
        "datasets",
        "tokenizers",
        "tensorboard",
        "ruff",
    ],
)