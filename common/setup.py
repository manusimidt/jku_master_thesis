import setuptools

setuptools.setup(
    name="common",
    version="0.0.1",
    description="Functions used in both environments",
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    author="Manuel Schmidt"
)

# install with:
# > pip install -e .
