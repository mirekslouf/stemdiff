import setuptools

def get_version():
    with(open("src/stemdiff/__init__.py", "r")) as fh:
        for line in fh:
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
        else:
            raise RuntimeError("Unable to find version string.")
                
def get_long_description():
    with open("README.md", "r") as fh: description = fh.read()
    return(description)
    
setuptools.setup(
    name="stemdiff",
    version=get_version(),
    author="Mirek Slouf",
    author_email="mirek.slouf@gmail.com",
    description="Convert 4D-STEM data to 2D-powder diffraction pattern.",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/mirekslouf/stemdiff/",
    project_urls={
        "Documentation": "https://mirekslouf.github.io/stemdiff/"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"],
    license='MIT',
    package_dir={"":"src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=
        ["numpy", "matplotlib", "scipy", "pandas", "scikit-image"], 
    include_package_data=True)
