import setuptools
from glob import glob

# Will load the README.md file into a long_description of the package
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
# Load the requirements file
with open('requirements.txt') as f:
    required = f.read().splitlines()
if __name__ == "__main__":
    setuptools.setup(
        name='SchNet4AIM',
        version='1.0',
        author='Miguel Gallegos',
        author_email='gallegosmiguel@uniovi.es',
        description="A code to train SchNet models on the prediction of atomic and pairwise properties.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url='https://github.com/m-gallegos/SchNet4AIM',
        project_urls = {
            "SchNet4AIM": "https://github.com/m-gallegos/SchNet4AIM"
        },
        license='MIT',
        install_requires=required,
        zip_safe= False,
        package_dir={"": "src"},
        package_data={
        'SchNet4AIM': ['examples/*'],
        },
        packages=setuptools.find_packages(where="src"),
    )
