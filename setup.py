from setuptools import setup, find_packages

setup(
  name="enformer-polaris",
  version="0.0.1",
  author="Rodrigo Bonazzola (rbonazzola)",
  author_email="rodbonazzola@gmail.com",
  description="Python package for running Enformer on ALCF's Polaris.",
  long_description=open("README.md", encoding="utf-8").read(),
  long_description_content_type="text/markdown",
  url="https://github.com/rbonazzola/enformer",
  packages=find_packages(),
  install_requires=[
    "mlflow",
    "git+https://github.com/saforem2/ezpz@mainrama#egg=ezpz"
  ],
  python_requires=">=3.8,<3.11"
)
