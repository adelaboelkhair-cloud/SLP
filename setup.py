from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip()]

setup(
    name="sign-language-production",
    version="1.0.0",
    description="AI-powered Sign Language Recognition Application",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SLP Team",
    author_email="slp@example.com",
    url="https://github.com/adelaboelkhair-cloud/SLP",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "slp=app.app:main",
        ],
    },
)
