"""Setup configuration for LLM Evaluation Platform."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llm-evaluation-platform",
    version="1.0.0",
    author="Krish Bakriwala",
    author_email="krishbakriwala8@gmail.com",
    description="A comprehensive platform for evaluating and benchmarking Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/krishbakriwala8/llm-evaluation-platform",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "python-dotenv>=1.0.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        "openai>=1.3.0",
        "anthropic>=0.7.8",
        "ragas>=0.1.0",
        "langfuse>=2.24.0",
        "sentence-transformers>=2.2.2",
        "scikit-learn>=1.3.2",
        "numpy>=1.24.3",
        "scipy>=1.11.4",
        "pandas>=2.1.1",
        "pytest>=7.4.3",
        "pytest-cov>=4.1.0",
        "pytest-asyncio>=0.21.1",
        "loguru>=0.7.2",
        "requests>=2.31.0",
        "aiohttp>=3.9.1",
        "tqdm>=4.66.1",
        "click>=8.1.7",
        "typer>=0.9.0",
        "rich>=13.7.0",
    ],
    entry_points={
        "console_scripts": [
            "llm-eval=scripts.main:main",
        ],
    },
)
