from setuptools import find_packages, setup

setup(
    name="robotics",
    version="0.1",
    description="Robotics in Python",
    author="WANG Lei",
    author_email="wlbksy@126.com",
    license="MIT",
    packages=find_packages(),
    platforms=["Windows", "Linux", "Mac OS-X"],
    install_requires=["numpy", "matplotlib"],
    python_requires=">=3.5",
    classifiers=[
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3 :: Only",
        "Framework:: Robot Framework",
        "Framework:: Robot Framework:: Library",
        "Framework:: Robot Framework:: Tool",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    zip_safe=False,
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
)
