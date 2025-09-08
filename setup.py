"""Setup script for ManiUniCon package."""

from setuptools import find_packages, setup

setup(
    name="maniunicon",
    version="0.1.0",
    description="A unified control interface for robotic manipulation",
    author="Zhengbang Zhu, Minghuan Liu, et al.",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "numpy-quaternion>=2022.4.3",  # For quaternion operations
        "typing-extensions>=4.0.0",
        "pydantic>=2.0.0",  # For data validation
        "python-dotenv>=0.19.0",  # For configuration management
        "opencv-python>=4.5.0",  # For image processing
        "pin-pink==3.3.0",  # For IK solver
        "robot_descriptions>=1.17.0",  # For loading robot URDFs
        "meshcat>=0.3.2",  # For 3D visualization
        "loop-rate-limiters>=0.1.0",  # For rate limiting in simulation
        "keyboard>=0.13.5",  # For keyboard input handling
        "pyspacemouse>=1.1.0",  # For SpaceMouse input handling
        "pynput",  # For keyboard control
        "yacs",  # For configuration
        "click",  # For command line interface
        "atomics==1.0.3",  # For atomic operations
        "threadpoolctl==3.6.0",
        "hydra-core==1.3.2",
        "torch",  # for policy model
        "open3d==0.19.0",  # for point cloud visualization
        "trimesh==4.6.10",
        "hidapi==0.14.0.post4",  # For SpaceMouse
        "zarr==2.18.3",
        "qpsolvers[daqp]==4.8.0",
    ],
    extras_require={
        "ur5": [
            "ur-rtde",
            "pyRobotiqGripper",
        ],  # Optional dependency for UR5 robot control
        "xarm": [
            "xarm-python-sdk==1.15.3",
            "pyRobotiqGripper",
        ],  # Optional dependency for XArm robot control
        "realsense": [
            "pyrealsense2",
            "opencv-python",
            "numba",
        ],  # Optional dependency for RealSense cameras
        "franky_fr3_franky": [
            "franky-control==1.1.1",
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
