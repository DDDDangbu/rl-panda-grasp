from setuptools import setup, find_packages

setup(
    name="rl-panda-grasp",
    version="1.0.0",
    description="Reinforcement Learning for Robotic Pick-and-Place with Adaptive Curriculum Learning",
    author="Jizhe Liu",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "gymnasium==0.29.1",
        "panda-gym==3.0.7",
        "pybullet==3.2.6",
        "stable-baselines3[extra]==2.3.2",
        "torch>=2.0.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "imageio>=2.31.0",
        "imageio-ffmpeg>=0.4.8",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "isort>=5.12.0",
        ],
    },
)
