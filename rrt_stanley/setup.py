from setuptools import setup
import os
from glob import glob

package_name = 'rrt_stanley'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'racelines'), glob('racelines/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Weiqi Lyu',
    maintainer_email='weiqi.lyu@tum.de',
    description='f1tenth_rrt_stanley',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rrt_stanley = rrt_stanley.rrt_stanley:main',        
        ],
    },
)

