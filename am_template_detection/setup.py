from setuptools import setup, find_packages

package_name = 'am_template_detection'

setup(
    name=package_name,
    version='1.1',
    packages=find_packages(package_name),
    package_dir={
        "models": f"./{package_name}/models"
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='armine',
    maintainer_email='934218777@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'template_detector = am_template_detection.ObjDetectorNode:main',
        ],
    },
)
