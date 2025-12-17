from setuptools import setup, find_packages
from typing import List
#define function for getting the list of requirements
def get_requirements(file_path: str) -> List[str]:
    requirements=[]
    with open (file_path) as obj:
        requirements=obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')
            
    return requirements
            


setup(    name='diabetic_retinopathy_detection',
    version='0.0.1',
    packages=find_packages(),
    author='Kavya Baxi, Swarit Samiran',
    author_email='baxikavya2018@gmail.com, swaritsamiran@gmail.com',
    install_requires=get_requirements('requirements.txt')
    )