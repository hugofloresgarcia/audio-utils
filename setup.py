from setuptools import setup


with open('README.md') as file:
    long_description = file.read()


setup(
    name='audio_utils',
    description='audio utilities 4 me :)',
    version='0.0.1',
    author='Hugo Flores Garcia',
    author_email='hf01049@georgiasouthern.edu',
    url='https://github.com/hugofloresgarcia/audio_utils',
    install_requires=['numpy', 'librosa'],
    packages=['audio_utils'],
    package_data={'audio_utils': ['assets/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=[],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT')
