from setuptools import setup, find_packages

setup(
    name='strawberry_firmness_prediction',
    version='0.1.0',
    description='A machine learning model for predicting strawberry yield and firmness using multilabel classification.',
    author='Hassan Hayat',
    author_email='hassan.hayat@udg.edu',
    packages=find_packages(),
    install_requires=[
        'pandas==2.0.3',
        'numpy==1.25.2',
        'scipy==1.11.2',
        'scikit-learn==1.3.0',
    ],
    entry_points={
        'console_scripts': [
            'train_model=python_code:process_and_train_model',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
