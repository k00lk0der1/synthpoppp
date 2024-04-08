#Basic Setup File
from setuptools import setup

setup(
	name='synthpoppp',
	version=0.1,
	install_requires=[
		'synthpop @ git+https://github.com/bhaveshneekhra/UDST_synthpop/',
		'scipy',
		'numpy',
		'pandas',
		'geopandas',
		'tqdm',
		'shapely',
	],
	packages=['synthpoppp']
)
