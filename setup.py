#Basic Setup File
from setuptools import setup

setup(
	name='synthpoppp',
	version=0.1,
	install_requires=[
		'synthpop @ git+https://github.com/UDST/synthpop/',
		'scipy==1.4.1',
		'numpy',
		'pandas',
		'geopandas',
		'tqdm',
		'shapely',
	],
	packages=['synthpoppp']
)