from . import helper_functions

import numpy as np
import pandas as pd
import random
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
from shapely.ops import unary_union
import gc

from tqdm import tqdm
tqdm.pandas()

class PopulationDensitySampler:
	def __init__(self, population_density_filename):
		self.population_density_data = pd.read_csv(population_density_filename)
		columns_rename = {"X":"longitude", "Y":"latitude", "Z":"population_density"}
		self.population_density_data['X'] = self.population_density_data['X'].round(6)
		self.population_density_data['Y'] = self.population_density_data['Y'].round(6)
		self.population_density_data.rename(columns_rename, axis=1, inplace=True)
		self.population_density_data['point_object'] = self.population_density_data.progress_apply(lambda x : Point(x['longitude'], x['latitude']), axis=1)

	def add_point(self, latitude, longitude):
		distances = pow(self.population_density_data['latitude']-latitude, 2) + pow(self.population_density_data['longitude']-longitude,2)
		sorted_df = self.population_density_data.loc[distances.sort_values().index]
		mean_population_density = sorted_df.iloc[:4]['population_density'].mean()
		
		new_row_index = len(self.population_density_data)
		
		self.population_density_data.at[new_row_index, 'longitude'] =  longitude
		self.population_density_data.at[new_row_index, 'latitude'] = latitude
		self.population_density_data.at[new_row_index, 'population_density'] = mean_population_density
		self.population_density_data.at[new_row_index, 'point_object'] = Point(longitude, latitude)

	def get_lat_long_samples(self, n, polygon):
		subset = self.population_density_data[self.population_density_data['point_object'].progress_apply(polygon.contains)]
		
		if(len(subset)==0):
			raise Exception("No points within the given polygon")
		
		sample = subset.sample(weights='population_density', n=(n*10), replace=True).copy()
		
		sample.reset_index(drop=True, inplace=True)
		
		sample['latitude'] = sample['latitude'] + np.random.uniform(-0.015, 0.015, size=sample.shape[0])
		
		sample['longitude'] = sample['longitude'] + np.random.uniform(-0.015, 0.015, size=sample.shape[0])
		
		points = sample.progress_apply(lambda x : Point(x['longitude'], x['latitude']), axis=1)
		
		contained = points.progress_apply(polygon.contains)
		
		return sample[contained][['longitude', 'latitude']].sample(n, replace=True).values

class HLatHlongAgeAddition:
	def __init__(self, admin_units_geojson_filename, admin_units_population_filename, population_density_filename):
		self.admin_units = gpd.read_file(admin_units_geojson_filename)
		self.admin_units.sort_values(by='name', inplace=True)
		self.admin_units.reset_index(drop=True, inplace=True)
		self.population_density_sampler = PopulationDensitySampler(population_density_filename)

		self.admin_unit_wise_population = pd.read_csv(admin_units_population_filename)

		self.admin_unit_wise_population['lower_limit'] = self.admin_unit_wise_population['TOT_P'].cumsum()-self.admin_unit_wise_population['TOT_P']
		self.admin_unit_wise_population['upper_limit'] = self.admin_unit_wise_population['TOT_P'].cumsum()

		for admin_unit in self.admin_units.iterrows():
			admin_unit_centroid = admin_unit[1]['geometry'].centroid
			self.population_density_sampler.add_point(admin_unit_centroid.y, admin_unit_centroid.x)

		self.total_population = int(np.ceil(self.admin_unit_wise_population['TOT_P'].sum()/10000)*10000)

	def perform_transforms(self, synthetic_population, synthetic_households):
		synthetic_population['Age'] = synthetic_population['Age'].apply(lambda x : random.randint(80,95) if (x=="80p") else int(x.split("to")[0])) + np.random.randint(0,5,size=len(synthetic_population))

		synthetic_households['hhsize'] = synthetic_population.groupby('household_id').size()

		for admin_unit_wise_population_info in self.admin_unit_wise_population.iterrows():
			subset_index = (synthetic_households['hhsize'].cumsum()>=admin_unit_wise_population_info[1]['lower_limit']) & (synthetic_households['hhsize'].cumsum()<=admin_unit_wise_population_info[1]['upper_limit'])
			synthetic_households.loc[subset_index, 'AdminUnitName'] = admin_unit_wise_population_info[1]['Name']
			synthetic_households.loc[subset_index, 'AdminUnitLatitude'] = admin_unit_wise_population_info[1]['Latitude']
			synthetic_households.loc[subset_index, 'AdminUnitLongitude'] = admin_unit_wise_population_info[1]['Longitude']

		synthetic_households.dropna(inplace=True)

		synthetic_households[['H_Lat', 'H_Lon']] = None

		for admin_unit_name in synthetic_households['AdminUnitName'].unique():
			print(admin_unit_name)
			admin_unit_polygon = self.admin_units[self.admin_units['name']==admin_unit_name]['geometry'].iloc[0]
			admin_unit_houses_index = synthetic_households['AdminUnitName']==admin_unit_name
			n_houses_in_admin_unit = len(synthetic_households[admin_unit_houses_index])
			points = self.population_density_sampler.get_lat_long_samples(n_houses_in_admin_unit, admin_unit_polygon)
			synthetic_households.loc[admin_unit_houses_index, ['H_Lon', 'H_Lat']] = points

		synthetic_households.index.name = 'hh_index'

		columns_to_join = ['household_id', 'H_Lat', 'H_Lon', 'AdminUnitName', 'AdminUnitLatitude', 'AdminUnitLongitude']
		merged_df = pd.merge(synthetic_population, synthetic_households[columns_to_join] ,on='household_id')
		
		return merged_df

class JobsPlacesAddition:
	def __init__(self, job_type_list, admin_units_geojson_filename, n_workplaces, n_public_places, population_density_filename, workplaces_p_type, schools_p_type, public_places_p_type):
		self.job_type_list = job_type_list

		self.admin_units = gpd.read_file(admin_units_geojson_filename)
		self.admin_units.sort_values(by='name', inplace=True)
		self.admin_units.reset_index(drop=True, inplace=True)
		self.combined_boundary = unary_union(self.admin_units['geometry'])

		self.population_density_sampler = PopulationDensitySampler(population_density_filename)

		self.n_workplaces = n_workplaces
		self.n_public_places = n_public_places
		self.city_id = 1
		self.workplaces_p_type = workplaces_p_type
		self.schools_p_type = schools_p_type
		self.public_places_p_type = public_places_p_type
		
		self.generate_places()
		
	
	def generate_workplaces(self):
		if(self.n_workplaces>len(list(set(self.job_type_list)))):
			random_workplace_types = np.random.choice(list(set(self.job_type_list)), self.n_workplaces-len(list(set(self.job_type_list))), replace=True)
			workplace_types = list(random_workplace_types)+list(set(self.job_type_list))+['Teachers']
		else:
			workplace_types = list(set(self.job_type_list))+['Teachers']
		print(len(workplace_types))
		lat_lon_pairs = self.population_density_sampler.get_lat_long_samples(len(workplace_types), self.combined_boundary)
		workplace_lats = lat_lon_pairs.T[1]
		workplace_longs = lat_lon_pairs.T[0]
		workplace_names = [2*pow(10,12)+self.city_id*pow(10,9)+counter for counter in range(len(workplace_types))]
		self.workplaces = pd.DataFrame([workplace_names, workplace_types, workplace_lats, workplace_longs]).T
		self.workplaces.columns = ['WorkplaceID', 'JobType', 'W_Lat', 'W_Lon']
		self.workplaces.sort_values(by='JobType', inplace=True)
		self.workplaces.reset_index(inplace=True, drop=True)
		
	def generate_schools(self):
		teachers_workplaces = self.workplaces[self.workplaces['JobType']=='Teachers'].copy()
		self.schools = pd.DataFrame([teachers_workplaces['WorkplaceID'], teachers_workplaces['W_Lat'], teachers_workplaces['W_Lon']]).T
		self.schools.columns = ['SchoolID', 'School_Lat', 'School_Lon']
		self.schools.reset_index(inplace=True, drop=True)
		self.schools['SchoolType'] = 'school'

	def generate_public_places(self):
		public_places_number = self.n_public_places
		lat_lon_pairs = self.population_density_sampler.get_lat_long_samples(public_places_number, self.combined_boundary)
		public_place_lats = lat_lon_pairs.T[1]
		public_place_longs = lat_lon_pairs.T[0]
		public_place_names = [3*pow(10,12)+self.city_id*pow(10,9)+counter for counter in range(public_places_number)]
		public_place_types = np.random.choice(['park', 'mall', 'gym'], public_places_number, replace=True)
		self.public_places = pd.DataFrame([public_place_names, public_place_types, public_place_lats, public_place_longs]).T
		self.public_places.columns = ['PublicPlaceID', 'PublicPlaceType', 'PublicPlaceLat', 'PublicPlaceLong']
	
	def generate_places(self):
		self.generate_workplaces()
		self.generate_schools()
		self.generate_public_places()

	def assign_workplaces(self, adult_synthetic_population):
		individuals = adult_synthetic_population
		indicies = []
		workplaceids = []
		for group in tqdm(individuals.groupby(['JobType', 'WorksAtSameCategory'])):
			current_individuals_set = individuals[(individuals['JobType']==group[0][0])&(individuals['WorksAtSameCategory']==group[0][1])]
			if(group[0][1]):
				current_workplaces_set = self.workplaces[self.workplaces['JobType']==group[0][0]]
			else:
				current_workplaces_set = self.workplaces[self.workplaces['JobType']!=group[0][0]]
			individuals_geocode = current_individuals_set[['H_Lat', 'H_Lon']].values.astype(np.float32)
			workplaces_geocode = current_workplaces_set[['W_Lat', 'W_Lon']].values.astype(np.float32)
			place_index = helper_functions.get_probabilistic_place_assignment(individuals_geocode, workplaces_geocode, p_type=self.workplaces_p_type)
			indicies.append(current_individuals_set.index.values)
			workplaceids.append(current_workplaces_set['WorkplaceID'].reset_index(drop=True).iloc[place_index].values)
		individuals['WorkplaceID'] = pd.Series(np.concatenate(workplaceids), index=np.concatenate(indicies))
		individuals = pd.merge(individuals, self.workplaces[['WorkplaceID', 'W_Lat', 'W_Lon']] ,on='WorkplaceID')
		return individuals
	
	def assign_schools(self, child_synthetic_population):
		individuals_geocode = child_synthetic_population[['H_Lat', 'H_Lon']].values.astype(np.float32)
		schools_geocode = self.schools[['School_Lat', 'School_Lon']].values.astype(np.float32)
		school_index = helper_functions.get_probabilistic_place_assignment(individuals_geocode, schools_geocode, p_type=self.schools_p_type)
		child_synthetic_population['SchoolID'] = pd.Series(self.schools['SchoolID'].iloc[school_index].values, index=child_synthetic_population.index)
		child_synthetic_population = pd.merge(child_synthetic_population, self.schools[['SchoolID', 'School_Lat', 'School_Lon']], on='SchoolID')
		return child_synthetic_population
	
	def assign_public_places(self, synthetic_population):
		individuals_geocode = synthetic_population[['H_Lat', 'H_Lon']].values.astype(np.float32)
		public_places_geocode = self.public_places[['PublicPlaceLat', 'PublicPlaceLong']].values.astype(np.float32)
		public_places_index = helper_functions.get_probabilistic_place_assignment(individuals_geocode, public_places_geocode, p_type=self.public_places_p_type)
		synthetic_population['PublicPlaceID'] = pd.Series(self.public_places['PublicPlaceID'].iloc[public_places_index].values, index=synthetic_population.index)
		synthetic_population = pd.merge(synthetic_population, self.public_places[['PublicPlaceID', 'PublicPlaceLat', 'PublicPlaceLong']], on='PublicPlaceID')
		return synthetic_population
	
	def perform_transforms(self, synthetic_population):
		adults = synthetic_population[synthetic_population['Age']>18]
		adults['JobType'] = np.random.choice(self.job_type_list, size=(adults.shape[0],))
		adults['WorksAtSameCategory'] = np.random.uniform(size=(adults.shape[0],))>0.05
		adults = self.assign_workplaces(adults)
  
		gc.collect()
		children = synthetic_population[synthetic_population['Age']<19]
		children['JobType'] = 'Student'
		children['WorksAtSameCategory'] = True
		children = self.assign_schools(children)
		gc.collect()

		total_population = pd.concat([adults,children], axis=0)
		total_population = self.assign_public_places(total_population)
		return total_population
