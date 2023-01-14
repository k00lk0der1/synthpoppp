from synthpop.zone_synthesizer import synthesize_all_zones, load_data
from . import helper_functions
import pandas as pd
import numpy as np
import random
from io import StringIO


class IPU():
	def __init__(self):
		pass

	def preprocess_individual_samples_ihds(self, filtered_ihds_individuals_data):

		# columns_to_keep_individuals = ['DISTRICT', 'IDHH', 'PERSONID', 'RO3', 'RO6', 'RO5','ED2', 'ID11', 'ID13', 'RO7', 'URBAN2011']
		columns_to_keep_individuals = ['DIST01','IDHH','PERSONID', 'RO3', 'RO5', 'ID11', 'ID13']
		columns_rename_dict_individuals = {'RO3':'SexLabel', 'RO5':'Age', 
			'RO6':'marital_status',
			'ED2':'literacy', 'ED6':'edu_years', 'EDUC7': 'edu_cat',
			'ID11':'religion', 'ID13':'caste', 
			'URBAN2011':'residence',
			'WS4':'job', 
			'RO7':'activity_status', 
			'IDHH':'serialno', 
			'PERSONID':'mem_id',
			'DIST01':'district', 
			'MB3':'M_Cataract', 'MB4':'M_TB', 'MB5':'M_High_BP',
			'MB6':'M_Heart_disease', 'MB7':'M_Diabetes', 'MB8':'M_Leprosy',
			'MB9':'M_Cancer', 'MB10':'M_Asthma', 'MB11':'M_Polio',
			'MB12':'M_Paralysis', 'MB13':'M_Epilepsy', 'SM4':'M_Fever', 'SM5':'M_Cough',
			'SM7':'M_Diarrhea'}

		filtered_ihds_individuals_data = filtered_ihds_individuals_data[columns_to_keep_individuals]
		filtered_ihds_individuals_data = filtered_ihds_individuals_data.rename(columns_rename_dict_individuals, axis='columns')

		individuals_data = filtered_ihds_individuals_data.dropna()

		gender_dict = {1:'Male', '1':'Male', 2:'Female', '2':'Female'}
		individuals_data['SexLabel'] = individuals_data['SexLabel'].map(gender_dict)

		# individuals_data.loc[individuals_data['marital_status']==' ','marital_status'] = 1
		# individuals_data['marital_status'] = individuals_data['marital_status'].astype(int)
		# marital_dict = {0:'married', 1:'married', 2:'unmarried', 3:'widowed', 4: 'separated', 5: 'married'}
		# individuals_data['marital_status'] = individuals_data['marital_status'].map(marital_dict)

		# individuals_data.loc[individuals_data['literacy']==' ','literacy'] = 0
		# individuals_data['activity_status'] = individuals_data['activity_status'].astype(int)
		# individuals_data.loc[(individuals_data['literacy']==' ') & (individuals_data['activity_status'] >=5) & (individuals_data['activity_status']<=10),'literacy']=1
		# individuals_data.loc[(individuals_data['literacy']==' ') & (individuals_data['activity_status'] ==12) & (individuals_data['age']!='0to4'),'literacy']=1

		# individuals_data['literacy'] = individuals_data['literacy'].astype(int)
		# individuals_data.loc[individuals_data.literacy == 1, 'literacy'] = 'literate'
		# individuals_data.loc[individuals_data.literacy == 0, 'literacy'] = 'illiterate'

		bins= [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,110]
		labels = ['0to4', '5to9', '10to14', '15to19','20to24', '25to29','30to34', '35to39', '40to44', '45to49',
				 '50to54', '55to59', '60to64', '65to69', '70to74', '75to79', '80p']

		individuals_data['Age'] = individuals_data['Age'].apply(lambda x : helper_functions.try_convert(x, np.float('nan'), int) )   

		individuals_data['Age'] = pd.cut(individuals_data['Age'], bins=bins, labels=labels, right=False)

		religion_dict = {1: 'Hindu', 2:'Muslim', 3:'Christian', 4:'Sikh', 5:'Buddhist', 6:'Jain',
						7: 'Other', 8:'Other', 9:'Other'}
		individuals_data['religion'] = individuals_data['religion'].map(religion_dict)

		individuals_data.loc[individuals_data['caste']==' ', 'caste'] =  random.randint(1,6)
		individuals_data.loc[(individuals_data['caste']==' ') & (individuals_data['religion']!='hindu'),'caste'] = 6
		individuals_data['caste'] = individuals_data['caste'].astype(int)
		caste_dict = {4: 'SC', 5:'ST', 1:'other', 2:'Other', 3:'Other', 6:'Other'}
		individuals_data['caste'] = individuals_data['caste'].map(caste_dict)

		# urbandict = {1:'urban', 0:'rural'}
		# individuals_data['residence'] = individuals_data['residence'].map(urbandict)

		# individuals_data['working'] = 'yes'
		# individuals_data['activity_status'] = individuals_data['activity_status'].apply(lambda x : helper_functions.try_convert(x, np.float('nan'), int) )   
		# individuals_data.loc[individuals_data.activity_status >= 10, 'working'] = 'no'

		individuals_data = individuals_data.drop(['job', 'activity_status', 'edu_years'], axis=1, errors='ignore')

		# individuals_data.loc[individuals_data['literacy']=='illiterate','edu_cat'] = 'illiterate'
		# individuals_data['edu_cat'] = individuals_data['edu_cat'].astype(str)
		# individuals_data.loc[individuals_data['edu_cat']=='0','edu_cat'] = 'illiterate'
		# edu_dict = {'3': 'below_primary', '5':'primary', '8':'middle', '10':'secondary', '12':'senior_secondary',
				#    '15':'grad_p', '16':'grad_p'}
		# individuals_data['edu_cat'].replace(edu_dict, inplace=True)

		display(individuals_data)
		individuals_data.to_csv("individual_microdata.csv")
		return individuals_data

	def preprocess_household_samples_ihds(self, filtered_ihds_households_data):

		columns_to_keep_households = ['DIST01', 'IDHH', 'NPERSONS']
		columns_rename_dict_households = {'URBAN2011':'residence', 'IDHH':'serialno','DIST01':'district', 'NPERSONS':'hhsize'}

		households_data = filtered_ihds_households_data[columns_to_keep_households]
		households_data = households_data.rename(columns_rename_dict_households, axis='columns')

		# urbandict = {1:'urban', 0:'rural'}
		# households_data['residence'] = households_data['residence'].map(urbandict)

		hhsize_dict = {1:'hhsize_1', 2:'hhsize_2', 3:'hhsize_3', 4:'hhsize_4', 5:'hhsize_5',
					  6:'hhsize_6', 7:'hhsize_710', 8:'hhsize_710', 9:'hhsize_710',
					  10:'hhsize_710', 11:'hhsize_1114', 12:'hhsize_1114', 13:'hhsize_1114',
					  14:'hhsize_1114'}

		households_data.loc[households_data['hhsize'] >=15, 'hhsize'] = 'hhsize_15p'
		households_data['hhsize'] = households_data['hhsize'].replace(hhsize_dict)
		display(households_data)
		households_data.to_csv("households_microdata.csv")
		return households_data

	def load_marginals(self, householdh_marginal_filename, individuals_marginal_filename, individuals_data, households_data):
		empty_file_households = StringIO("1,2,3") #Creating empty files so that load_data function can be used which is built to load samples as well
		empty_file_individuals = StringIO("1,2,3") #Creating empty files so that load_data function can be used which is built to load samples as well

		household_marginal, individuals_marginal, hh_sample_empty, p_sample_empty, xwalk = load_data(householdh_marginal_filename, individuals_marginal_filename, empty_file_households, empty_file_individuals)

		display(household_marginal)
		display(individuals_marginal)
		household_marginal = household_marginal[list(household_marginal.columns)].astype(float)
		household_marginal = household_marginal[list(household_marginal.columns)].astype(int)

		individuals_marginal = individuals_marginal[list(individuals_marginal.columns)].astype(float)
		individuals_marginal = individuals_marginal[list(individuals_marginal.columns)].astype(int)

		district_dict = pd.Series(individuals_marginal.index, index=individuals_marginal.distid.distid.values).to_dict()
		individuals_data['district'] = individuals_data['district'].replace(district_dict)
		households_data['district'] = households_data['district'].replace(district_dict)
		households_data['sample_geog'] = 1
		individuals_data['sample_geog'] = 1

		household_marginal.drop('distid', axis=1, inplace=True)

		individuals_marginal = individuals_marginal.drop(['distid','total_pop'], axis=1)
		# individuals_marginal = individuals_marginal.drop(['illiterate_males','illiterate_females', 
							#  'literate_males', 'literate_females',
							#  'marginal_less3', 'marginal_6', 'non_worker'], axis=1, level=1)
		# individuals_marginal = individuals_marginal.rename({'main_workers': 'yes', 'non_worker2': 'no'}, axis='columns', level=1)

		# individuals_marginal[('marital_status','separated')] = (individuals_marginal['marital_status']['separated'] + individuals_marginal['marital_status']['divorced']).values

		# individuals_marginal[('edu_cat','senior_secondary')] = (individuals_marginal['edu_cat']['senior_secondary'] + individuals_marginal['edu_cat']['dip_cert_nontech'] + individuals_marginal['edu_cat']['dip_cert_tech']).values
		# individuals_marginal[('edu_cat','illiterate')] = (individuals_marginal['edu_cat']['illiterate'] + individuals_marginal['edu_cat']['lit_wo_edu']).values

		# individuals_marginal.drop(['divorced','dip_cert_nontech', 'dip_cert_tech', 'lit_wo_edu'], axis=1, level=1, inplace=True)

		# individuals_marginal = individuals_marginal.drop(['marital_status', 'edu_cat'], axis=1)

		district_not_in_survey = [] ####### remove rows based on data. This step needs to be adjusted when we add many rows to the marginal file.
		xwalk_dict = dict(xwalk)
		xwalk_dict = {key: xwalk_dict[key] for key in xwalk_dict if key not in district_not_in_survey}
		xwalk = list(tuple(xwalk_dict.items()))

		return household_marginal, individuals_marginal, households_data, individuals_data, xwalk

	def generate_data(self, filtered_ihds_individuals_data, filtered_ihds_households_data, householdh_marginal_filename, individuals_marginal_filename):
		individuals_data = self.preprocess_individual_samples_ihds(filtered_ihds_individuals_data)

		households_data = self.preprocess_household_samples_ihds(filtered_ihds_households_data)

		household_marginal, individuals_marginal, households_data, individuals_data, xwalk = self.load_marginals(householdh_marginal_filename, individuals_marginal_filename, individuals_data, households_data)
	
		individuals_data.dropna(inplace=True)
		households_data.dropna(inplace=True)  

		synthetic_households, synthetic_individuals, synthetic_stats = synthesize_all_zones(household_marginal, individuals_marginal, households_data, individuals_data, xwalk)
		synthetic_households['household_id'] = synthetic_households.index
		return synthetic_households, synthetic_individuals, synthetic_stats
