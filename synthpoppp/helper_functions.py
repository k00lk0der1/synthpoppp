import numpy as np

TINY = 1e-20

###Helper functions for data cleaning
def try_convert(value, default, *types):
	for t in types:
		try:
			return t(value)
		except (ValueError, TypeError):
			continue
	return default

def get_probabilistic_place_assignment_batch_zipf(individuals_geocode, places_geocode):
	individuals_geocode_sq_sum = np.power(np.linalg.norm(individuals_geocode, axis=1, keepdims=True),2)
	workplaces_geocode_sq_sum = np.power(np.linalg.norm(places_geocode, axis=1, keepdims=True),2)
	inverse_distances = np.power(2, -(individuals_geocode_sq_sum+workplaces_geocode_sq_sum.T-2*np.matmul(individuals_geocode, places_geocode.T)+TINY))
	inverse_distances = inverse_distances/inverse_distances.sum(axis=1,keepdims=True)
	return (inverse_distances.cumsum(axis=1)>np.random.uniform(size=(individuals_geocode.shape[0],1))).argmax(axis=-1)

def get_probabilistic_place_assignment_batch_default(individuals_geocode, places_geocode):
	individuals_geocode_sq_sum = np.power(np.linalg.norm(individuals_geocode, axis=1, keepdims=True),2)
	workplaces_geocode_sq_sum = np.power(np.linalg.norm(places_geocode, axis=1, keepdims=True),2)
	inverse_distances = 1/(individuals_geocode_sq_sum+workplaces_geocode_sq_sum.T-2*np.matmul(individuals_geocode, places_geocode.T)+TINY)
	inverse_distances = inverse_distances/inverse_distances.sum(axis=1,keepdims=True)
	return (inverse_distances.cumsum(axis=1)>np.random.uniform(size=(individuals_geocode.shape[0],1))).argmax(axis=-1)

def get_probabilistic_place_assignment(individuals_geocode, places_geocode, batch_size=10000, p_type='default'):
	n_batches = int(np.ceil(len(individuals_geocode)/batch_size))
	batch_wise_indicies = []

	if(p_type=='default'):
		get_probabilistic_place_assignment_batch = get_probabilistic_place_assignment_batch_default
	elif(p_type=='zipf'):
		get_probabilistic_place_assignment_batch = get_probabilistic_place_assignment_batch_zipf
	else:
		get_probabilistic_place_assignment_batch = get_probabilistic_place_assignment_batch_default

	for batch_counter in range(n_batches):
		batch_wise_indicies.append(get_probabilistic_place_assignment_batch(individuals_geocode[(batch_counter)*batch_size:(batch_counter+1)*batch_size], places_geocode))
	return np.concatenate(batch_wise_indicies)
