import tensorflow as tf, numpy as np, data_v2, os, math, pickle, matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras import backend as K
from classes_v2 import modelSet, modelDictVal

huberDelta = .5

"""WE ARE BASING THIS ON TENSFORFLOW AS I WASN'T ABLT TO GET RANSAC KERNEL TO WORK ON CUDA T_T"""

def L1_smooth(y_true, y_pred):
	x = tf.keras.backend.abs(y_true - y_pred)
	x = tf.where(x < huberDelta, 0.5 * x ** 2, huberDelta * (x - 0.5 * huberDelta))
	return	tf.keras.backend.sum(x)


"""THE MODEL IS BASED ON THE PVNET, which inherently kinda looks like RESNET"""

def coords_output(x): # add coordinate output layer
	coords = tf.keras.layers.Conv2D(18, (1,1), name = 'coordsOut', kernel_initializer = tf.keras.initializers.GlorotUniform(seed=0), padding = 'same')(x)
	return coords


def conv_layer(x, numFilters, kernelSize, strides = 1, dilation = 1):
	"""CONV BLOCK FOR RESNET"""
	x = tf.keras.layers.Conv2D(numFilters, kernelSize, strides = strides, kernel_initializer = tf.keras.initializers.GlorotUniform(seed=0), padding = 'same', dilation_rate = dilation)(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Activation('relu')(x)
	
	return x



def pvNET(inputShape = (480, 640, 3), outVectors = True, modelName = "stvNetNew"):
	
	xIn = tf.keras.Input(inputShape, dtype = np.dtype('uint8'))
	
	x = tf.keras.layers.Lambda(lambda x: x / 255) (xIn)
	
	x = tf.keras.layers.Conv2D(64, 7, input_shape = inputShape, kernel_initializer = tf.keras.initializers.GlorotUniform(seed=0), padding = 'same')(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.Activation('relu')(x)
	
	res1 = x
	
	x = tf.keras.layers.MaxPool2D(pool_size = 3, strides = 2, padding = 'same')(x)
	
	skip = x
	
	x = conv_layer(x, 64, 3)
	x = conv_layer(x, 64, 3)
	
	x = tf.keras.layers.Add()([x, skip])
	skip = x
	
	x = conv_layer(x, 64, 3)
	x = conv_layer(x, 64, 3)
	
	x = tf.keras.layers.Add()([x, skip])
	skip = tf.keras.layers.MaxPool2D(pool_size = 2, padding = 'same')(x)
	skip = tf.pad(skip, [[0, 0], [0, 0], [0, 0], [32, 32]]) # linear projection
	res2 = x
	
	x = conv_layer(x, 128, 3, 2)
	x = conv_layer(x, 128, 3)
	
	x = tf.keras.layers.Add()([x, skip])
	skip = x
	
	x = conv_layer(x, 128, 3)
	x = conv_layer(x, 128, 3)
	
	x = tf.keras.layers.Add()([x, skip])
	skip = tf.keras.layers.MaxPool2D(pool_size = 2, padding = 'same')(x)
	skip = tf.pad(skip, [[0, 0], [0, 0], [0, 0], [64, 64]]) # linear projection
	res3 = x
	
	x = conv_layer(x, 256, 3, 2)
	x = conv_layer(x, 256, 3)
	
	x = tf.keras.layers.Add()([x, skip])
	skip = x
	
	x = conv_layer(x, 256, 3)
	x = conv_layer(x, 256, 3)
	
	x = tf.keras.layers.Add()([x, skip])
	skip = tf.pad(x, [[0, 0], [0, 0], [0, 0], [128, 128]])
	res4 = x
	
	x = conv_layer(x, 512, 3, dilation = 2)
	x = conv_layer(x, 512, 3, dilation = 2)
	
	x = tf.keras.layers.Add()([x, skip])
	skip = x
	
	x = conv_layer(x, 512, 3, dilation = 2)
	x = conv_layer(x, 512, 3, dilation = 2)
	
	x = tf.keras.layers.Add()([x, skip])
	
	x = conv_layer(x, 256, 3)
	x = conv_layer(x, 256, 3)
	
	x = tf.keras.layers.Add()([x, res4])
	x = tf.keras.layers.UpSampling2D()(x)
	
	x = conv_layer(x, 128, 3)
	
	x = tf.keras.layers.Add()([x, res3])
	x = tf.keras.layers.UpSampling2D()(x)
	
	x = conv_layer(x, 64, 3)
	
	x = tf.keras.layers.Add()([x, res2])
	x = tf.keras.layers.UpSampling2D()(x)
	
	x = tf.keras.layers.Add()([x, res1])
	
	x = conv_layer(x, 32, 3)
	
	outputs = []
	
	outputs.append(coords_output(x))
	
	return tf.keras.Model(inputs = xIn, outputs = outputs, name = modelName)


def get_models_path():
	path = os.path.join(os.getcwd(), 'models')
	os.makedirs(path, exist_ok=True)
	return path

def get_history_path():
	path = os.path.join(os.getcwd(), 'models', 'history')
	os.makedirs(path, exist_ok=True)
	return path



def trainModel(modelStruct, modelGen, modelClass = 'cat', batchSize = 2, optimizer = tf.keras.optimizers.Adam, learning_rate = 0.01, losses = None, metrics = ['accuracy'], saveModel = True, modelName = 'stvNet_weights', epochs = 1, loss_weights = None, outVectors = False, outClasses = False, dataSplit = True, altLabels = True, augmentation = True , inputShape = (480, 640, 3)):
	
	if not (outVectors):
		print("At least one of outVectors  must be set to True.")
		return
	
	model = modelStruct(inputShape = inputShape, outVectors = outVectors,modelName = modelName)
	model.compile(optimizer = optimizer(learning_rate = learning_rate), loss = losses, metrics = metrics)
	
	if dataSplit:
		trainData, validData = data_v2.get_data_split(modelClass = modelClass)
	
	logger = tf.keras.callbacks.CSVLogger(os.path.join(get_history_path(), f"{modelName}_{modelClass}_history.csv"), append = True)
	data_h, data_w = inputShape[0], inputShape[1]
	if dataSplit:
		historyLog = model.fit(modelGen(modelClass, batchSize = batchSize, masterList = trainData, altLabels = altLabels, augmentation = augmentation, height = data_h, width = data_w),
							 steps_per_epoch = math.ceil(len(trainData) / batchSize),
							 epochs = epochs,
							 validation_data = modelGen(modelClass, batchSize = batchSize, masterList = validData, altLabels = altLabels, augmentation = False, height = data_h, width = data_w),
							 validation_steps = math.ceil(len(validData) / batchSize),
							 callbacks = [logger],
							 max_queue_size = 2)
	else:
		historyLog = model.fit(modelGen(modelClass, batchSize = batchSize, altLabels = altLabels, augmentation = augmentation, height = data_h, width = data_w),
							 steps_per_epoch = 100,
							 epochs = epochs,
							 callbacks = [logger],
							 max_queue_size = 2)
	
	if saveModel:
		model_path = os.path.join(get_models_path(), f"{modelName}_{modelClass}")
		model.save_weights(model_path)
		chunkify(model_path)
		
		history_path = os.path.join(get_history_path(), f"{modelName}_{modelClass}_trainHistory")
		if not os.path.exists(history_path):
			with open(history_path, 'wb') as f:  # create model history
				pickle.dump([], f)
		with open(history_path, 'rb') as f:  # loading old history 
			histories = pickle.load(f)
		histories.append(historyLog)
		with open(history_path, 'wb') as f:  # saving the history of the model
			pickle.dump(histories, f)
	
	K.clear_session()
	K.reset_uids()
	return model

def chunkify(input_file, chunk_size = 104856576):
	"""CHUNK THE MODEL & SAVE"""
	data_file = input_file + '.data-00000-of-00001'
	with open(data_file, 'rb') as f:
		chunk_num = 0
		while True:
			chunk = f.read(chunk_size)
			if not chunk:
				break
			with open(f'{input_file}.part-{chunk_num}', 'wb') as chunk_file:
				chunk_file.write(chunk)
			chunk_num += 1

	os.remove(data_file)
	

def joinFiles(input_file, model):
	data_file = input_file + '.data-00000-of-00001'
	part_files = []
	chunk_num = 0
	while True:
		part_file = f'{input_file}.part-{chunk_num}'
		if not os.path.exists(part_file):
			break
		part_files.append(part_file)
		chunk_num += 1
	
	if not part_files:
		raise Exception(f"No .part files found for {input_file}. Expected files like {input_file}.part-0, {input_file}.part-1, etc.")
	
	print(f"Found {len(part_files)} part files: {', '.join(part_files)}")
	
	try:
		with open(data_file, 'wb') as combined:
			for part_file in part_files:
				with open(part_file, 'rb') as f:
					combined.write(f.read())
		try:
			model.load_weights(input_file)
			print("Successfully loaded model weights")
			return data_file  # Return the data file path so it can be cleaned up later
		except Exception as e:
			if os.path.exists(data_file):
				os.remove(data_file)
			raise Exception(f"Error loading model weights: {str(e)}")
	
	except Exception as e:
		if os.path.exists(data_file):
			os.remove(data_file)
		raise Exception(f"Error joining files: {str(e)}")

def load_model_weights(modelStruct, modelName, modelClass = 'cat', outVectors = False, optimizer = tf.keras.optimizers.legacy.Adam, learning_rate = 0.01, losses = None, metrics = ['accuracy'], input_shape = (480, 640, 3)):
	if not (outVectors):
		raise Exception("outVectors must be set to True.")
	model = modelStruct(outVectors = outVectors, modelName = modelName, inputShape = input_shape)
	model_path = os.path.join(get_models_path(), f"{modelName}_{modelClass}")
	data_file = joinFiles(model_path, model)  # Get the data file path
	try:
		model.compile(optimizer = optimizer(learning_rate = learning_rate), loss = losses, metrics = metrics)
	finally:
		if data_file and os.path.exists(data_file):  # Clean up the data file after compilation
			os.remove(data_file)
	return model

def evaluateModel(modelStruct, modelName, evalGen, modelClass = 'cat', outVectors = False,  batchSize = 2, optimizer = tf.keras.optimizers.Adam, learning_rate = 0.01, losses = None, metrics = ['accuracy'], samples = 100):
	model = tf.keras.models.load_model(os.path.join(get_models_path(), f"{modelName}_{modelClass}"))
	model.evaluate(evalGen(modelClass, batchSize), steps = samples // batchSize)
	

	
def evaluateModels(modelSets, batchSize = 2, dataSplit = True):
	
	for modelSet in modelSets:
		validData = (data_v2.getDataSplit(modelClass = modelSet.modelClass)[1] if dataSplit else None)
		modelEnt = modelsDict[modelSet.name]
		model = load_model_weights(modelEnt.structure, modelSet.name, modelSet.modelClass, modelEnt.outVectors, losses = modelEnt.losses, metrics = modelEnt.metrics, input_shape=modelEnt.inputShape)
		if type(model.losses) is dict:
			outKeys = list(model.losses.keys())
			if len(outKeys) == 2: # combined output
				model.evaluate(modelEnt.generator(modelSet.modelClass, batchSize = batchSize, masterList = validData, out0 = outKeys[0], out1 = outKeys[1], altLabels = modelEnt.altLabels, augmentation = False), steps = math.ceil(len(validData) / batchSize), max_queue_size = 2)
			else:
				raise Exception("Probably shouldn't be here ever..")
		else:
			model.evaluate(modelEnt.generator(modelSet.modelClass, batchSize = batchSize, masterList = validData, altLabels = modelEnt.altLabels, augmentation = False), steps = math.ceil(len(validData) / batchSize), max_queue_size = 2)

def loadHistory(modelName, modelClass = 'cat'):
	with open(os.path.join(get_history_path(), f"{modelName}_{modelClass}_trainHistory"), 'rb') as f:  # loading old history 
		histories = pickle.load(f)
		for hist in histories:
			if isinstance(hist, dict):
				# Old format - dictionary
				print("Structure: {0}\nClass: {1}\nOptimizer: {2}\nLearningRate: {3}\nLosses: {4}\nName: {5}\nEpochs: {6}\nTimestamp: {7}\nTraining History:\n".format(
					hist['struct'], hist['class'], hist['optimizer'], hist['lr'], hist['losses'], 
					hist['name'], hist['epochs'], hist['timestamp']))
				for i, epoch in enumerate(hist['history']):
					print("{0}: {1}".format(i, epoch))
				print("\nEvaluation History:\n")
				for i, epoch in enumerate(hist['evalHistory']):
					print("{0}: {1}".format(i, epoch))
			else:
				# New format - History object
				print("Training History:")
				for metric_name, values in hist.history.items():
					print(f"{metric_name}: {values}")
			print("\n")

def loadHistories(modelSets):
	for modelSet in modelSets:
		print("Loading {0}".format(modelSet.name))
		loadHistory(modelSet.name, modelSet.modelClass)
		
def plotHistories(modelSets): # display loss values over epochs using pyplot
	for modelSet in modelSets:
		plt.figure(figsize=(12, 8))
		maxLen = 0
		with open(os.path.join(get_history_path(), f"{modelSet.name}_{modelSet.modelClass}_trainHistory"), 'rb') as f: # loading old history 
			histories = pickle.load(f)
		for hist in histories:
			if isinstance(hist, dict):
				# Old format - dictionary
				if len(hist['history']) > maxLen:
					maxLen = len(hist['history'])
				plt.subplot(211)
				plt.plot([x['loss'] for x in hist['history']], label=hist['name'])
				plt.subplot(212)
				plt.plot([x[0] for x in hist['evalHistory']], label=hist['name'])
			else:
				# New format - History object
				history_len = len(hist.history['loss'])
				if history_len > maxLen:
					maxLen = history_len
				plt.subplot(211)
				plt.plot(hist.history['loss'], label=f"{modelSet.name} (training)")
				if 'val_loss' in hist.history:
					plt.subplot(212)
					plt.plot(hist.history['val_loss'], label=f"{modelSet.name} (validation)")
	
		plt.subplot(211)
		plt.ylabel("Training Loss")
		plt.xlabel("Epoch")
		plt.xticks(np.arange(0, maxLen, 1.0))
		plt.legend()
		plt.title(f"Training and Validation Loss - {modelSet.name}")
	
		plt.subplot(212)
		plt.ylabel("Validation Loss")
		plt.xlabel("Epoch")
		plt.xticks(np.arange(0, maxLen, 1.0))
		plt.legend()
	
		# Create plots directory if it doesn't exist
		plots_dir = os.path.join(os.getcwd(), 'plots')
		os.makedirs(plots_dir, exist_ok=True)
		
		# Save the plot
		plt.savefig(os.path.join(plots_dir, f"{modelSet.name}_{modelSet.modelClass}_loss_history.png"))
		plt.close()
		
# modelDictVal(pvNet, data.coordsTrainingGenerator, 
# 			 tf.keras.losses.Huber(), True, epochs = 10, 
# 			 lr = 0.001, metrics = ['mae', 'mse'], altLabels = False, 
# 			 augmentation = False, inputShape = (1440,1920, 3)),
# }
modelsDict = {
	'stvNet_new_coords_LINEMOD' : modelDictVal(pvNET, data_v2.coordsTrainingGenerator, tf.keras.losses.Huber(), True, epochs = 10, lr = 0.001, metrics = ['mae', 'mse'], altLabels = False, augmentation = False, inputShape = (480, 640, 3)),
	'stvNet_new_coords_HANDAL' : modelDictVal(pvNET, data_v2.coordsTrainingGenerator, tf.keras.losses.Huber(), True, epochs = 10, lr = 0.001, metrics = ['mae', 'mse'], altLabels = False, augmentation = False, inputShape = (1440,1920, 3)),
}

if __name__ == "__main__" :
	model_info = modelsDict['stvNet_new_coords_LINEMOD']
	input_shape = model_info.inputShape if hasattr(model_info, 'inputShape') else None
	model_set = modelSet('stvNet_new_coords_LINEMOD', modelClass='cat', inputShape=input_shape)
	print("Training {0}".format(model_set.name))
	model = modelsDict[model_set.name]
	trainModel(model.structure, model.generator, modelClass = modelSet.modelClass, epochs = model.epochs, losses = model.losses, modelName = modelSet.name, outClasses = model.outClasses, outVectors = model.outVectors, learning_rate = model.lr, metrics = model.metrics, altLabels = model.altLabels, augmentation = model.augmentation, inputShape=(480, 640, 3))
	evaluateModels([model_set])
	loadHistories([model_set])
	plotHistories([model_set])
	
