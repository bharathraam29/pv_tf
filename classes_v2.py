class modelSet:
	def __init__(self, modelName, modelClass = 'cat', inputShape = None):
		self.name = modelName
		self.modelClass = modelClass
		self.inputShape = inputShape


class modelDictVal:
	def __init__(self, structure, generator, losses, outVectors, epochs = 3, lr = 0.01, metrics = ['accuracy'], outVecName = None, outClassName = None, altLabels = False, augmentation = True, inputShape = (480, 640, 3)):
		self.structure = structure
		self.generator = generator
		self.losses = losses
		self.outVectors = outVectors
		self.epochs = epochs
		self.metrics = metrics
		self.lr = lr
		self.outVecName = outVecName
		self.outClassName = outClassName
		self.augmentation = augmentation
		self.inputShape = inputShape
		

# modelDictVal(pvNet, data.coordsTrainingGenerator, 
# 			 tf.keras.losses.Huber(), True, epochs = 10, 
# 			 lr = 0.001, metrics = ['mae', 'mse'], altLabels = False, 
# 			 augmentation = False, inputShape = (1440,1920, 3)),
# }