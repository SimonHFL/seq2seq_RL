class Experiment():

	def __init__(self, hyperparams, train_method, start_checkpoint, data, tokenization, name = None, max_hours = None):
		self.hyperparams = hyperparams
		self.train_method = train_method
		self.start_checkpoint = start_checkpoint
		self.data = data
		self.tokenization = tokenization
		self.name = name
		self.max_hours = max_hours