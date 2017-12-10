from .memory import ExperienceReplay
import numpy as np
import os

initial_state = ['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.33', '1.0', '1.0', '1.0', '1.0', '0.0', '0.0', '1.0', '1.0', '1.0', '0.0', '1.0', '1.0', '0.0', '0.0', '0.0', '1.0', '1.0', '1.0', '0.0', '1.0', '1.0', '0.0', '1.0', '1.0', '0.0', '1.0', '1.0', '0.0', '1.0', '1.0', '0.0', '1.0', '1.0', '0.0', '1.0', '1.0', '0.0', '1.0', '1.0', '0.0', '1.0', '1.0', '0.0', '1.0', '1.0', '0.0', '1.0', '1.0', '0.0']

class Agent:

	def __init__(self, model, memory=None, memory_size=1000):
		assert len(model.output_shape) == 2, "Model's output shape should be (nb_samples, nb_actions)."

		if memory:
			self.memory = memory
		else:
			self.memory = ExperienceReplay(memory_size)

		self.model = model
		self.frames = None

	@property
	def memory_size(self):
		return self.memory.memory_size

	@memory_size.setter
	def memory_size(self, value):
		self.memory.memory_size = value

	def reset_memory(self):
		self.memory.reset_memory()

	def get_game_data(self, game_data):
		input_data = game_data[0]
		reward = game_data[1]
		return np.expand_dims(input_data, 0), reward

	def clear_frames(self):
		self.frames = None

	def train(self, game_data, nb_actions, nb_epoch=1000, batch_size=50, gamma=0.9, epsilon=[1., .1], epsilon_rate=0.5, reset_memory=False, observe=0, checkpoint=None):
		if type(epsilon) in {tuple, list}:
			delta =  ((epsilon[0] - epsilon[1]) / (nb_epoch * epsilon_rate))
			final_epsilon = epsilon[1]
			epsilon = epsilon[0]
		else:
			final_epsilon = epsilon

		model = self.model
		nb_actions = model.output_shape[-1]
		win_count = 0

		for epoch in range(nb_epoch):
			loss = 0.
			self.clear_frames()

			if reset_memory:
				self.reset_memory()

			game_over = False
			S, r = self.get_game_data(game_data)

			while not game_over:
				if np.random.random() < epsilon or epoch < observe:
					a = int(np.random.randint(nb_actions))
				else:
					q = model.predict(S)
					a = int(np.argmax(q[0]))

				#game.play(a)

				r = game.get_score()
				S_prime = self.get_game_data(game)
				game_over = game.is_over()
				transition = [S, a, r, S_prime, game_over]
				self.memory.remember(*transition)
				S = S_prime
				if epoch >= observe:
					batch = self.memory.get_batch(model=model, batch_size=batch_size, gamma=gamma)
					if batch:
						inputs, targets = batch
						loss += float(model.train_on_batch(inputs, targets))
				if checkpoint and ((epoch + 1 - observe) % checkpoint == 0 or epoch + 1 == nb_epoch):
					model.save_weights('weights.dat')
			if game.is_won():
				win_count += 1
			if epsilon > final_epsilon and epoch >= observe:
				epsilon -= delta
			print("Epoch {:03d}/{:03d} | Loss {:.4f} | Epsilon {:.2f} | Win count {}".format(epoch + 1, nb_epoch, loss, epsilon, win_count))

	def play(self, game, nb_epoch=10, epsilon=0., visualize=True):
		self.check_game_compatibility(game)
		model = self.model
		win_count = 0
		frames = []
		for epoch in range(nb_epoch):
			game.reset()
			self.clear_frames()
			S = self.get_game_data(game)
			if visualize:
				frames.append(game.draw())
			game_over = False
			while not game_over:
				if np.random.rand() < epsilon:
					print("random")
					action = int(np.random.randint(0, game.nb_actions))
				else:
					q = model.predict(S)[0]
					possible_actions = game.get_possible_actions()
					q = [q[i] for i in possible_actions]
					action = possible_actions[np.argmax(q)]
				game.play(action)
				S = self.get_game_data(game)
				if visualize:
					frames.append(game.draw())
				game_over = game.is_over()
			if game.is_won():
				win_count += 1
		print("Accuracy {} %".format(100. * win_count / nb_epoch))

