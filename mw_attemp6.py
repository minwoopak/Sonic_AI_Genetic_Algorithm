import retro
import os
import numpy as np
import cv2
import neat
import visualize
import pickle, sys
from datetime import datetime

def show_input(ob, iny, inx):
	scaledimg = cv2.cvtColor(ob, cv2.COLOR_BGR2RGB)
	scaledimg = cv2.resize(scaledimg, (iny, inx))
	cv2.imshow('main', scaledimg)
	cv2.waitKey(1)


def eval_genomes(genomes, config):
	for genome_id, genome in genomes:
		ob = env.reset()
		ac = env.action_space.sample()

		# iny, inx, inc = env.observation_space.shape

		inx = int(ob.shape[1]/8)
		iny = int(ob.shape[0]/8)
		# print('inx:',inx,'iny',iny)

		net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

		current_max_fitness = 0
		fitness_current = 0
		# frame = 0
		counter = 0
		# xpos = 0
		xpos_max = 0
		ring_count = 0
		current_score = 0
		height_max = 0
		height_current = 0

		done = False
		# cv2.namedWindow("main", cv2.WINDOW_NORMAL)

		while not done:
			env.render()
			# frame += 1


			ob = cv2.resize(ob, (inx, iny))
			ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
			ob = np.reshape(ob, (inx, iny))

			imgarray = ob.flatten()
			#show_input(ob, iny, inx)
			nnOutput = net.activate(imgarray)


			ob, rew, done, info = env.step(nnOutput)

			xpos = info['x']
			xpos_end = info['screen_x_end']
			num_rings = info['rings']
			score = info['score']
			height = info['screen_y']

			if xpos > xpos_max:
				fitness_current += 1
				xpos_max = xpos

			# how to give rewards for going left a little too?
                        # how to give rewards for going higher? screen y position?
                        # print fitness for all frames and see if - rewards are being applied
                        # how to give rewards for speed
                        # if count > 200 : xpos decrease -> reward

			# if height > height_max:
			# 	fitness_current += 20
			# 	height_max = height 
			# 	print('height reward!: ', height-height_current, 'height now: ', height)
			# height_current = height

                        # if ring_status < 0: penalize more when negative ring reward
			ring_status = 30*(num_rings - ring_count)
			fitness_current += ring_status
                        ###############################
			if ring_status != 0:
                            print('ring changed!', ring_status, 'fitness: ', fitness_current)
                        ###############################
			ring_count = num_rings

			score_status = 30*(score - current_score)
			fitness_current += score_status
                        ###############################
			if score_status != 0:
                            print('score changed!', score_status, 'fitness: ', fitness_current)
                        ###############################
			current_score = score

			if xpos >= xpos_end and xpos > 600:
				fitness_current += 100000
				done = True

			if fitness_current > current_max_fitness:
				current_max_fitness = fitness_current
				counter = 0
			else:
				counter += 1

			xpos_checkpoint = xpos
			if counter >= 200 and (xpos_checkpoint-xpos) <= -20:
				fitness_current += 200

			if done or counter == 250:
				done = True
				print(genome_id, fitness_current)

			genome.fitness = fitness_current
			# print('fitness: ', fitness_current)


env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')
while True:
	timestamp = datetime.now().strftime("%d_%B_%Y_%I_%M%p")
	# sys.stdout = open('results/winner_{}.txt'.format(timestamp), 'w')

	config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
		neat.DefaultSpeciesSet, neat.DefaultStagnation,
		'config-feedforward')

	p = neat.Population(config)

	p.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	p.add_reporter(stats)
	p.add_reporter(neat.Checkpointer(10))

	winner = p.run(eval_genomes, n=30)
	print('\nBest genome:\n{!s}'.format(winner))

	# node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
	visualize.draw_net(config, winner, True)#, node_names=node_names)
	visualize.plot_stats(stats, ylog=False, view=True)
	visualize.plot_species(stats, view=True)

	p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-11')

	with open('results/winner_{}.pkl'.format(timestamp), 'wb') as output:
		pickle.dump(winner, output, 1)

	env.reset()
	sys.stdout.close()
