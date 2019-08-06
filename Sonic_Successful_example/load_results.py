import retro
import neat
import pickle as pkl
import cv2
import numpy as np
import argparse
import time


def show_input(ob, iny, inx):
	scaledimg = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
	scaledimg = cv2.resize(scaledimg, (iny, inx))
	cv2.imshow('main', scaledimg)
	cv2.waitKey(1)


#Parsing dos argumento da linha de comando
parser = argparse.ArgumentParser(description='Execucao de resultado')
parser.add_argument('-input, -i', dest='input', action='store',
                    required=True,help='Winner structure')
args = parser.parse_args()


with open(args.input, 'rb') as file:
	winner = pkl.load(file)

print (winner)

print('\nOutput:')


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
	neat.DefaultSpeciesSet, neat.DefaultStagnation,
	'config-feedforward')



#env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')
env = retro.make('SonicTheHedgehog2-Genesis', 'EmeraldHillZone.Act1')




ob = env.reset()
inx, iny, inc = env.observation_space.shape

inx = int(inx/8)
iny = int(iny/8)

net = neat.nn.recurrent.RecurrentNetwork.create(winner, config)


#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#out = cv2.VideoWriter('results/output.avi',fourcc, 20.0, (640,480))


frame = 0
done = False
time.sleep(5)
while not done:
	env.render()
	#show_input(ob, iny, inx)
	if frame == 0:
		time.sleep(10)
	frame += 1
	#cv2.imshow(args.input, ob[:,:,::-1])
	#cv2.waitKey(1)
	#out.write(ob[:,:,::-1])


	ob = cv2.resize(ob, (inx, iny))
	ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
	ob = np.reshape(ob, (inx, iny))

	imgarray = ob.flatten()

	nnOutput = net.activate(imgarray)

	ob, rew, done, info = env.step(nnOutput)

#out.release()
#cv2.destroyAllWindows()


'''
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
for xi, xo in zip(xor_inputs, xor_outputs):
    output = winner_net.activate(xi)
    print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
visualize.draw_net(config, winner, True, node_names=node_names)
visualize.plot_stats(stats, ylog=False, view=True)
visualize.plot_species(stats, view=True)

p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
p.run(eval_genomes, 10)
'''