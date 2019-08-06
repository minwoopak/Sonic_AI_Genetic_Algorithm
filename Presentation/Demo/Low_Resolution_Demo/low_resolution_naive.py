import numpy as np
import retro
import neat
import cv2
import pickle
# import visualize

# Create the environment & Import the game and state
# while recording
env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')

# Define the function that calculates the fitness of each genome in the whole population
def eval_genomes(genomes, config):
    # imgarray = []
    # xpos_end = 0

# Loops through each genome in the population
    for genome_id, genome in genomes:
        ob = env.reset()
        # ac = env.action_space.sample()

        iny, inx, inc = env.observation_space.shape

# Scale down the input(screen resolution) by 1/8
        inx = int(inx/8)
        iny = int(iny/8)

# Create the network (Here, we use RNN)
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        xpos = 0
        xpos_max = 0
        done = False

        cv2.namedWindow("main", cv2.WINDOW_NORMAL) # Alternate Rendering

# For each step until the game is done,
# Flatten the screen image and feed it to the network to get the action values(button press)
        while not done:
            frame += 1

            scaledimg = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY) # Alternate Rendering
            scaledimg = cv2.resize(scaledimg, (inx, iny)) # Alternate Rendering

            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx,iny))

            cv2.imshow('main', scaledimg) # Alternate Rendering
            cv2.waitKey(1) # Alternate Rendering

            # env.render()

            imgarray = np.ndarray.flatten(ob)
            nnOutput = net.activate(imgarray)
            ob, rew, done, info = env.step(nnOutput)

# Define the reward function
            xpos = info['x']
            # xpos_screen = info['screen_x']
            xpos_end = info['screen_x_end']

            if xpos > xpos_max:
                fitness_current += 1
                xpos_max = xpos

# When it reaches the end of the map, terminate
            if xpos == 9550:
                fitness_current += 100000
                done = True

# If no progress is made for 250 frames, terminate
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            if done or counter == 250:
                done = True
                print("Genome ID: ", genome_id, "Reward Acquired: ", fitness_current)
            genome.fitness = fitness_current

# Import network setting
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

# Create population
p = neat.Population(config)

# Print out training information
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

# Save the progress every 10 iteration
p.add_reporter(neat.Checkpointer(10))
p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-6') # Restore the progress and train from there

# Runs the eval_genomes and save the result to winner variable once done
winner = p.run(eval_genomes)

# Save the winner genome to a pickle file
with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)

# # Visualize network
# visualize.draw_net(config, winner, True)
# visualize.plot_stats(stats, ylog=False, view=True)
# visualize.plot_species(stats, view=True)
