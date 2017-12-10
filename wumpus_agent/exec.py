'''
Author : Somshubra Majumdar
Date : 25-Jan-17

Tests the wumpus world environment 100 times (default, can be changed),
wherein each test creates 10,000 games. Then provides statistics of the games,
including mean, standard deviation, max and min score achieved in 100 runs of
10,000 games.

'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import asyncio
import sys
from asyncio.subprocess import PIPE, STDOUT

import argparse
import os
import numpy as np

from memory import ExperienceReplay
from model import build_model

assert os.path.exists("WorldApplication.class"), "WorldApplication.class not found. Aborting. Read instructions to use stats.py"

parser = argparse.ArgumentParser('Wumpus World statistics')
parser.add_argument('-i', default=10000, type=int, help='Number of iterations to test Wumpus world')

parser.add_argument('-d', default='4', type=str, help='Sets the dimensions of the Wumpus World to be dimension x dimension. Default: 4 (a 4x4 world)')
parser.add_argument('-s', default='50', type=str, help='Sets the maximum number of time steps. Default: 50')
parser.add_argument('-t', default='1', type=str, help='Sets the number of trials. Default: 10000')
parser.add_argument('-a', default="false", type=str, help="Sets whether the agent's location and orientation is randomly generated. Default: true")
parser.add_argument('-r', default=-1, type=str, help='Sets the seed for the random Wumpus World generator. Default: (random integer)')
parser.add_argument('-f', default='wumpus_out.txt', type=str, help='sets the filename for the output file (containing the terminal output). Default: wumpus_out.txt')
parser.add_argument('-n', default="false", type=str, help="sets whether the agent's GO_FORWARD action behavior is non-deterministic. Default: true")
parser.add_argument('-p', default="false", type=str, help='sets whether the terminal will print out the environment along with the actions. When off, will still print final score Default: false')

args = parser.parse_args()

iterations = args.i # number of iterations to check

param_args = ["java", "WorldApplication", "-d", args.d, "-s", args.s, "-t", args.t, "-a", args.a, "-f", args.f, "-n",
              args.n, "-p", args.p]

if args.r != -1:
    param_args.append("-r")
    param_args.append(args.r)

print("Testing Wumpus world environment %d times" % iterations, "\n", "*" * 60, "\n")

out_path = r"wumpus_out.txt"

scores = []
FNULL = open(os.devnull, 'w') # Prevent java code output on screen

''' Constants '''
nb_actions = 6
memory_size = 100
observe = 0
batch_size = 50

epsilon = (1.0, 0.1)
epsilon_rate = 0.5

delta =  ((epsilon[0] - epsilon[1]) / (iterations * epsilon_rate))
final_epsilon = epsilon[1]
epsilon = epsilon[0]

win_count = 0

''' Memory and Model '''
memory = ExperienceReplay(memory_size)
model = build_model()

''' Agent Code '''
initial_state = ['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.33', '1.0', '1.0', '1.0', '1.0', '0.0', '0.0', '1.0', '1.0', '1.0', '0.0', '1.0', '1.0', '0.0', '0.0', '0.0', '1.0', '1.0', '1.0', '0.0', '1.0', '1.0', '0.0', '1.0', '1.0', '0.0', '1.0', '1.0', '0.0', '1.0', '1.0', '0.0', '1.0', '1.0', '0.0', '1.0', '1.0', '0.0', '1.0', '1.0', '0.0', '1.0', '1.0', '0.0', '1.0', '1.0', '0.0', '1.0', '1.0', '0.0']

if sys.platform == "win32":
    loop = asyncio.ProactorEventLoop() # for subprocess' pipes on Windows
    asyncio.set_event_loop(loop)
else:
    loop = asyncio.get_event_loop()

async def run_loop():
    global epsilon, win_count

    print("Beginning new game iteration : ", step + 1)
    loss = 0.
    game_over = 0
    final_score = 0
    S = np.asarray(initial_state)[np.newaxis]

    # p = Popen(param_args, stdin=PIPE, stdout=PIPE)

    p = await asyncio.create_subprocess_exec(*param_args, stdin=PIPE, stdout=PIPE)

    for _ in range(10):  # skip first 10 lines
        await asyncio.wait_for(p.stdout.readline(), timeout=2)

    curr_score = 0

    for i in range(50):  # Play for 50 game steps
        if np.random.random() < epsilon or i < observe:
            a = int(np.random.randint(nb_actions))
        else:
            a = model.predict_classes(S)

        action = str(a + 1) + "\n"  # ArgMax returns in range of [0-5], whereas actions are [1-6]

        if "[" in action:
            action = action[1:2]

        # print("Action : ", action)

        p.stdin.write(bytes(action, encoding='utf-8'))  # Perform action
        #try:
        #    p.stdin.flush()
        #except OSError:
        #    print('**warning** : Failed to flush')

        # result = str(p.stdout.readline(), 'utf-8') # Get new state
        # curr_score = str(p.stdout.readline(), 'utf-8') # Get new reward

        try:
            result = await asyncio.wait_for(p.stdout.readline(), timeout=1)
            result = result[:-4]
        except asyncio.TimeoutError:
            print("Timeout error, breaking.")
            break

        try:
            curr_score = await asyncio.wait_for(p.stdout.readline(), timeout=1)
        except asyncio.TimeoutError:
            print("Timeout error, breaking.")
            break

        result = str(result, 'utf-8')

        ''' Game End Criteria '''
        try:
            curr_score = float(curr_score)
        except ValueError:
            curr_score = str(curr_score, 'utf-8')
            game_over = 1

        if 'Average Score: ' in result:
            result = result.replace('Average Score: ', '')
            final_score = float(result)
            game_over = 1

        if not isinstance(curr_score, float):
            if 'Average Score: ' in curr_score:
                curr_score = curr_score.replace('Average Score: ', '')
                final_score = float(curr_score)
                game_over = 1

        if result == 'Finished.':
            game_over = 1

        ''' Updates '''

        S_prime = result.split(' ')
        S_prime[-1] = S_prime[-1].replace('\r\n', '')
        S_prime = np.asarray(S_prime)[np.newaxis]

        r = curr_score

        memory.remember(S, a, r, S_prime, game_over)
        S = S_prime

        if i >= observe:
            batch = memory.get_batch(model=model, batch_size=batch_size, gamma=0.9)
            if batch:
                inputs, targets = batch
                loss += float(model.train_on_batch(inputs, targets))

        model.save_weights('dnn.h5', overwrite=True)

        if game_over:
            break

    print('Final Score :', final_score)
    if isinstance(final_score, float):
        if final_score > 0:  # Assume won if score is this high
            win_count += 1

    if epsilon > final_epsilon and step >= observe:
        epsilon -= delta

    print("Epoch {:03d}/{:03d} | Loss {:.4f} | Epsilon {:.2f} | Win count {}".format(step + 1, iterations, loss, epsilon,
                                                                                   win_count))

for step in range(iterations):
    loop.run_until_complete(run_loop())

loop.close()
print('Finished.')