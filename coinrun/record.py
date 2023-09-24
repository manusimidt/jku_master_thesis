import pygame
import gym
import cv2
import numpy as np
import os


SEEDS = list(range(120, 130))
SUCCESSFULL_EPISODES_PER_SEED = 20

def _get_episode_nr(target_folder: str, seed: int) -> int:
    """
    Each episode is named after the seed of the env and an appended increment.
    This function looks into the target_folder and returns the next increment
    """
    return len([filename for filename in os.listdir(target_folder) if filename.startswith(str(seed) + '-')])


for seed in SEEDS:
    env  = gym.make('procgen:procgen-coinrun-v0', start_level=seed, paint_vel_info=True, num_levels=1, 
                    distribution_mode="easy")
    

    # Initialize Pygame
    pygame.init()



    # Constants
    WINDOW_WIDTH = 640
    WINDOW_HEIGHT = 640

    # Initialize the screen
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Image Viewer")


    # Create a Pygame clock object to control frame rate
    clock = pygame.time.Clock()

    # Main game loop
    running = True
    
    successful_runs =0

    while running:

        done = False
        left, right, up, down = False, True, False, False
        states, actions, rewards = [], [], []

        state = env.reset() 
        image = cv2.resize(state, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST_EXACT)
        current_image = pygame.surfarray.make_surface(np.swapaxes(state, 0, 1))
        print("START!")
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYUP or event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                        right = event.type == pygame.KEYDOWN
                    elif event.key == pygame.K_LEFT:
                        left = event.type == pygame.KEYDOWN
                    elif event.key == pygame.K_UP:
                        up = event.type == pygame.KEYDOWN

            if right and up:
                action = 8
            elif left and up:
                action = 2
            elif right:
                action = 6
            elif left:
                action = 0
            elif up: 
                action = 5
            else:
                action = 3

            next_state, r, done, info = env.step(action) 
            states.append(state)
            actions.append(action)
            rewards.append(r)

            state = next_state

            image = cv2.resize(state, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST_EXACT)
            current_image = pygame.surfarray.make_surface(np.swapaxes(image, 0, 1))

            # Clear the screen
            screen.fill((0, 0, 0))

            # Display the current image
            screen.blit(current_image, (0, 0))
            pygame.display.flip()
            clock.tick(15)

            if done and r == 10:
                successful_runs += 1
                print(f"Finished episode successfuly! {SUCCESSFULL_EPISODES_PER_SEED-successful_runs} left")

                # save the episode!
                target_folder = f'./coinrun/dataset/seeds-{min(SEEDS)}-{max(SEEDS)}'
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                file_path = f'{target_folder}/{info["level_seed"]}-{_get_episode_nr(target_folder, info["level_seed"])}.npz'
                np.savez(file_path, state=np.array(states), action=np.array(actions), reward=np.array(rewards))

                left, right, up, down = False, False, False, False
                states, actions, rewards = [], [], []
                if successful_runs == SUCCESSFULL_EPISODES_PER_SEED:
                    running=False
                



    # Quit Pygame
    pygame.quit()
