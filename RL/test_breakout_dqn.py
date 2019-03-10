from breakout_dqn import *
import time

EPISODES=50000

if __name__ == "__main__":
    env = gym.make('BreakoutDeterministic-v4')
    agent = DQNAgent(action_size=3, train=False)
    agent.load_model('./save_model/breakout_dqn_pretrained.h5')

    for e in range(EPISODES):
        done = False
        dead = False

        step, score, start_life = 0, 0, 5
        observe = env.reset()

        for _ in range(random.randint(1, agent.no_op_steps)):
            observe, _, _, _ = env.step(1)

        state = pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:
            env.render()
            time.sleep(0.05)
            step += 1

            action = agent.get_action(history)

            real_action = action + 1

            if dead:
                real_action = 1
                dead = False

            observe, reward, done, info = env.step(real_action)

            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            score += reward

            history = next_history

            if done:
                print("episode:", e, "  score:", score)