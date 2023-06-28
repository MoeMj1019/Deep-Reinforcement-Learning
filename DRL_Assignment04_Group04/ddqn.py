import gymnasium as gym
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime 
from tqdm import tqdm 
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Disables TensorFlow logging messages
# Disable TensorFlow debug info
tf.get_logger().setLevel('ERROR')

# import logging
# logging.getLogger("tensorflow").setLevel(logging.WARNING)
# tf.debugging.experimental.disable_dump_debug_info()



class ExperienceReplayBuffer():
    def __init__(self,max_size:int, env_name:str, parallel_game_unrolls:int,unroll_steps:int, obv_preprocessing_function:callable):
        self.max_size = max_size
        self.env_name = env_name
        self.parallel_game_unrolls = parallel_game_unrolls
        self.unroll_steps = unroll_steps
        self.obv_preprocessing_function = obv_preprocessing_function

        self.envs = gym.vector.make(env_name, num_envs=parallel_game_unrolls)
        self.num_possible_actions = self.envs.single_action_space.n
        self.current_states, _ = self.envs.reset() 

        self.data = []

    def fill_with_samples(self,dqn_network,epsilon:float):
        states_list = []
        actions_list = []
        rewards_list = []
        subsequent_states_list = []
        terminal_list = []

        for i in range(self.unroll_steps):
            actions = self.sample_epsilon_greedy(dqn_network,epsilon)
            # print(actions)
            next_observations, rewards, terminateds, _, _ = self.envs.step(actions) # in vectorized envs
            # put the sample into the buffer ( s,a,r,s',t )
            states_list.append(self.current_states)
            actions_list.append(actions)
            rewards_list.append(rewards)
            subsequent_states_list.append(next_observations)
            terminal_list.append(terminateds)
            
            # update the current state
            self.current_states = next_observations

        def data_generator():
            for s_batch, a_batch, r_batch, s_prime_batch, t_batch in zip(states_list, actions_list, rewards_list, subsequent_states_list, terminal_list):
                for game_idx in range(self.parallel_game_unrolls):
                    s = s_batch[game_idx, :, :, :]
                    a = a_batch[game_idx]
                    r = r_batch[game_idx]
                    s_prime = s_prime_batch[game_idx, :, :, :]
                    t = t_batch[game_idx]
                    # self.buffer.append((s, a, r, s_prime, t))
                    yield (s, a, r, s_prime, t)

        dataset_tensor_specs = (tf.TensorSpec(shape=(210, 160, 3), dtype=tf.uint8, name='s'),
                                tf.TensorSpec(shape=(), dtype=tf.int32, name='a'),
                                tf.TensorSpec(shape=(), dtype=tf.float32, name='r'),
                                tf.TensorSpec(shape=(210, 160, 3), dtype=tf.uint8, name='s_prime'),
                                tf.TensorSpec(shape=(), dtype=tf.bool, name='t'))
        
        sample_dataset_tf = tf.data.Dataset.from_generator(data_generator, output_signature=dataset_tensor_specs)

        # not optimal but showcase the steps clearly (it's applied 2-3 times )
        sample_dataset_tf = sample_dataset_tf.map(lambda s, a, r, s_prime, t: 
                        (self.obv_preprocessing_function(s), a, r, self.obv_preprocessing_function(s_prime), t))
        sample_dataset_tf = sample_dataset_tf.cache().shuffle(buffer_size=self.unroll_steps*self.parallel_game_unrolls,
                                                              reshuffle_each_iteration=True)

        # make sure that cache is applied on all the datapoints
        for datapoint in sample_dataset_tf:
            continue

        self.data.append(sample_dataset_tf)
        datapoints_in_buffer = len(self.data) * self.unroll_steps * self.parallel_game_unrolls
        if datapoints_in_buffer > self.max_size:
            self.data.pop(0)

    def create_dataset_tf(self):
        erp_dataset = tf.data.Dataset.sample_from_datasets(self.data,weights= [1/float(len(self.data)) for _ in self.data],
                                                            stop_on_empty_dataset=False)
        return erp_dataset

    def sample_epsilon_greedy(self,dqn, epsilon:float):
        observations = self.obv_preprocessing_function(self.current_states)

        q_values = dqn(observations) # tensor of type tf.float32 shape (parallel_game_unrolls, num_actions)
        gready_actions = tf.argmax(q_values, axis=1) # tensor of type tf.int64 shape (parallel_game_unrolls,)
        random_actions = tf.random.uniform(shape=(self.parallel_game_unrolls,), minval=0,
                                            maxval=self.num_possible_actions, dtype=tf.int64) # tensor of type tf.int64 shape (parallel_game_unrolls,)
        epsilon_sampling = tf.random.uniform(shape=(self.parallel_game_unrolls,), minval=0, maxval=1, dtype=tf.float32) > epsilon # tensor of type tf.bool shape (parallel_game_unrolls,)
        actions = tf.where(epsilon_sampling, gready_actions, random_actions).numpy() # tensor of type tf.int64 shape (parallel_game_unrolls,)
        return actions


# ---------------------------- network ----------------------------
# -----------------------------------------------------------------
# ----------------------------------------------------------------

INPUT_RESHAPED_SIZE = (84, 84, 3)

def obv_preprocessing_function(obsevation):
    # obsevation = tf.image.resize(obsevation, size=INPUT_RESHAPED_SIZE[:2])
    obsevation = tf.image.resize(obsevation, size=(84, 84))
    obsevation = tf.cast(obsevation, tf.float32) / 128 -1.0
    return obsevation

def create_dqn_model(num_possible_actions:int):
    # create the input layer
    # input_layer = tf.keras.Input(shape=INPUT_RESHAPED_SIZE ,dtype=tf.float32 ,name="input_layer")
    input_layer = tf.keras.Input(shape=(84, 84, 3) ,dtype=tf.float32 ,name="input_layer")

    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', padding='same', name='h_1')(input_layer)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', padding='same', name='h_2')(x) + x # residual connection
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', padding='same', name='h_3')(x) + x 
    x = tf.keras.layers.MaxPool2D(pool_size=2, name='pool_layer_1')(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', name='h_4')(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', name='h_5')(x) + x # residual connection
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', name='h_6')(x) + x
    x = tf.keras.layers.MaxPool2D(pool_size=2, name='pool_layer_2')(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', name='h_7')(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', name='h_8')(x) + x # residual connection
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', name='h_9')(x) + x
    x = tf.keras.layers.MaxPool2D(pool_size=2, name='pool_layer_3')(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', name='h_10')(x) + x # residual connection
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', name='h_11')(x) + x 
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', name='h_12')(x) + x
    x = tf.keras.layers.GlobalAveragePooling2D(name='pool_layer_4_global')(x)
    x = tf.keras.layers.Dense(units=64, activation='relu', name='h_13_dense')(x) + x # residual connection
    x = tf.keras.layers.Dense(units=num_possible_actions, activation='linear', name='output_layer')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=x)
    return model

def train_dqn(dqn_network:tf.keras.Model,dqn_target_network:tf.keras.Model,
              dataset,optimizer,gamma,max_training_steps:int,batch_size:int):
    dataset = dataset.batch(batch_size).prefetch(4)

    @tf.function
    def training_step(q_target, obvs, actions):
        with tf.GradientTape() as tape:
            q_predictions_all_actions = dqn_network(obvs) # tensor of type tf.float32 shape (batch_size "parallel_game_unrolls", num_actions)
            q_predictions = tf.gather(q_predictions_all_actions, actions, batch_dims=1) # axis=1 ? tensor of type tf.float32 shape (batch_size "parallel_game_unrolls",)
            loss = tf.reduce_mean(tf.square(q_predictions - q_target))
        gradients = tape.gradient(loss,dqn_network.trainable_variables)
        optimizer.apply_gradients(zip(gradients,dqn_network.trainable_variables))
        return loss

    losses = []
    q_values = []
    for i, state_transition in enumerate(dataset):
        # train on data
        state, action, reward, subsequent_state, terminated = state_transition
        # calculate q_target
        q_vals_target = dqn_target_network(subsequent_state)
        q_values.append(q_vals_target.numpy())

        q_vals_dqn = dqn_network(subsequent_state) # double dqn
        max_q_value_idx = tf.argmax(q_vals_dqn, axis=1) # double dqn
        # max_q_value = tf.reduce_max(q_vals_target, axis=1)
        max_q_value = tf.gather(q_vals_target, max_q_value_idx, batch_dims=1) # double dqn

        use_subsequent_state = tf.where(terminated, tf.zeros_like(max_q_value,dtype=tf.float32), tf.ones_like(max_q_value,dtype=tf.float32))

        q_target = reward + ( gamma * max_q_value * use_subsequent_state) # if in terminal state, q_target = reward
        loss = training_step(q_target,obvs=state, actions=action)
        losses.append(loss.numpy())

        if i > max_training_steps:
            break

    return np.mean(losses) , np.mean(q_values)

def test_q_network(dqn_network, env_name:str, num_parallel_tests:int, max_steps_per_game:int, gamma:float,test_epsilon:float = 0.05):
    envs = gym.vector.make(env_name, num_envs=num_parallel_tests) #  asynchronous=True ?
    num_possible_actions = envs.single_action_space.n
    states, _ = envs.reset()
    
    time_step = 0
    done = False
    # eposides_finished is to keep track of how many games have finished
    episodes_finished = np.zeros(num_parallel_tests, dtype=bool)
    returns = np.zeros(num_parallel_tests)

    test_steps = 0
    while not done and time_step < max_steps_per_game:
        states = obv_preprocessing_function(states)
        q_values = dqn_network(states)
        greedy_actions = tf.argmax(q_values, axis=1)
        random_actions = tf.random.uniform(shape=(num_parallel_tests,), minval=0,
                                           maxval=num_possible_actions, dtype=tf.int64)
        epsilon_sampling = tf.random.uniform(shape=(num_parallel_tests,), minval=0, maxval=1, dtype=tf.float32) > test_epsilon
        actions = tf.where(epsilon_sampling, greedy_actions, random_actions).numpy()

        states, rewards, terminateds, _, _ = envs.step(actions)
        
        episodes_finished = np.logical_or(episodes_finished,terminateds)
        # update returns only for games that are not finished yet
        returns += ( (gamma**time_step) * rewards ) * np.logical_not(episodes_finished).astype(np.float32)

        time_step += 1
        done = np.all(episodes_finished)
    
        if test_steps % 100 == 0:
            print("====================================")
            print(f"test_steps: {test_steps}, returns: {returns} {np.sum(episodes_finished) / num_parallel_tests} , {terminateds.shape} , {episodes_finished.shape}")
        test_steps += 1

    return np.mean(returns)


# ----------------- ddqn_algorithm -----------------
# -------------------------------------------------
# -------------------------------------------------

def visualise_results(results:pd.DataFrame , step = None):
    columns = results.columns.values
    columns_count = len(columns)
    fig, axes = plt.subplots(3,1,sharex=True,figsize=(10,5*columns_count))
    fig.suptitle(f'DQN results - {step} steps')
    fig.tight_layout(pad=3.0)
    for idx ,key in enumerate(columns):
        sns.lineplot(data=results, x=results.index, y=key, ax=axes[idx])
        axes[idx].grid()

    timestring = datetime.datetime.now().strftime("%m_%d-%H_%M_%S")
    plt.savefig(f"./results/dqn_results_{timestring}_{step}.png")

def polyak_averaging_weights(source_network:tf.keras.Model, target_network:tf.keras.Model, polyak_averaging_factor:float = 0.01):
    source_network_weights = source_network.get_weights()
    target_network_weights = target_network.get_weights()
    averaged_weights = []
    for sourc_weight, target_weight in zip(source_network_weights, target_network_weights):
        fraction_of_kept_weights = polyak_averaging_factor * target_weight
        fraction_of_updated_weights = (1 - polyak_averaging_factor) * sourc_weight
        average_weight = fraction_of_kept_weights + fraction_of_updated_weights
        averaged_weights.append(average_weight)
    
    target_network.set_weights(averaged_weights)


def ddqn_algorithm(max_train_steps:int = 10000, max_test_steps:int = 1000, prefill_erp_steps:int = 100, save_testing_visuals:bool = False):
    ENVIRONMENT_NAME = "ALE/Breakout-v5"
    NUM_ACTIONS = gym.make(ENVIRONMENT_NAME).action_space.n
    ERP_MAX_SIZE = 100000
    PARALLEL_GAME_UNROLLS = 16
    UNROLL_STEPS = 4
    OBV_PREPROCESSING_FUNC = obv_preprocessing_function

    EPSILON = 0.2
    GAMMA = 0.98
    NUM_TRAINING_STEPS_PER_ITER = 4
    NUM_TRAINING_ITERATION = max_train_steps
    BATCH_SIZE = 16
    PROGRESS_REPORT_INTERVAL = int(NUM_TRAINING_ITERATION * 0.2)
    if PROGRESS_REPORT_INTERVAL == 0:
        PROGRESS_REPORT_INTERVAL = 1
    NUM_PARALLEL_TEST_ENVS = 16
    MAX_STEPS_PER_GAME = max_test_steps
    PREFILL_ERP_STEPS = prefill_erp_steps
    POLYAK_AVERAGING_FACTOR = 0.99

    erp = ExperienceReplayBuffer(max_size=ERP_MAX_SIZE,env_name=ENVIRONMENT_NAME,
                                 parallel_game_unrolls=PARALLEL_GAME_UNROLLS,
                                 unroll_steps=UNROLL_STEPS,
                                 obv_preprocessing_function=OBV_PREPROCESSING_FUNC)
    
    # the network to be trained
    dqn_agent = create_dqn_model(NUM_ACTIONS)
    # the target network to calculate the q_targets (to address the moving target problem)
    dqn_target = create_dqn_model(NUM_ACTIONS)
    dqn_agent.summary()
    dqn_optimizer = tf.keras.optimizers.Adam()
    # test the network
    dqn_agent(tf.random.uniform(shape=(1,84,84,3), minval=0, maxval=255, dtype=tf.float32))
    dqn_target(tf.random.uniform(shape=(1,84,84,3), minval=0, maxval=255, dtype=tf.float32))
    print("test_done")
    # copy the weights from the agent to the target
    polyak_averaging_weights(dqn_agent, dqn_target, polyak_averaging_factor=POLYAK_AVERAGING_FACTOR)

    return_tracker = []
    dgn_prediction_erroe_tracker = []
    average_q_values_tracker = []
    test_iteration_tracker = []

    # prefill the replay buffer
    prefill_expploration = 1.0
    for i in tqdm(range(PREFILL_ERP_STEPS), desc="Prefilling ERP"):
        erp.fill_with_samples(dqn_agent, prefill_expploration) # epsilon = 1 -> always take random actions for the prefilling

    iteration = 0
    done = False

    prog_bar = tqdm(total=NUM_TRAINING_ITERATION, desc="Training DDQN")
    while not done and iteration < NUM_TRAINING_ITERATION:
        iteration += 1
        # step 1: interact with environment and put some s,a,r,s' into replay buffer
        erp.fill_with_samples(dqn_agent,EPSILON)
        dataset = erp.create_dataset_tf()
        # step 2: train the network on some samples from the replay buffer
        mean_loss, mean_q_values = train_dqn(dqn_agent,dqn_target,dataset,dqn_optimizer,GAMMA,NUM_TRAINING_STEPS_PER_ITER,BATCH_SIZE)
        # update the target network weights
        polyak_averaging_weights(dqn_agent, dqn_target, polyak_averaging_factor=0.01)
        prog_bar.update(1)
        # step 3: test the network
        if (iteration - 1) % PROGRESS_REPORT_INTERVAL == 0:
            average_return = test_q_network(dqn_agent,ENVIRONMENT_NAME,NUM_PARALLEL_TEST_ENVS,
                                            MAX_STEPS_PER_GAME,gamma=GAMMA)
            return_tracker.append(average_return)
            dgn_prediction_erroe_tracker.append(mean_loss)
            average_q_values_tracker.append(mean_q_values)
            test_iteration_tracker.append(iteration)
            print("====================================")
            print(f"######### TESTING , iteration: {iteration} :")
            print("average return: ", average_return)
            print("mean loss: ", mean_loss)
            print("mean q values estimates: ", mean_q_values)
            print("====================================")
            results_df = pd.DataFrame({"test_iteration":test_iteration_tracker,"average_return":return_tracker,
                                       "average_q_values":average_q_values_tracker, "average_loss":dgn_prediction_erroe_tracker})
            results_df = results_df.set_index("test_iteration")
            timestring = datetime.datetime.now().strftime("%m_%d-%H_%M_%S")
            results_df.to_csv(f"./results/dqn_results_{timestring}_{iteration}.csv")
            if save_testing_visuals:
                visualise_results(results_df,iteration)
        # done = True # some condition to stop the training
    prog_bar.close()
    
    # make a dataframe to store the results
    results_df = pd.DataFrame({"test_iteration":test_iteration_tracker,"average_return":return_tracker,
                               "average_q_values":average_q_values_tracker,"average_loss":dgn_prediction_erroe_tracker})
    results_df = results_df.set_index("test_iteration")
    timestring = datetime.datetime.now().strftime("%m_%d-%H_%M_%S")
    visualise_results(results_df,iteration)
    results_df.to_csv(f"./results/dqn_results_{timestring}_{iteration}.csv")
    dqn_agent.save_weights(f"./checkpoints/dqn_agent_weights_{timestring}_{iteration}")
    return results_df

def str2bool(v):
    return v.lower() in ('true')

# ----------------- main -----------------
# ----------------------------------------
# ----------------------------------------

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--max_train_steps", type=int, default=1)
    parser.add_argument("--max_test_steps", type=int, default=1)
    parser.add_argument("--prefill_erp_steps", type=int, default=1)
    parser.add_argument("--save_testing_visuals", type=str2bool, default=False)

    args = parser.parse_args()
    # config = vars(args)
    # print(config)


    results = ddqn_algorithm(max_train_steps=args.max_train_steps,
                            max_test_steps=args.max_test_steps,
                            prefill_erp_steps=args.prefill_erp_steps,
                            save_testing_visuals=args.save_testing_visuals)
    
    visualise_results(results, step='final')
   

