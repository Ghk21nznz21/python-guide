def Tensorflow_Agent(env):
    tf_agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=loss,
        train_step_counter=tf.Variable(0) )

    tf_agent.initialize()
    summarize_network(q_net)
    tf_agent.train = common.function(tf_agent.train)
    tf_agent.train_step_counter.assign(0)
    return tf_agent

def make_replay_buffer(tf_agent, train_env):
    replay_buffer_capacity = 100000
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_capacity,
    )
    return replay_buffer

def collect_episode(env, policy, num_episodes, replay_buffer):
    episode_counter = 0
    env.reset()

    while episode_counter < num_episodes:
        time_step = env.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = env.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        replay_buffer.add_batch(traj)

        if traj.is_boundary():
            episode_counter += 1



def main():
    train_env = make_env()
    tf_agent = make_tf_agent(train_env)
    replay_buffer = make_replay_buffer(tf_agent, train_env)
    collect_episodes_per_iteration = 1
    batch_size = 64
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2
    ).prefetch(3)
    iterator = iter(dataset)
    collect_episode(
        train_env,
        tf_agent.collect_policy,
        collect_episodes_per_iteration + batch_size,
        replay_buffer,
    )

    returns = []
    losses = []
    weights = []
    num_iterations = 20000
    eval_interval = 50
    num_eval_episodes = 10
    for _ in tqdm.tqdm(range(num_iterations)):
        collect_episode(
            train_env,
            tf_agent.collect_policy,
            collect_episodes_per_iteration,
            replay_buffer,
        )
        experience, unused_info = next(iterator)
        experience2 = trajectory.Trajectory(
            experience.step_type,
            experience.observation,
            experience.action,
            experience.policy_info,
            experience.next_step_type,
            experience.reward,
            experience.discount * 0.95,
        )
        losses.append(tf_agent.train(experience2).loss)
        step = tf_agent.train_step_counter.numpy()

        if step % eval_interval == 0:
            avg_return = compute_avg_return(
                train_env, tf_agent.policy, num_eval_episodes
            )
            returns.append(avg_return)
            weights.append(tf_agent._q_network.layers[0].get_weights())
            plot_returns(eval_interval, returns)
            plot_loss(losses)
            plot_weights(eval_interval, weights)