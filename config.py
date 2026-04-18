class BerkeleyCfg:
    # --- ROOT LEVEL KEYS ---
    obs_groups = {
        "actor": ["obs"],
        "critic": ["privileged_obs"]
    }
    # Increased data volume to match SB3!
    num_steps_per_env = 2048 

    class algorithm:
        class_name = 'PPO'
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1e-3
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        max_iterations = 50
        save_interval = 50
        experiment_name = "berkeley_humanoid_asym"
        run_name = "initial_test"
        device = 'cpu'
        seed = 42

    class policy:
        class_name = 'ActorCritic'

    class actor:
        class_name = 'MLPModel' 
        hidden_dims = [256, 128, 64]
        activation = 'elu'
        distribution_cfg = {
            "class_name": "GaussianDistribution",
            "init_std": 1.0
        }

    class critic:
        class_name = 'MLPModel'
        hidden_dims = [512, 256, 128]
        activation = 'elu'
        distribution_cfg = None

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05 # Squashes high velocities!
        clip_observations = 100.0
        clip_actions = 100.0

    class env:
        num_envs = 8 # Match your SB3 setup