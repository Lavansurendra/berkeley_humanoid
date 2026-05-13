List of changes made from berkeley_humanoid github
===============

The following is a rough summary of some of the changes made to the original berkeley_humanoid github. It still needs to be completed and corrected for accuracy.

**Addition of berkeley\_humanoid.usd and berkeley\_scene.xml files**

- Push date

  - March 18th, 2026

- What was the change

  - Added new files for kinematics simulation in mujoco

- Why this change was made

  - Isaaclab relies on GPU based PhysX engine while Mujoco uses CPU-based continuous time physics and soft contact solvers

  - This change was necessary to test the kinematics and physics of the robot before setting up reinforcement learning

- What the change allowed us to do

  - Visually inspect robot joins, response to forces, rendering and other features in mujoco engine

**Addition of berkeley\_mujoco\_env.py**

- Push date

  - March 18th, 2026

- What was the change

  - Added a python file containing the logic of several removed old implementation python files which all modify the logic behind physics simulation in mujoco

    - More specifics in the original content of removed files section

  - Moved properties specified in several removed old implementation python files into the berkeley\_scene.xml file

- Why this change was made

  - We required the addition of this file to transform all the logic behind the physics of the original implementation into a version that would be compatible with mujoco

    - We also stripped away a lot of the existing reward and domain randomization logic and added parts of it back in later updates as we determined what was actually necessary

- Original content of removed files

  - velocity\_env\_cfg.py 

    - This file contained some of the configuration for the terrain and robot that would be used in the Isaac Lab simulator, along with logic that would help the policy train more effectively

      - The ground and robot USD were specified here along with some sensors attached to specific body parts which would keep track of the height of the terrain surrounding the robot and the contact forces it was experiencing

        - The ground properties and robot.xml files are both now specified in the berkeley\_scene.xml file

        - The height of the terrain surrounding the robot and the contact forces the robot was experiencing have not been implemented

      - Additionally the method of control (Joint Position control) was specified in this file for each joint along with a scaling factor to prevent movements from becoming too aggressive

        - The initialization of the method of control is now done in the init function and the scaling is now done in the step function of the berkeley\_mujoco\_env file

      - Also a goal for the agent was specified to be a velocity of 1.0m/s with some random direction

        - This was not implemented in the new version

      - Also a class containing the observation information which was sent to the learning policy was defined containing the torso velocity, torso angular velocity, projected gravity, velocity targets, joint positions, joint velocities, last action and the height of the terrain surrounding the robot

        - This was reimplemented in the berkeley\_mujoco\_env.py file although we did not provide all the information originally provided (only the raw position, velocity and angular velocity values were provided)

        - In the Isaaclab implementation an action manager object (defined within isaaclab) which would take the values outputted by the policy and use the specifications of this class to perform output scaling

          - Additionally if there were differences between the control frequency and the update rate of the physics engine the action manager would hold the same joint command over multiple steps of the physics engine

        - The action manager does not exist in mujoco and so we had to directly write code to implement the features it did in the Isaac Lab implementation later on

      - Also the parameters for domain randomization were implemented here with a class that would randomly change the floor friction, scale the weight of each robot limb, add or remove ghost weight to the robot’s joints, change the robot’s starting positions, change the joint angles, and randomly push the robot (these were then fed to [events.py](http://events.py) which would actually perform the randomization at every reset of the robot).

        - Much of this was removed but later on randomized pushes and random initial starting positions were implemented in the berkeley\_mujoco\_env.py file 

      - Additionally the weighting for each of the rewards of the original implementation was defined in this file

        - This was all removed and replaced with a placeholder reward in the berkeley\_mujoco\_env.py file

      - Additionally the termination conditions for training were specified here with a class which would end episodes after a defined episode length and end episodes if the torso made physical contact with the ground

        - These were reimplemented in the berkeley\_mujoco\_env.py file although the torso termination condition was modified to end an episode if the robots height ever dropped below 0.4 m (this height was later dropped to 0.3 m)

      - Additionally curriculum learning logic was implemented in this file by setting up code which would gradually increase the force the robot was pushed with and the target velocity in successive runs and also move the robot to bumpier parts of the terrain if it was able to walk a certain distance without falling

        - These were not implemented in the berkeley\_mujoco\_env.py file although much later the push logic and curriculum learning were implemented

  - [events.py](http://events.py) 

    - This file contained code which would perform domain randomization of the join default position and friction

      - This code was not implemented in the berkeley\_mujoco\_env.py file

  - [rewards.py](http://rewards.py)

    - This file contained function which calculated the rewards used by a reward manager object (specific to Isaac Lab)

      - None of the rewards from this file were implemented although the feet\_sliding reward was saved and modified to work with mujoco in case we wanted to use it later

      - We will have to write all the code for weighting and setting up rewards later

- What the change allowed us to do

  - This allowed us to actually perform multiple physics steps according to the previously implemented Isaac Lab logic but using Mujoco as our simulator

**Modification to actuator\_pd.py file**

- Push date

  - March 18, 2026

- What was the change

  - Introduced manual PID calculations with a friction penalty to simulate real berkeley humanoid actuator physics

- Why this change was made

  - In the original Isaac Lab implementation the physics engine PhysX did all motor PID calculations and the berkeley humanoid team added a friction penalty to simulate what actual actuators would be like in the berkeley humanoids gearboxes but in mujoco we had to do this manually

    - NOTE: mujoco does have motor tags which can do the friction math behind the scenes but it uses a simpler non differentiable mathematical formula (linear force model) than what the berkeley humanoid team does and since real gearboxes have more complex friction the berkeley humanoid teams formula is much more accurate to real world motors

      - We decided to use the berkeley humanoid team's formula on top of mujoco joint tags instead of mujoco’s motor tags (or their frictionloss parameter) in order to implement this more complicated friction formula and avoid potential sim to real issues

  - Additionally instead of using torch we used numpy for math as we needed everything to work on CPU

- What the change allowed us to do

  - This allowed us to use mujoco’s actuators tag and still have accurate friction estimations

**Rewriting of** [**train.py**](http://train.py)

- Push date

  - March 18th, 2026

- What was the change

  - Rewrote [train.py](http://train.py) so that it works with mujoco and stablebaselines3 PPO instead of Isaac Lab and RSL-RL PPO

- Why this change was made

  - The original logic used the RSL-RL library which is designed for running parallel environments on GPU and so had a “Runner” class which used configclass objects for training

    - We wanted this robot to be trainable in CPU and the industry standard for CPU uses the StableBaselines3 implementation of PPO so we had to set everything up to work with SB3’s PPO(MlpPolicy) class

  - Additionally the previous implementation had GPU parallelization which was done entirely via Isaac Sim (users would just set the number of environments) and in order to do CPU parallelization we had to use SB3’s make\_vec\_env and SubprocVecEnv

  - Hyperparameters (learning rate, batch size) used to be set up in the rsl\_rl\_cfg.py file but in the SB3 implementation we just hard coded them directly into the [train.py](http://train.py) file

  - Also we added the checkpoint callback which is a specialized SB3 function that maintains a counter and saves model parameters after a certain number of timesteps

- What this change allowed us to do

  - We were able to train without requiring GPU usage and we were also able to iterate on training much faster

**Rewriting of** [**play.py**](http://play.py)

- Push date

  - March 18, 2026

- What was the change

  - Rewrote play[.py](http://train.py)  to accurately view the models in mujoco’s viewer instead of the original isaac lab versions

- Why this change was made

  - The original implementation used the Isaac Lab task registry and used an omniverse simulation window which required GPU and a longer loading time. Since we wanted to run on CPU we used the BerkeleyHumanoidMujocoEnv class implemented in mujoco’s viewer 

**Removal of Isaac Lab configuration and terrain generation files**

- Push date

  - March 18th, 2026

- Why this change was made

  - The files in the original implementation were explicitly associated with omniverse and pytorch which are incompatible with the Mujoco setup 

- What this change allowed us to do 

  - More easily navigate the complicated github

**Addition of the berkeley humanoid URDF**

- Push date

  - March 18th, 2026

- What was the change

  - Imported the berkeley\_humanoid\_description github

- Why this change was made

  - Up until this point we had only the berkeley\_humanoid.usd file which is not compatible with mujoco

  - From this github we were able to get a urdf and several configuration files (STL and meshes and ROS configuration files)

- What this change allows us to do

  - We can now use the parameters and urdf file to create an xml file with these parameters specified that we can iterate on for accuracy afterwards

**Transformation of the robot.urdf to a robot.xml file**

- Push date

  - March 19th, 2026

- What was the change

  - Used mujoco’s compiler function to convert URDF into an xml file

- Why this change was made

  - Previously the URDF for the robot was loaded directly into the berkeley\_scene.xml file via an include tag but Mujoco’s inbuilt URDF parser ignores some of the tags in that file (ex. Transmission tags for ROS) which meant our model had no actuators or accurate joints

    - Instead of using the URDF parser every time we ran training or watched a model run we used a one time compiler function within Mujoco to convert the existing URDF file into a robot.xml file which we could then 

- What this change allows us to do

  - This allows us to adjust all the physics of the robot directly and create tags which act as actuators accurate to those found on the physical robot

**Modifications to PD control**

- Push date

  - March 22nd, 2026

- What was the change

  - Organization of PD control code

- Why this change was made

  - The PD control logic was stored in the actuator\_pd.py file but we moved that math into the berkeley\_mujoco\_env.py file for simplification sake

  - Additionally we chose arbitrary values for the proportional gain and derivative gain parameters (we would later change these based on the actual humanoid’s configurations)

- What this change allowed us to do

  - Organize the github

**Introduction of Torque Clipping**

- Push date

  - March 22nd, 2026

- What was the change

  - Introduced torque clipping (limits on how much torque any motor could apply)

- Why this change was made

  - After the conversion from a URDF file to an xml file we had several physics instabilities resulting in nonphysical movement and simulation errors when numerical solvers were unable to perform calculations due to the nearly infinite force that the robot was attempting apply

    - We needed to introduce torque clipping to limit the amount of force a motor could apply based on what is feasible for that actual robot. These limits were arbitrarily chosen at first but in later commits modified based on the specifications of the actual berkeley\_humanoid

- What this change allowed us to do

  - This change in combination with several other changes allowed us to simulate far more accurately to what the real life humanoid would be able to do

**Nominal Stance & Reset**

- Push date

  - March 22nd, 2026

- What was the change

  - Introduced a nominal stance (specific position the robot resets to in simulation for both training and viewing) with slightly bent joints

  - Reduced the z height termination condition to accommodate for this new slightly crouched starting position

- Why this change was made

  - The initial configuration of the robot had it falling from slightly above the floor with perfectly straight joints, which led to all of the actuators not having to do any initial work with the robot’s “bones” taking 100% of the force of the fall

  - When this occurs the knee has a lever arm of 0 and since torque is equal to force multiplied by the length of the lever arm in order to calculate the required force needed to apply a certain amount of torque a mathematical solver (in our case Mujoco’s solver) inverts this relationship and estimate the required force by essentially dividing the required torque by the lever arm

    - In doing this a division by 0 errors occur leading to an infinite or NaN being set as the amount of torque to apply and crashing the simulation window 

    - We needed to start the robot in a slightly bent position to avoid this issue

    - NOTE: the actual math that the solver does is much more complicated so see the mujoco constrain solver section for more details (<https://mujoco.readthedocs.io/en/stable/computation/index.html#constraint-solver>)

- What this change allows us to do 

  - Train without several NaN errors and crashes in simulation

**Action Scaling**

- Push date

  - March 22nd, 2026

- What was the change

  - Limited the range of actions the agent could take by forcing them to remain within 0.3 radians (approximately 17.2 degrees) from the nominal stance

- Why this change was made

  - Scaling requirement

    - We must limit the range of actions the policy can output to prevent the policy from choosing in its exploration to select actions which tell the motors to travel 1 or more radians (approximately 57 degrees) within a fraction of a second as this could lead to violent forces being applied

  - Nominal pose scaling requirement

    - Bidirectional movement

      - In order to walk a joint must be able to flex and extend

        - However if the position that action scaling was based around had all joints straight (initially 0) the joints are locked and cannot physically explore in one direction of movement

          - For example the agent would be initially unable to explore by flexing the knee joint as it would start already fully extended

            - This would significantly slow down training as instead of the policy initially being able to experience what occurs when it straightens and bends its legs (pushing off the floor vs absorbing force) it can only learn one of those things

      - Also, when a policy is initially untrained its average output is approximately a vector of 0s

        - However if the position that action scaling was based around had all joints straight (initially 0)  this could lead to the policy commanding joints to remain near their absolute zero positions (leading to the infinite errors explained in the previous update)

          - Even if the errors did not occur this would slow down training as the robot would fall over far more frequently initially and the agent would not be able to tell which of its random actions actually improve its circumstances

- What this change allows us to do

  - Significantly improve our training time and limit NaN errors and crashes in simulation

**Contact Dimension changes**

- Push date

  - March 22nd, 2026

- What was the change

  - Changed the condim attribute on the floor geometry from 3 to 4

- Why this change was made

  - Initially when this attribute was set to 3 any time an object came into contact with the floor the friction coefficients associated with that object were placed into the 3d coulomb’s law of friction inequality which is a mathematical representation of the idea that the tangential friction force between two solid surfaces is directly proportional to the normal force pressing them together (and more specifically that these tangential friction forces are always less than or equal to the product of the coefficient of friction and the normal force)

    - $$f_1 \ge 0, \quad f_1^2 \ge \frac{f_2^2}{\mu_1^2} + \frac{f_3^2}{\mu_2^2}$$

      - i is one of the user specified coefficients of friction

        - NOTE: the same value is set as the static and kinematic friction to avoid issues with the Newton solver

      - f1 is the normal force

      - f2 is the tangential friction force in one direction

      - f3 is the tangential friction force a perpendicular direction to the direction of f2

  - Then using convex projection mujoco determines the optimal set of forces that fulfill that inequality to determine the strength of the friction force that the floor applies on the contacting object

  - However this discludes torsional friction which would normally act against the torsion of the contacting object about the axis that the normal force is applied in

    - This allowed the robot to make any foot twisting motions on the floor without experiencing any friction (which caused simulation to act as though it was slipping on ice)

  - By setting this attribute to 4 we include torsional friction in the new inequality for coulomb’s law of friction and apply a force against any twisting of the robot’s feet (or other contact points)

    - $$f_1 \ge 0, \quad f_1^2 \ge \frac{f_2^2}{\mu_1^2} + \frac{f_3^2}{\mu_2^2} + \frac{f_4^2}{\mu_3^2}$$

      - i is one of the user specified coefficients of friction

        - NOTE: the same value is set as the static and kinematic friction to avoid issues with the Newton solver

      - f1 is the normal force

      - f2 is the tangential friction force in one direction

      - f3 is the tangential friction force a perpendicular direction to the direction of f2

      - f3 is the friction force applied against the torsion of the contacting object

- What did this change allow us to do

  - Train without excessive non physical slipping due to lack of friction when the agent would twist its feet on the floor
