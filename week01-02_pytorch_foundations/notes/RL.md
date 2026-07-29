RL Primer
-----------
- Supervised Learning: we had a fixed dataset including inputs and correct labels, without any interactions from us the model predicted a known right answer

- but RL, an agent alwasy interact with an environment over time in a loop. 
    1. Environment gives agent a STATE (also called "observation")
    2. Agent picks an ACTION based on that state
    3. Environment responds with:
   - a new STATE (the consequence of that action)
   - a REWARD (a number saying how good/bad that action was)
    4. Repeat, until the episode ends

a. key vocabulary mapped to cartpole specifically
    a.1. State/Observation - number that explains whats happening right now
    a.2. Action - a choice the agent makes, cartpole has 2 possible actions i.e. push cart left, push cart right
    a.3. Reward - a number given to the agent to assess its step
    a.4. Episode - one full run from reset to termination
    a.5. Policy - certain conditions helping to decide which action to take given a state

** the key differentiator between SL and RL is reward often delayed and doesnt directly say what the right action was but to the SL side, every training came with an immediate correct answer. 

** CartPole gives you +1 per timestamp regardless of the which direction you pushed, that do not say the action was correct. The agent has to figure it out through trial and error to learn which sequences lend to lead to good long term outcomes

CartPole
--------
** CartPole is simply a pole on a cart

** CartPole is actually a 2D problem (the cart moves along a single track, left-right only; the pole tips forward-backward in that same plane)

** State/Obeservation: it has 4 states,
    - Cart Position, Velocity
    - Pole Angle, Angular Velocity

** CartPole's reward policy, moving left or right for every timestamp give +1 reward. so until the episode terminates (Pole tips over) the reward system will give +1. The reward does not signal anything, it keeps rewarding until the episode termiantes. 

** credit assignment problem: the agent learns only after many iterations of actions, till that it delayed unit one step closure to the termination of the episode.

** the single biggest structural reason RL training is harder and less stable than the supervised training you've been doing, where every single example came with instant, direct feedback.

** Random-action CartPole episodes typically only last somewhere around 10-30 steps on average before the pole tips too far, purely by chance, since there's no strategy involved at all.