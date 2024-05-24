import numpy as np
import random
import pyspiel


game = pyspiel.load_game("goofspiel(num_cards=5,players=2)")

if __name__ == "__main__":
    print("game type = ", game.get_type())
    print("obs_dim = ", game.information_state_tensor_size())
    print("max step = ", game.max_game_length())
    print("act_dim = ", game.num_distinct_actions())

    # Create the initial state
    state = game.new_initial_state()

    # Print the initial state
    print(str(state))

    while not state.is_terminal():
        # The state can be three different types: chance node,
        # simultaneous node, or decision node
        if state.is_chance_node():
            # Chance node: sample an outcome
            outcomes = state.chance_outcomes()
            num_actions = len(outcomes)
            print("Chance node, got " + str(num_actions) + " outcomes")
            action_list, prob_list = zip(*outcomes)
            action = np.random.choice(action_list, p=prob_list)
            print("Sampled outcome: ",
                    state.action_to_string(state.current_player(), action))
            state.apply_action(action)
        elif state.is_simultaneous_node():
            # Simultaneous node: sample actions for all players.
            random_choice = lambda a: np.random.choice(a) if a else [0]
            chosen_actions = [
                random_choice(state.legal_actions(pid))
                for pid in range(game.num_players())
            ]
            # print("state = ", [
            #     state.information_state_tensor(pid)
            #     for pid, action in enumerate(chosen_actions)
            #     ])
            # print(state.information_state_tensor(0) == state.information_state_tensor(1))
            print("cur players = ", state.current_player())
            print("Chosen actions: ", [
                state.action_to_string(pid, action)
                for pid, action in enumerate(chosen_actions)
            ], chosen_actions)
            state.apply_actions(chosen_actions)
        else:
            # Decision node: sample action for the single current player
            action = random.choice(state.legal_actions(state.current_player()))
            action_string = state.action_to_string(state.current_player(), action)
            print("Player ", state.current_player(), ", randomly sampled action: ",
                    action_string)
            state.apply_action(action)
            print(str(state))

    print(game.get_type().chance_mode)

    # Game is now done. Print utilities for each player
    returns = state.returns()
    for pid in range(game.num_players()):
        print("Utility for player {} is {}".format(pid, returns[pid]))