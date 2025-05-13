import copy
import torch
import argparse
import os
import sys
import traceback

# macOS-specific tweaks for Retina and focus issues
if sys.platform == 'darwin':
    os.environ['SDL_HINT_VIDEO_HIGHDPI_DISABLED'] = '0'
    os.environ['SDL_VIDEO_MAC_FULLSCREEN_SPACES'] = '1'

def check_display():
    """Check if a display is available."""
    try:
        import pygame
        pygame.init()
        pygame.display.init()
        pygame.display.quit()
        pygame.quit()
        return True
    except Exception as e:
        print("Error: No display available. Make sure you're running this on a system with a display.")
        print(f"Error details: {str(e)}")
        return False

def load_policy_model(value, device):
    """Load policy model with error handling."""
    model_path = f"RL/results/default_after_update_{value}.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        return torch.load(model_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Error loading model {model_path}: {str(e)}")

try:
    from game.enums import PlayerId
    from env.wrapper import EnvWrapper
    from ui.display import Display
    from RL.models.build_agent_model import build_agent_model
    from RL.forward_search_policy.policy import ForwardSearchPolicy
    from RL.forward_search_policy.sample_actions_fn import default_sample_actions
except ImportError as e:
    print(f"Error importing required modules: {str(e)}")
    print("Make sure all dependencies are installed and the project structure is correct.")
    sys.exit(1)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():  # For Apple Silicon
    device = "mps"

if __name__ == "__main__":
    try:
        if not check_display():
            sys.exit(1)

        parser = argparse.ArgumentParser()
        parser.add_argument('--policy1', type=str, default="human")
        parser.add_argument('--policy2', type=str, default="human")
        parser.add_argument('--policy3', type=str, default="human")
        parser.add_argument('--policy4', type=str, default="human")
        parser.add_argument('--num-processes', type=int, default=1, help="number of processes for forward search policy.")
        parser.add_argument('--max-init-actions', type=int, default=10, help="default maximum actions to be considered by forward search.")
        parser.add_argument('--thinking-time', type=float, default=10, help="default thinking time (seconds) per decision for forward search policy.")
        parser.add_argument('--max-depth', type=int, default=15, help="number of decisions forward search simulations go.")
        parser.add_argument('--dont-consider-all-opening-moves', action='store_true', default=False, help="by default in initial placement phase, forward search considers all possible moves. requires more thinking time for these decisions.")
        parser.add_argument('--trades-on', action='store_true', default=False, help="turn trades on (agents are not generally very good at trading which can be annoying...)")
        args = parser.parse_args()

        policies = {}
        players = [PlayerId.White, PlayerId.Red, PlayerId.Orange, PlayerId.Blue]
        
        for i, policy in enumerate([args.policy1, args.policy2, args.policy3, args.policy4]):
            try:
                if policy == "human":
                    policies[players[i]] = "human"
                elif policy.startswith("RL"):
                    value = int(policy.split("_")[1])
                    policy_state_dict = load_policy_model(value, device)
                    policy = build_agent_model(device=device)
                    policy.load_state_dict(policy_state_dict)
                    policies[players[i]] = copy.deepcopy(policy)
                elif policy.startswith("forward_search"):
                    value = int(policy.split("_")[2])
                    policy_state_dict = load_policy_model(value, device)
                    consider_all_opening_moves = not args.dont_consider_all_opening_moves
                    dont_propose_trades = not args.trades_on
                    policy_fs = ForwardSearchPolicy(
                        policy_state_dict, 
                        default_sample_actions, 
                        max_init_actions=args.max_init_actions,
                        max_depth=args.max_depth, 
                        max_thinking_time=args.thinking_time, 
                        gamma=0.999,
                        num_subprocesses=args.num_processes,
                        consider_all_moves_for_opening_placement=consider_all_opening_moves,
                        dont_propose_trades=dont_propose_trades, 
                        player_id=players[i]
                    )
                    policy_fs.initialise_policy()
                    policies[players[i]] = policy_fs
                else:
                    raise ValueError(f"Incorrect argument supplied for policy: {policy}")
            except Exception as e:
                print(f"Error setting up policy {policy} for player {players[i]}: {str(e)}")
                sys.exit(1)

        max_trades = 4 if args.trades_on else 0
        env = EnvWrapper(policies=policies, max_proposed_trades_per_turn=max_trades)
        env.reset()
        
        try:
            display = Display(env=env, game=env.game, interactive=True, policies=policies, test=False, debug_mode=False)
        except Exception as e:
            print(f"Error initializing display: {str(e)}")
            print("This might be due to pygame/SDL issues on your system.")
            sys.exit(1)

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        sys.exit(1)