#!/usr/bin/env python3
"""
AWS Trainium Chess Workshop - Example Code

Demonstrates practical usage patterns for deploying and testing your fine-tuned
chess model with vLLM and Stockfish baselines.

For CLI usage and tournaments, see: python run_game.py --help
"""

import sys
from agents import VLLMAgent, StockfishAgent, RandomAgent
from env import ChessEnvironment


def test_vllm_connection():
    """Test connection to vLLM server."""
    print("=== Testing vLLM Connection ===\n")

    agent = VLLMAgent()

    if agent.test_connection():
        print("✓ vLLM server is running and accessible")
        print(f"  Base URL: {agent.base_url}")
        print(f"  Model: {agent.model}")
        print(f"  Temperature: {agent.temperature}")
        print(f"  Max tokens: {agent.max_tokens}")
        return True
    else:
        print("✗ Cannot connect to vLLM server")
        print("\nTroubleshooting:")
        print("1. Check if server is running: lsof -i :8000")
        print("2. Start server: cd vllm-server && bash vllm.sh")
        print("3. View logs: tail -f vllm-server/vllm-server.log")
        return False


def quick_test_game():
    """Run a quick test game with verbose output."""
    print("\n=== Quick Test Game: vLLM vs Random ===\n")

    # Create agents
    vllm = VLLMAgent()
    random = RandomAgent()

    # Create environment with short game for testing
    env = ChessEnvironment(vllm, random, max_moves=20, time_limit=30.0)

    print(f"White: {vllm.__class__.__name__}")
    print(f"Black: {random.__class__.__name__}")
    print(f"Max moves: {env.max_moves}")
    print(f"Time limit: {env.time_limit}s per move\n")

    # Play game with verbose output
    result = env.play_game(verbose=True)

    # Display results
    print("\n" + "="*50)
    print("GAME RESULTS")
    print("="*50)
    print(f"Result: {result['result']}")
    print(f"Moves played: {result['moves_played']}")
    print(f"Game over reason: {result['game_over_reason']}")
    print(f"White illegal attempts: {result['white_illegal_attempts']}")
    print(f"Black illegal attempts: {result['black_illegal_attempts']}")

    # Export to PGN
    pgn_file = "quick_test.pgn"
    if env.export_pgn_file(pgn_file):
        print(f"\n✓ Game exported to {pgn_file}")

    return result


def test_against_stockfish():
    """Test vLLM agent against Stockfish baseline."""
    print("\n=== Testing Against Stockfish Baseline ===\n")

    # Create agents
    vllm = VLLMAgent()
    stockfish = StockfishAgent(skill_level=5, depth=10)

    # Create environment
    env = ChessEnvironment(vllm, stockfish, max_moves=100, time_limit=30.0)

    print(f"White (your model): VLLMAgent")
    print(f"Black (baseline): StockfishAgent (skill=5, depth=10)")
    print(f"Max moves: {env.max_moves}")
    print(f"Time limit: {env.time_limit}s per move\n")

    # Play game
    print("Playing game... (this may take a few minutes)\n")
    result = env.play_game(verbose=False)

    # Display results
    print("\n" + "="*50)
    print("BASELINE TEST RESULTS")
    print("="*50)
    print(f"Result: {result['result']}")
    print(f"Moves played: {result['moves_played']}")
    print(f"Game over reason: {result['game_over_reason']}")

    # Interpret result
    if result['result'] == '1-0':
        print("\n✓ Your model won! Strong performance against skill 5.")
    elif result['result'] == '0-1':
        print("\n✗ Your model lost. Consider:")
        print("  - Reviewing model outputs (run with verbose=True)")
        print("  - Testing against weaker baseline (skill 1-3)")
        print("  - Checking illegal move rate")
    else:
        print("\n⚡ Draw. Your model competed well!")

    print(f"\nWhite illegal attempts: {result['white_illegal_attempts']}")
    print(f"Black illegal attempts: {result['black_illegal_attempts']}")

    # Export to PGN
    pgn_file = "baseline_test.pgn"
    if env.export_pgn_file(pgn_file):
        print(f"\n✓ Game exported to {pgn_file}")

    return result


def multiple_baseline_tests():
    """Run multiple games against Stockfish to estimate strength."""
    print("\n=== Multiple Baseline Tests ===\n")

    num_games = 3
    skill_level = 5

    print(f"Running {num_games} games vs Stockfish (skill={skill_level})\n")

    # Track results
    wins = 0
    losses = 0
    draws = 0
    total_moves = 0
    illegal_attempts = 0

    for i in range(num_games):
        print(f"Game {i+1}/{num_games}...", end=" ", flush=True)

        # Create agents
        vllm = VLLMAgent()
        stockfish = StockfishAgent(skill_level=skill_level, depth=10)

        # Alternate colors
        if i % 2 == 0:
            env = ChessEnvironment(vllm, stockfish, max_moves=100, time_limit=30.0)
            white_is_vllm = True
        else:
            env = ChessEnvironment(stockfish, vllm, max_moves=100, time_limit=30.0)
            white_is_vllm = False

        # Play game
        result = env.play_game(verbose=False)

        # Update statistics
        total_moves += result['moves_played']

        if white_is_vllm:
            illegal_attempts += result['white_illegal_attempts']
            if result['result'] == '1-0':
                wins += 1
            elif result['result'] == '0-1':
                losses += 1
            else:
                draws += 1
        else:
            illegal_attempts += result['black_illegal_attempts']
            if result['result'] == '0-1':
                wins += 1
            elif result['result'] == '1-0':
                losses += 1
            else:
                draws += 1

        print(f"Result: {result['result']} ({result['moves_played']} moves)")

    # Display summary
    print("\n" + "="*50)
    print("BASELINE TEST SUMMARY")
    print("="*50)
    print(f"Games played: {num_games}")
    print(f"Wins: {wins} ({wins/num_games*100:.1f}%)")
    print(f"Losses: {losses} ({losses/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
    print(f"Average moves per game: {total_moves/num_games:.1f}")
    print(f"Total illegal attempts: {illegal_attempts}")
    print(f"Illegal move rate: {illegal_attempts/(total_moves or 1)*100:.2f}%")

    # Interpretation
    print("\n" + "="*50)
    print("INTERPRETATION")
    print("="*50)

    win_rate = wins / num_games * 100
    if win_rate >= 60:
        print("✓ Excellent performance! Your model is strong.")
        print("  Try testing against skill 10 baseline.")
    elif win_rate >= 40:
        print("⚡ Good performance. Your model is competitive.")
        print("  Continue optimizing and test more games.")
    else:
        print("✗ Needs improvement. Consider:")
        print("  - Test against weaker baseline (skill 1-3)")
        print("  - Review model outputs for invalid moves")
        print("  - Check prompt template effectiveness")

    illegal_rate = illegal_attempts / (total_moves or 1) * 100
    if illegal_rate > 5:
        print(f"\n⚠ High illegal move rate ({illegal_rate:.1f}%)")
        print("  - Check model output format (UCI notation required)")
        print("  - Verify legal moves are provided in prompt")
        print("  - Consider increasing max_tokens for reasoning")


def demonstrate_custom_configuration():
    """Demonstrate custom agent configuration."""
    print("\n=== Custom Agent Configuration ===\n")

    # Custom vLLM configuration
    custom_vllm = VLLMAgent(
        base_url="http://localhost:8000/v1",
        model="Qwen3-chess",
        temperature=0.05,      # Lower for more deterministic play
        max_tokens=30,         # Fewer tokens for faster inference
        timeout=60.0           # Longer timeout for complex positions
    )

    print("Custom VLLMAgent configuration:")
    print(f"  Base URL: {custom_vllm.base_url}")
    print(f"  Model: {custom_vllm.model}")
    print(f"  Temperature: {custom_vllm.temperature}")
    print(f"  Max tokens: {custom_vllm.max_tokens}")
    print(f"  Timeout: {custom_vllm.timeout}s")

    # Custom Stockfish configuration
    custom_stockfish = StockfishAgent(
        skill_level=3,         # Weaker for testing
        depth=5,               # Shallower search
        time_limit_ms=500      # Faster moves
    )

    print("\nCustom StockfishAgent configuration:")
    print(f"  Skill level: {custom_stockfish.skill_level}")
    print(f"  Depth: {custom_stockfish.depth}")
    print(f"  Time limit: {custom_stockfish.time_limit_ms}ms")

    # Custom environment
    custom_env = ChessEnvironment(
        custom_vllm,
        custom_stockfish,
        max_moves=50,          # Shorter games for testing
        time_limit=15.0        # Faster games
    )

    print("\nCustom Environment configuration:")
    print(f"  Max moves: {custom_env.max_moves}")
    print(f"  Time limit per move: {custom_env.time_limit}s")


def debugging_example():
    """Demonstrate debugging techniques."""
    print("\n=== Debugging Example ===\n")

    print("Debugging tips:")
    print("\n1. Test vLLM connection:")
    print("   python -c \"from agents import VLLMAgent; print(VLLMAgent().test_connection())\"")

    print("\n2. Run game with verbose output:")
    print("   env.play_game(verbose=True)")

    print("\n3. Check server logs:")
    print("   tail -f vllm-server/vllm-server.log")

    print("\n4. Inspect game results:")
    print("   result = env.play_game()")
    print("   print(result['white_illegal_attempts'])")
    print("   print(result['black_illegal_attempts'])")

    print("\n5. Export and analyze PGN:")
    print("   env.export_pgn_file('debug_game.pgn')")
    print("   # Open in Lichess, Chess.com, or other analysis tool")

    print("\n6. Check Neuron runtime:")
    print("   neuron-ls")
    print("   neuron-top")

    print("\n7. Verify model output format:")
    print("   # Model should output: <uci_move>e2e4</uci_move>")
    print("   # UCI notation: source + destination (e.g., e2e4, g1f3)")


def main():
    """Main demonstration program."""
    print("="*60)
    print("AWS Trainium Chess Workshop - Example Code")
    print("="*60)

    # Test vLLM connection first
    if not test_vllm_connection():
        print("\n⚠ vLLM server is not accessible. Please start the server first.")
        print("\nTo start vLLM server:")
        print("  cd vllm-server")
        print("  bash vllm.sh")
        sys.exit(1)

    print("\n" + "="*60)
    print("Choose an example to run:")
    print("="*60)
    print("1. Quick test game (vLLM vs Random)")
    print("2. Test against Stockfish baseline")
    print("3. Multiple baseline tests (3 games)")
    print("4. Show custom configuration examples")
    print("5. Show debugging tips")
    print("6. Run all examples")
    print("0. Exit")

    try:
        choice = input("\nEnter choice (0-6): ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n\nExiting...")
        sys.exit(0)

    if choice == '1':
        quick_test_game()
    elif choice == '2':
        test_against_stockfish()
    elif choice == '3':
        multiple_baseline_tests()
    elif choice == '4':
        demonstrate_custom_configuration()
    elif choice == '5':
        debugging_example()
    elif choice == '6':
        quick_test_game()
        test_against_stockfish()
        multiple_baseline_tests()
        demonstrate_custom_configuration()
        debugging_example()
    elif choice == '0':
        print("Exiting...")
        sys.exit(0)
    else:
        print("Invalid choice. Please run again.")
        sys.exit(1)

    print("\n" + "="*60)
    print("For tournament mode and advanced features:")
    print("  python run_game.py --help")
    print("\nFor documentation:")
    print("  See README.md and docs/WORKSHOP_GUIDE.md")
    print("="*60)


if __name__ == "__main__":
    main()
