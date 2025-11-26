"""
vLLM chess agent implementation.

This agent connects to a local vLLM server (running on AWS Neuron/Trainium hardware)
to make chess moves. It uses the OpenAI-compatible API that vLLM exposes.
"""

import os
import time
from typing import Any, Dict, List, Optional

import openai
from dotenv import load_dotenv

import chess

from .base import ChessAgent

# Load environment variables from .env file
load_dotenv()


class VLLMAgent(ChessAgent):
    """
    Chess agent that connects to a local vLLM server.
    
    vLLM provides an OpenAI-compatible API, so this agent reuses the same
    interface but connects to localhost instead of OpenAI's servers.
    
    Key features:
    - Connects to local vLLM server (default: http://localhost:8000/v1)
    - No API key required (local deployment)
    - Uses configurable prompt templates with placeholders
    - Parses moves in UCI notation via <uci_move></uci_move> tags
    - Handles legal move validation
    - Optimized for AWS Neuron/Trainium hardware
    """
    
    # Unicode chess piece characters (same as chess_renderer.py)
    UNICODE_PIECES = {
        'P': '♙',  # White pawn
        'R': '♖',  # White rook
        'N': '♘',  # White knight
        'B': '♗',  # White bishop
        'Q': '♕',  # White queen
        'K': '♔',  # White king
        
        'p': '♟',  # Black pawn
        'r': '♜',  # Black rook
        'n': '♞',  # Black knight
        'b': '♝',  # Black bishop
        'q': '♛',  # Black queen
        'k': '♚',  # Black king
    }
    
    # Default prompt template - STRICT output-only format
    DEFAULT_PROMPT_TEMPLATE = """You are a chess engine.

Given the current position and legal moves, choose the single best move.

FEN: {FEN}
Side to move: {side_to_move}
Legal moves (UCI): {legal_moves_uci}

Output the answer in this exact format and nothing else:

<uci_move>MOVE</uci_move>

Where MOVE is one of the legal moves listed above.
Do NOT include any explanation, commentary, or additional text before or after the tags.
"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        prompt_template: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        fallback_behavior: str = "random_move",
        **kwargs
    ):
        """
        Initialize the vLLM agent.
        
        Args:
            base_url: vLLM server URL. If None, uses VLLM_BASE_URL env var or http://localhost:8080/v1
            model: Model name (matches what vLLM is serving). If None, uses VLLM_MODEL env var or "Qwen3-chess"
            prompt_template: Custom prompt template with placeholders. If None, uses default.
            temperature: Generation temperature (0.0 = deterministic, 1.0 = random)
            max_tokens: Maximum tokens to generate for the move
            timeout: API call timeout in seconds
            retry_attempts: Number of retry attempts for failed API calls
            retry_delay: Delay between retry attempts in seconds
            fallback_behavior: What to do when no valid move is found ("random_move" or "resign")
            **kwargs: Additional vLLM API parameters
        """
        # Set up vLLM client (OpenAI-compatible)
        # vLLM doesn't require an API key, but the OpenAI client requires one
        self.base_url = base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8080/v1")
        self.api_key = "not-needed"  # vLLM ignores this, but OpenAI client requires it
        
        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        
        # Load configuration from environment variables if not provided
        self.model = model or os.environ.get("VLLM_MODEL", "kunhunjon/ChessLM_Qwen3_Trainium_AWS_Format")
        self.temperature = temperature if temperature is not None else float(os.environ.get("VLLM_TEMPERATURE", "0.1"))
        self.max_tokens = max_tokens or int(os.environ.get("VLLM_MAX_TOKENS", "2048"))
        self.timeout = timeout or float(os.environ.get("VLLM_TIMEOUT", "30.0"))
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Validate and set fallback behavior
        if fallback_behavior not in ["random_move", "resign"]:
            raise ValueError(
                f"Invalid fallback_behavior: {fallback_behavior}. "
                "Must be 'random_move' or 'resign'"
            )
        self.fallback_behavior = fallback_behavior
        
        # Load fallback behavior from environment variable if not provided
        if fallback_behavior is None:
            env_fallback = os.environ.get("OPENAI_FALLBACK_BEHAVIOR")
            if env_fallback:
                if env_fallback not in ["random_move", "resign"]:
                    raise ValueError(
                        f"Invalid OPENAI_FALLBACK_BEHAVIOR: {env_fallback}. "
                        "Must be 'random_move' or 'resign'"
                    )
                self.fallback_behavior = env_fallback
        
        # vLLM generation parameters
        self.generation_params: Dict[str, Any] = {}
        
        # Add temperature if specified
        if self.temperature is not None:
            self.generation_params["temperature"] = self.temperature
        
        # Add other parameters from kwargs
        self.generation_params.update(kwargs)
        
        # Use max_completion_tokens for the API
        if self.max_tokens is not None:
            self.generation_params["max_completion_tokens"] = self.max_tokens
        
        # ------------------------------------------------------------------
        # Qwen3: Disable "thinking" mode on every request via chat_template_kwargs
        # ------------------------------------------------------------------
        extra_body = self.generation_params.get("extra_body") or {}
        chat_kwargs = extra_body.get("chat_template_kwargs") or {}
        # Hard switch: always disable thinking for this agent
        chat_kwargs["enable_thinking"] = False
        extra_body["chat_template_kwargs"] = chat_kwargs
        self.generation_params["extra_body"] = extra_body
        # ------------------------------------------------------------------
        
        # Prompt template
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        
        # Validate prompt template has required placeholders
        self._validate_prompt_template()
    
    def _validate_prompt_template(self):
        """Validate that the prompt template is valid and contains at least basic placeholders."""
        # Check for basic template validity
        if not self.prompt_template or not isinstance(self.prompt_template, str):
            raise ValueError("Prompt template must be a non-empty string")
        
        # Check for at least one placeholder to ensure it's a template
        if "{" not in self.prompt_template or "}" not in self.prompt_template:
            raise ValueError("Prompt template must contain at least one placeholder (e.g., {FEN})")
        
        # Check for balanced braces
        open_braces = self.prompt_template.count("{")
        close_braces = self.prompt_template.count("}")
        if open_braces != close_braces:
            raise ValueError("Prompt template has unbalanced braces - check for missing { or }")
        
        # Optional: Check for common issues with placeholder syntax
        import re
        placeholder_pattern = r'\{[^}]*\}'
        placeholders = re.findall(placeholder_pattern, self.prompt_template)
        
        # Warn about potentially problematic placeholders (but don't fail)
        for placeholder in placeholders:
            if placeholder in ["{}", "{ }", "{  }"]:
                print(f"Warning: Empty placeholder '{placeholder}' found in template")
        
        # Note: We don't require specific placeholders anymore - users can create custom templates
        # with only the variables they need
    
    def _format_prompt(
        self,
        board: chess.Board,
        legal_moves: List[chess.Move],
        move_history: List[str],
        side_to_move: str,
    ) -> str:
        """
        Format the prompt template with actual game data.
        
        Args:
            board: Current chess board state
            legal_moves: List of legal moves available
            move_history: List of moves played so far
            side_to_move: Which side is to move ('White' or 'Black')
            
        Returns:
            Formatted prompt string
        """
        # Get FEN representation
        fen = board.fen()
        
        # Get UTF board representation with Unicode chess pieces
        board_utf = self._render_board_unicode(board)
        
        # Get last move description
        if board.move_stack:
            last_move = board.move_stack[-1]
            # We need to get the SAN before the move was made
            # Create a temporary board to get the SAN
            temp_board = chess.Board()
            for move in board.move_stack[:-1]:
                temp_board.push(move)
            last_move_san = temp_board.san(last_move)
            last_side = "Black" if board.turn else "White"
            last_move_desc = f"{last_side} played {last_move_san}"
        else:
            last_move_desc = "(start of game)"
        
        # Format legal moves as both UCI and SAN lists
        legal_moves_uci = [move.uci() for move in legal_moves]
        legal_moves_san = [board.san(move) for move in legal_moves]
        legal_moves_uci_str = ", ".join(legal_moves_uci)
        legal_moves_san_str = ", ".join(legal_moves_san)
        
        # Format move history as both UCI and SAN
        if move_history:
            # Move history is already in UCI format
            move_history_uci_str = " ".join(move_history)
            
            # Convert UCI moves to SAN if possible
            try:
                history_board = chess.Board()
                move_history_san = []
                for uci_move in move_history:
                    try:
                        move = chess.Move.from_uci(uci_move)
                        san = history_board.san(move)
                        move_history_san.append(san)
                        history_board.push(move)
                    except Exception:
                        move_history_san.append(uci_move)
                
                move_history_san_str = " ".join(move_history_san)
            except Exception:
                move_history_san_str = " ".join(move_history)
        else:
            move_history_uci_str = "(no moves yet)"
            move_history_san_str = "(no moves yet)"
        
        # Format the prompt safely, handling missing placeholders
        try:
            prompt = self.prompt_template.format(
                board_utf=board_utf,
                FEN=fen,
                last_move=last_move_desc,
                legal_moves_uci=legal_moves_uci_str,
                legal_moves_san=legal_moves_san_str,
                move_history_uci=move_history_uci_str,
                move_history_san=move_history_san_str,
                side_to_move=side_to_move
            )
        except KeyError as e:
            # Handle missing placeholders gracefully
            missing_key = str(e).strip("'")
            print(f"Warning: Prompt template references placeholder '{missing_key}' that is not available")
            print("Available placeholders: board_utf, FEN, last_move, legal_moves_uci, legal_moves_san, move_history_uci, move_history_san, side_to_move")
            print("Consider updating your template or using the default template")
            
            # Fall back to a minimal template that should always work
            fallback_template = """You are a chess engine.

Given the current position and legal moves, choose the single best move.

Legal moves (UCI): {legal_moves_uci}

Output the answer in this exact format and nothing else:

<uci_move>MOVE</uci_move>

Where MOVE is one of the legal moves listed above.
Do NOT include any explanation, commentary, or additional text before or after the tags.
"""
            
            prompt = fallback_template.format(legal_moves_uci=legal_moves_uci_str)
        
        return prompt
    
    def _render_board_unicode(self, board: chess.Board) -> str:
        """
        Render the chess board using Unicode chess pieces.
        
        Args:
            board: The chess board to render
            
        Returns:
            String representation of the board with Unicode pieces
        """
        lines = []
        
        # Board coordinates
        files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        ranks = ['8', '7', '6', '5', '4', '3', '2', '1']
        
        # Add top coordinate line with proper alignment
        coord_parts = []
        for file in files:
            coord_parts.append(f" {file} ")  # 3-character spacing to match board squares
        coord_line = "   " + "".join(coord_parts) + "  "
        lines.append(coord_line)
        # Calculate border width: 8 squares × 3 characters each = 24 characters
        border_width = len(files) * 3
        lines.append("   +" + "-" * border_width + "+")
        
        # Render board squares
        for rank in ranks:
            line_parts = []
            
            # Add rank coordinate
            line_parts.append(f"{rank} |")
            
            # Add squares
            for file in files:
                square = chess.parse_square(file + rank)
                piece = board.piece_at(square)
                
                # Get piece symbol or empty square character
                if piece is None:
                    piece_char = "·"  # Empty square
                else:
                    piece_char = self.UNICODE_PIECES[piece.symbol()]
                
                # Format square
                square_str = f" {piece_char} "
                line_parts.append(square_str)
            
            # Add closing coordinate
            line_parts.append(f"| {rank}")
            lines.append("".join(line_parts))
        
        # Add bottom coordinate line
        lines.append("   +" + "-" * border_width + "+")
        coord_line = "   " + "".join(coord_parts) + "  "
        lines.append(coord_line)
        
        return "\n".join(lines)
    
    def _call_vllm_api(self, prompt: str) -> str:
        """
        Call vLLM server to get the model's response.
        
        Args:
            prompt: Formatted prompt to send to the model
            
        Returns:
            Model's response text
            
        Raises:
            Exception: If API call fails after all retry attempts
        """
        last_error = None
        
        for attempt in range(self.retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    **self.generation_params,
                    timeout=self.timeout
                )
                
                # Extract the response content
                if response.choices and response.choices[0].message:
                    message = response.choices[0].message
                    content = (message.content or "").strip()
                    
                    if not content:
                        raise ValueError("vLLM API returned empty content")
                    
                    # Expected format: <uci_move>e2e4</uci_move>
                    return content
                else:
                    raise ValueError("Empty response from vLLM API")
                    
            except Exception as e:
                last_error = e
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    break
        
        # If we get here, all retry attempts failed
        raise Exception(f"vLLM API call failed after {self.retry_attempts} attempts: {last_error}")
    
    def _parse_move(self, response: str, legal_moves: List[chess.Move], board: chess.Board) -> chess.Move:
        """
        Parse the model's response to extract a valid move.

        This method tries multiple parsing strategies:
        1. Look for <uci_move>...</uci_move> tags (preferred)
        2. Look for UCI move with common suffixes (e.g., "e2e4MOVE", "e2e4Move")
        3. Look for any valid UCI move anywhere in the response

        Args:
            response: Raw response from the model
            legal_moves: List of legal moves available
            board: Current chess board state

        Returns:
            The chosen chess move

        Raises:
            ValueError: If no valid move can be parsed
        """
        # Clean the response
        response = response.strip()

        import re

        # Strategy 1: Extract UCI move from <uci_move> tags (preferred format)
        uci_pattern = r'<uci_move>(.*?)</uci_move>'
        uci_matches = re.findall(uci_pattern, response, re.IGNORECASE | re.DOTALL)

        if uci_matches:
            uci_move = uci_matches[0].strip()

            # Check for explicit resignation
            if uci_move.lower() == "resign":
                raise ValueError("Model chose to resign")

            # Try to parse the UCI move
            try:
                move = chess.Move.from_uci(uci_move)
                if move in legal_moves:
                    return move
                else:
                    raise ValueError(f"Model provided illegal move '{uci_move}'")
            except Exception:
                raise ValueError(f"Model provided invalid UCI format '{uci_move}'")

        # Strategy 2: Look for UCI move with common suffixes
        # Handles cases like "g8h6MOVE", "e2e4Move", "d2d4move", etc.
        suffix_pattern = r'([a-h][1-8][a-h][1-8][qrbn]?)(?:MOVE|Move|move|[:\s]|$)'
        suffix_matches = re.findall(suffix_pattern, response, re.IGNORECASE)

        for potential_move in suffix_matches:
            try:
                move = chess.Move.from_uci(potential_move.lower())
                if move in legal_moves:
                    return move
            except Exception:
                continue

        # Strategy 3: Look for any valid UCI move anywhere in the response
        # This is the most permissive pattern, without word boundaries
        uci_move_pattern = r'([a-h][1-8][a-h][1-8][qrbn]?)'
        potential_moves = re.findall(uci_move_pattern, response, re.IGNORECASE)

        for potential_move in potential_moves:
            try:
                move = chess.Move.from_uci(potential_move.lower())
                if move in legal_moves:
                    return move
            except Exception:
                continue

        # No valid move found
        raise ValueError("Model did not respond with a valid UCI move")
    
    def _extract_comment(self, response: str) -> str:
        """
        Extract the full comment from the model's response.
        
        For this strict setup, the response is usually just <uci_move>...</uci_move>,
        but we keep this method in case you later relax the prompt.
        
        Args:
            response: Raw response from the model
            
        Returns:
            Full response as comment string
        """
        return response.strip()
    
    def choose_move(
        self,
        board: chess.Board,
        legal_moves: List[chess.Move],
        move_history: List[str],
        side_to_move: str,
    ) -> tuple[chess.Move | None, str | None]:
        """
        Choose the best move using the vLLM server.
        
        Args:
            board: Current chess board state
            legal_moves: List of legal moves available
            move_history: List of moves played so far (in UCI notation)
            side_to_move: Which side is to move ('White' or 'Black')
            
        Returns:
            Tuple of (chosen_move, optional_comment)
            - chosen_move: The chosen chess move, or None to resign
            - optional_comment: Comment from the AI model explaining the move or resignation
            
        Raises:
            ValueError: If no legal moves are available
        """
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        # Format the prompt
        prompt = self._format_prompt(board, legal_moves, move_history, side_to_move)
        
        # Call vLLM API
        try:
            response = self._call_vllm_api(prompt)
        except Exception as e:
            # If API call fails, fall back to first legal move
            print(f"Warning: vLLM API call failed: {e}, using first legal move")
            return legal_moves[0], f"FALLBACK MOVE - vLLM API failed: {e}"
        
        # Parse the response to get the move
        try:
            move = self._parse_move(response, legal_moves, board)
            # Use the full API response as the comment
            comment = self._extract_comment(response)
            return move, comment
        except ValueError as e:
            # Check if the model explicitly chose to resign
            if "resign" in str(e).lower():
                # Model explicitly chose to resign - always respect this choice
                return None, f"RESIGNATION - Model chose to resign. Full API response: {response}"
            
            # If parsing fails due to invalid/unparseable moves, handle according to fallback behavior
            if self.fallback_behavior == "resign":
                # Parsing failed and fallback behavior is resign
                return None, f"RESIGNATION - Unable to parse valid move: {e}. Full API response: {response}"
            else:
                # Parsing failed but fallback behavior is random_move - select a random legal move
                import random
                random_move = random.choice(legal_moves)
                print(f"Warning: Could not parse move from response: {e}, using random legal move")
                return random_move, f"RANDOM MOVE - Unable to parse move: {e}. Full API response: {response}"
    
    def update_prompt_template(self, new_template: str):
        """
        Update the prompt template.
        
        You can create custom templates with any combination of available placeholders:
        - {board_utf}: Visual board representation with Unicode pieces
        - {FEN}: FEN notation of the current position
        - {side_to_move}: Which side is to move ('White' or 'Black')
        - {legal_moves_uci}: Available moves in UCI notation
        - {legal_moves_san}: Available moves in SAN notation
        - {move_history_uci}: Game history in UCI notation
        - {move_history_san}: Game history in SAN notation
        - {last_move}: Description of the last move played
        
        For reliable parsing, ensure your template still instructs the model to
        output exactly: <uci_move>MOVE</uci_move>
        
        Args:
            new_template: New prompt template string with desired placeholders
        """
        self.prompt_template = new_template
        self._validate_prompt_template()
    
    def update_generation_params(self, **kwargs):
        """
        Update generation parameters.
        
        Args:
            **kwargs: New generation parameters
        """
        # Handle token parameter updates
        if "max_tokens" in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        
        self.generation_params.update(kwargs)
        
        # Re-apply strict thinking disable for Qwen3
        extra_body = self.generation_params.get("extra_body") or {}
        chat_kwargs = extra_body.get("chat_template_kwargs") or {}
        chat_kwargs["enable_thinking"] = False
        extra_body["chat_template_kwargs"] = chat_kwargs
        self.generation_params["extra_body"] = extra_body
        
        # Update instance variables for commonly used params
        if "temperature" in kwargs:
            self.temperature = kwargs["temperature"]
        if "max_completion_tokens" in kwargs:
            self.max_tokens = kwargs["max_completion_tokens"]
    
    def update_fallback_behavior(self, behavior: str):
        """
        Update the fallback behavior.
        
        Args:
            behavior: New fallback behavior ("random_move" or "resign")
            
        Raises:
            ValueError: If behavior is invalid
        """
        if behavior not in ["random_move", "resign"]:
            raise ValueError(
                f"Invalid fallback_behavior: {behavior}. "
                "Must be 'random_move' or 'resign'"
            )
        self.fallback_behavior = behavior
    
    def get_prompt_template(self) -> str:
        """Get the current prompt template."""
        return self.prompt_template
    
    def get_generation_params(self) -> Dict[str, Any]:
        """Get the current generation parameters."""
        return self.generation_params.copy()
    
    def get_fallback_behavior(self) -> str:
        """Get the current fallback behavior."""
        return self.fallback_behavior
    
    def test_connection(self) -> bool:
        """
        Test the vLLM server connection.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Use minimal parameters for connection test
            test_params = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "timeout": 10,
                "max_completion_tokens": 5,
                "extra_body": {
                    "chat_template_kwargs": {
                        "enable_thinking": False
                    }
                },
            }
            
            response = self.client.chat.completions.create(**test_params)
            return True
        except Exception as e:
            print(f"vLLM server connection test failed: {e}")
            print(f"Make sure vLLM server is running at {self.base_url}")
            return False
