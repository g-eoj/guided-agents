"""Pytest configuration and fixtures."""

import sys
from unittest.mock import MagicMock


# Mock MLX modules that are Apple Silicon specific
mock_mlx = MagicMock()
mock_mlx.core = MagicMock()
mock_mlx.nn = MagicMock()

sys.modules['mlx'] = mock_mlx
sys.modules['mlx.core'] = mock_mlx.core
sys.modules['mlx.nn'] = mock_mlx.nn
sys.modules['mlx_lm'] = MagicMock()
sys.modules['mlx_vlm'] = MagicMock()
sys.modules['mlx_whisper'] = MagicMock()

# Mock other optional dependencies that may not be installed
sys.modules['openai'] = MagicMock()
sys.modules['xgrammar'] = MagicMock()
sys.modules['transformers'] = MagicMock()

# Mock xgrammar.kernels.apply_token_bitmask_mlx
mock_xgrammar = MagicMock()
mock_kernels = MagicMock()
mock_apply_token_bitmask_mlx = MagicMock()
mock_kernels.apply_token_bitmask_mlx = mock_apply_token_bitmask_mlx
mock_xgrammar.kernels = mock_kernels
sys.modules['xgrammar.kernels.apply_token_bitmask_mlx'] = mock_apply_token_bitmask_mlx
