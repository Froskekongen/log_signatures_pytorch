import pytest
import torch
from log_signatures_pytorch.signature import windowed_signature
from log_signatures_pytorch.log_signature import windowed_log_signature

class TestWindowedErrors:
    def test_window_size_too_small(self):
        path = torch.randn(1, 10, 2)
        with pytest.raises(ValueError, match="window_size must be at least 2"):
            windowed_signature(path, depth=2, window_size=1, hop_size=1)

    def test_window_size_larger_than_path(self):
        path = torch.randn(1, 5, 2)
        with pytest.raises(ValueError, match="window_size cannot exceed the path length"):
            windowed_signature(path, depth=2, window_size=6, hop_size=1)

    def test_hop_size_non_positive(self):
        path = torch.randn(1, 10, 2)
        with pytest.raises(ValueError, match="hop_size must be positive"):
            windowed_signature(path, depth=2, window_size=5, hop_size=0)
            
    def test_hop_size_negative(self):
        path = torch.randn(1, 10, 2)
        with pytest.raises(ValueError, match="hop_size must be positive"):
            windowed_signature(path, depth=2, window_size=5, hop_size=-1)

    def test_windowed_log_signature_propagates_errors(self):
        # Verify that windowed_log_signature also performs these checks (since it calls windowed_signature)
        path = torch.randn(1, 5, 2)
        with pytest.raises(ValueError, match="window_size cannot exceed the path length"):
            windowed_log_signature(path, depth=2, window_size=6, hop_size=1)

    def test_minimal_valid_window(self):
        # Test minimal window size of 2
        path = torch.randn(1, 4, 2)
        # Should not raise
        sig = windowed_signature(path, depth=2, window_size=2, hop_size=1)
        assert sig.shape[1] == 3 # 1 + (4-2)//1 = 3 windows
        
    def test_disjoint_windows(self):
        # Test hop_size >= window_size
        path = torch.randn(1, 10, 2)
        # Windows: [0,1,2], [3,4,5], [6,7,8] ... (indices)
        # If size=3, hop=3
        sig = windowed_signature(path, depth=2, window_size=3, hop_size=3)
        # Num windows = 1 + (10-3)//3 = 1 + 2 = 3
        assert sig.shape[1] == 3
