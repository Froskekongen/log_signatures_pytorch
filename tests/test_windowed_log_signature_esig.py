import pytest
import torch
import numpy as np

# Skip if esig is not installed
esig = pytest.importorskip("esig")

from log_signatures_pytorch.log_signature import windowed_log_signature

def _esig_windowed_ref(path, depth, window_size, hop_size):
    """
    Manually slice path and compute log-signature using esig for each window.
    """
    # Path: (length, dim)
    n_points = path.shape[0]
    results = []
    
    # Generate slice indices
    # Matches the logic: num_windows = 1 + (length - window_size) // hop_size
    # start indices: 0, hop, 2*hop, ...
    
    num_windows = 1 + (n_points - window_size) // hop_size
    
    for i in range(num_windows):
        start = i * hop_size
        end = start + window_size
        
        # Ensure we don't go out of bounds (though range calculation should prevent this)
        if end > n_points:
            break
            
        window_path = path[start:end].numpy()
        
        # Compute log-signature for this window
        ls = esig.stream2logsig(window_path, depth)
        results.append(ls)
        
    if not results:
        return torch.empty(0)
        
    return torch.tensor(np.array(results)) # (num_windows, logsig_dim)

@pytest.mark.external
@pytest.mark.parametrize("window_size, hop_size", [(10, 2), (10, 5), (10, 10)])
def test_windowed_log_signature_vs_esig_manual_unfold(window_size, hop_size):
    torch.manual_seed(42)
    B, L, D = 2, 50, 2
    depth = 3
    
    # Create random path (cumulative sum of increments)
    path = torch.randn(B, L, D, dtype=torch.float64).cumsum(dim=1)
    
    # 1. Compute using our optimized implementation
    # Note: match esig's Hall basis
    ours = windowed_log_signature(
        path, depth=depth, window_size=window_size, hop_size=hop_size, mode="hall"
    )
    
    # 2. Compute using explicit loop over esig
    expected_list = []
    for i in range(B):
        ref = _esig_windowed_ref(path[i], depth, window_size, hop_size)
        expected_list.append(ref)
    expected = torch.stack(expected_list)
    
    # Check shape first
    assert ours.shape == expected.shape
    
    # Check values
    torch.testing.assert_close(ours, expected, atol=1e-5, rtol=1e-5)
