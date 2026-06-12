import numpy as np
import awkward as ak
import vector
from ntupelizer.tools import general as g
from ntupelizer.scripts import preprocess_torch as p

def test_reinitialize_p4_optimization():
    """
    Verify that reinitialize_p4 correctly maps different coordinate names
    and produces a vector object that supports standard kinematic properties.
    This tests the fix where vector.zip was standardized to recognize MomentumArray4D.
    """
    # Sample data with 'rho', 'phi', 'eta', 't' (common in the parquet files)
    data = ak.Array([
        {"rho": 10.0, "phi": 0.5, "eta": 1.0, "t": 20.0},
        {"rho": 20.0, "phi": -0.5, "eta": -1.0, "t": 40.0}
    ])
    
    p4 = g.reinitialize_p4(data)
    
    # Check that it is a vector array
    assert isinstance(p4, vector.backends.awkward.MomentumArray4D)
    
    # Check values
    assert np.allclose(ak.to_numpy(p4.pt), [10.0, 20.0])
    assert np.allclose(ak.to_numpy(p4.phi), [0.5, -0.5])
    assert np.allclose(ak.to_numpy(p4.eta), [1.0, -1.0])
    assert np.allclose(ak.to_numpy(p4.energy), [20.0, 40.0])
    
    # Check that 'rho' and 't' were correctly mapped
    assert hasattr(p4, "pt")
    assert hasattr(p4, "energy")

def test_stack_and_pad_features_optimization():
    """
    Verify that stack_and_pad_features correctly pads and clips jagged arrays.
    This tests the use of ak.to_regular for efficient padding.
    """
    # Jagged array
    features = ak.Array({
        "f1": [[1.0, 2.0], [3.0], []],
        "f2": [[0.1, 0.2], [0.3], []]
    })
    
    max_cands = 2
    result = p.stack_and_pad_features(features, max_cands)
    
    # Expected shape: [N=3, max_cands=2, n_fields=2]
    assert result.shape == (3, 2, 2)
    
    # Check padding (0.0) and clipping
    # Row 0: [1.0, 2.0], [0.1, 0.2]
    assert np.allclose(result[0, :, 0], [1.0, 2.0])
    assert np.allclose(result[0, :, 1], [0.1, 0.2])
    
    # Row 1: [3.0, 0.0], [0.3, 0.0]
    assert np.allclose(result[1, :, 0], [3.0, 0.0])
    assert np.allclose(result[1, :, 1], [0.3, 0.0])
    
    # Row 2: [0.0, 0.0], [0.0, 0.0]
    assert np.allclose(result[2, :, 0], [0.0, 0.0])
    assert np.allclose(result[2, :, 1], [0.0, 0.0])

if __name__ == "__main__":
    test_reinitialize_p4_optimization()
    test_stack_and_pad_features_optimization()
    print("Optimization tests passed!")
