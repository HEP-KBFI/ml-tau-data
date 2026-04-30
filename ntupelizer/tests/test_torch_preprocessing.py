import os
import numpy as np
import pandas as pd
import torch

def test_tau_decaymode_charge_kinematics_distributions():
	"""
	Test that the distributions of tau decay modes, charges, and kinematic variables
	are preserved between the raw parquet and preprocessed .pt files.
	Update file paths and field mappings as needed.
	"""
	parquet_path = os.path.join(os.path.dirname(__file__), "test_data.parquet")
	pt_path = os.path.join(os.path.dirname(__file__), "test_data.pt")

	# Load parquet
	df = pd.read_parquet(parquet_path)

	# Load torch tensor
	data = torch.load(pt_path)
	# If your .pt file is a dict or tuple, adjust accordingly
	if isinstance(data, dict):
		tensor_data = data
	else:
		raise ValueError(".pt file must be a dict with named fields matching preprocessing output")

	# --- Tau decay mode distribution ---
	parquet_decay = df["gen_jet_tau_decaymode"].values
	pt_decay = tensor_data["tau_decaymode"].cpu().numpy()
	parquet_counts = np.bincount(parquet_decay)
	pt_counts = np.bincount(pt_decay)
	assert np.allclose(parquet_counts, pt_counts, rtol=0, atol=1), "Tau decay mode counts differ"

	# --- Tau charge distribution ---
	parquet_charge = df["gen_jet_tau_charge"].values
	pt_charge = tensor_data["tau_charge"].cpu().numpy()
	parquet_charge_counts = np.bincount(parquet_charge + 1)  # shift if charges are -1,0,1
	pt_charge_counts = np.bincount(pt_charge + 1)
	assert np.allclose(parquet_charge_counts, pt_charge_counts, rtol=0, atol=1), "Tau charge counts differ"

	# --- Tau kinematic variables (rho, phi, eta, t) ---
	for var in ["rho", "phi", "eta", "t"]:
		parquet_kin = df["gen_jet_tau_p4"].apply(lambda x: x[var] if isinstance(x, dict) else getattr(x, var)).values
		pt_kin = tensor_data[f"tau_{var}"].cpu().numpy()
		# Compare distributions via histogram
		hist_parquet, bins = np.histogram(parquet_kin, bins=50, range=(np.min(parquet_kin), np.max(parquet_kin)))
		hist_pt, _ = np.histogram(pt_kin, bins=bins)
		assert np.allclose(hist_parquet, hist_pt, rtol=0, atol=2), f"Tau {var} distribution differs"
