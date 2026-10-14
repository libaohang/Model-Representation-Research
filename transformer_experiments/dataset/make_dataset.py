from .sample import sample_balanced, split_cube, save_npz
import time

n = 300000
t0 = time.time()
cube = sample_balanced(total_samples=n, seed=1)
parts1 = split_cube(cube, {"trainGPT": 0.8, "holdout": 0.2}, seed=1)
dt = time.time() - t0
print(f"Sampled {n} records in {dt:.2f}s\n")

save_npz(parts1["trainGPT"],  "data/trainGPT.npz")
print(f"Saved training data for GPT\n")

parts2 = split_cube(parts1["holdout"], {"valGPT_trainProbe": 0.5, "holdout": 0.5}, seed=1)
save_npz(parts2["valGPT_trainProbe"],  "data/valGPT_trainProbe.npz")
print(f"Saved validation data for GPT / training data for probe\n")

save_npz(parts2["holdout"],  "data/testGPT.npz")
print(f"Saved final test data for GPT\n")

parts3 = split_cube(parts2["holdout"], {"valProbe": 0.5, "testProbe": 0.5}, seed=1)
save_npz(parts3["valProbe"],  "data/valProbe.npz")
print(f"Saved validation data for probe\n")

save_npz(parts3["testProbe"],  "data/testProbe.npz")
print(f"Saved final testing data for probe\n")