import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency

# List of indices marked as "bad" (too much degradation)
bad_indices = [2,3,6,9,11,12,17,18,19,26,29,30,35,38,46,50,55,57,66,68,69,70,72,74,75,79,83,90,91,97,99,102,106,108,113,115,118,128,131,133,134,135,138,143,144,146,149,151,152,153,154,155,160,162,164,166,168,169,174,178,185,188,190,195,202,204,211,213,214,215,217,224,229,230,234,235,237,238,239,242,244,246,247]

aug_test_dir = Path('aug_test')

# Load all metadata
all_meta = {}
for meta_file in aug_test_dir.glob('*_meta.json'):
    idx = int(meta_file.stem.split('_')[0])
    with open(meta_file, 'r') as f:
        data = json.load(f)
    all_meta[idx] = data['augs']

bad_indices_set = set(bad_indices)
good_indices = [idx for idx in all_meta.keys() if idx not in bad_indices_set]

print(f"Total images: {len(all_meta)}")
print(f"Bad images: {len(bad_indices)}")
print(f"Good images: {len(good_indices)}")

# Collect data for bad and good
bad_augs = []
good_augs = []
bad_intensities = defaultdict(list)
good_intensities = defaultdict(list)
bad_ns = defaultdict(list)
good_ns = defaultdict(list)

for idx, augs in all_meta.items():
    if idx in bad_indices_set:
        for aug in augs:
            bad_augs.append(aug['name'])
            bad_intensities[aug['name']].append(aug['intensity'])
            # collect numeric 'n' parameter when available
            n = aug.get('params', {}).get('n', None)
            if n is not None:
                bad_ns[aug['name']].append(n)
    else:
        for aug in augs:
            good_augs.append(aug['name'])
            good_intensities[aug['name']].append(aug['intensity'])
            n = aug.get('params', {}).get('n', None)
            if n is not None:
                good_ns[aug['name']].append(n)

bad_counts = Counter(bad_augs)
good_counts = Counter(good_augs)

all_aug_names = set(bad_counts.keys()) | set(good_counts.keys())

print("\nAugmentation analysis:")
print("Name | Bad Freq | Good Freq | Diff | Bad Avg Int | Good Avg Int | Int Diff | Bad Avg N | Good Avg N | N Diff | p-value")
print("-" * 120)

correlations = []

for aug in sorted(all_aug_names):
    bad_freq = bad_counts.get(aug, 0) / len(bad_indices) if bad_indices else 0
    good_freq = good_counts.get(aug, 0) / len(good_indices) if good_indices else 0
    freq_diff = bad_freq - good_freq
    
    bad_ints = bad_intensities.get(aug, [])
    good_ints = good_intensities.get(aug, [])
    
    bad_avg = np.mean(bad_ints) if bad_ints else 0
    good_avg = np.mean(good_ints) if good_ints else 0
    int_diff = bad_avg - good_avg
    # numeric 'n' analysis
    bad_n_list = bad_ns.get(aug, [])
    good_n_list = good_ns.get(aug, [])
    bad_n_avg = np.mean(bad_n_list) if bad_n_list else 0
    good_n_avg = np.mean(good_n_list) if good_n_list else 0
    n_diff = bad_n_avg - good_n_avg
    
    # Statistical test for intensity difference
    if len(bad_ints) > 1 and len(good_ints) > 1:
        t_stat, p_val = ttest_ind(bad_ints, good_ints, equal_var=False)
    else:
        p_val = 1.0  # Not significant
    
    print(f"{aug:<25} | {bad_freq:.3f} | {good_freq:.3f} | {freq_diff:+.3f} | {bad_avg:.3f} | {good_avg:.3f} | {int_diff:+.3f} | {bad_n_avg:.2f} | {good_n_avg:.2f} | {n_diff:+.2f} | {p_val:.3f}")
    
    correlations.append((aug, freq_diff, int_diff, p_val))

# Chi-square for overall presence
total_bad = len(bad_indices)
total_good = len(good_indices)
chi_results = []
for aug in all_aug_names:
    present_bad = bad_counts.get(aug, 0)
    absent_bad = total_bad - present_bad
    present_good = good_counts.get(aug, 0)
    absent_good = total_good - present_good
    
    contingency = [[present_bad, absent_bad], [present_good, absent_good]]
    if all(sum(row) > 0 for row in contingency):
        chi2, p, dof, ex = chi2_contingency(contingency)
        chi_results.append((aug, chi2, p))

print("\nChi-square test for augmentation presence (p < 0.05 indicates significant association with bad images):")
for aug, chi2, p in sorted(chi_results, key=lambda x: x[2]):
    sig = "YES" if p < 0.05 else "NO"
    print(f"  {aug}: chi2={chi2:.2f}, p={p:.3f} ({sig})")

# Suggest changes based on correlations
print("\nSuggested adjustments (based on correlations):")
high_corr = sorted(correlations, key=lambda x: (x[1], x[2]), reverse=True)[:5]  # Top by freq and int diff
for aug, freq_diff, int_diff, p_val in high_corr:
    if freq_diff > 0.1 or int_diff > 0.1:
        print(f"- Reduce intensity or frequency of '{aug}' (freq diff: {freq_diff:.3f}, int diff: {int_diff:.3f})")
    if p_val < 0.05:
        print(f"- '{aug}' significantly associated with bad images (p={p_val:.3f})")

# Check for combinations
print("\nCommon augmentation combinations in bad images:")
bad_combos = Counter()
for idx in bad_indices:
    if idx in all_meta:
        combo = tuple(sorted([aug['name'] for aug in all_meta[idx]]))
        bad_combos[combo] += 1

for combo, count in bad_combos.most_common(5):
    if len(combo) > 1:
        print(f"  {combo}: {count} times")