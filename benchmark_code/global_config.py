"""
Global configs for running code of time-series imputation survey.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

# random seeds for five rounds of experiments
RANDOM_SEEDS = [
    2024, 2025, 2026, 2027, 2028,
    2029, 2030, 2031, 2032, 2033,
    2034, 2035, 2036, 2037, 2038,
    2039, 2040, 2041, 2042, 2043,
    2044, 2045, 2046, 2047, 2048,
]
# number of threads for pytorch
TORCH_N_THREADS = 1

# whether to apply lazy-loading strategy (help save memory) to read datasets
LAZY_LOAD_DATA = False
