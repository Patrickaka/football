import math
from collections import Counter, defaultdict
from typing import List, Dict, Tuple


def chi_square_test(numbers_list: List[List[int]]) -> Dict:
    total_draws = len(numbers_list)
    total_numbers = sum(len(nums) for nums in numbers_list)

    freq = Counter()
    for nums in numbers_list:
        for num in nums:
            freq[num] += 1

    expected = total_numbers / 80
    chi2 = 0
    for num in range(1, 81):
        observed = freq.get(num, 0)
        chi2 += (observed - expected) ** 2 / expected

    df = 79
    p_value = 1 - _chi2_cdf(chi2, df)

    return {
        'test': 'chi_square_frequency',
        'chi2': round(chi2, 4),
        'df': df,
        'p_value': round(p_value, 6),
        'expected_per_number': round(expected, 2),
        'min_freq': min(freq.values()),
        'max_freq': max(freq.values()),
    }


def runs_test(numbers_list: List[List[int]]) -> Dict:
    present = []
    for nums in numbers_list:
        row = [1 if num in nums else 0 for num in range(1, 81)]
        present.append(row)

    runs_results = []
    for num in range(1, 81):
        sequence = [row[num - 1] for row in present]
        n1 = sum(sequence)
        n0 = len(sequence) - n1

        if n1 == 0 or n0 == 0:
            runs_results.append({'num': num, 'runs': 0, 'p_value': 1.0})
            continue

        runs = 1
        for i in range(1, len(sequence)):
            if sequence[i] != sequence[i - 1]:
                runs += 1

        expected_runs = (2 * n1 * n0) / (n1 + n0) + 1
        variance = (2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / ((n1 + n0) ** 2 * (n1 + n0 - 1))
        std_dev = math.sqrt(variance)

        if std_dev == 0:
            z = 0
        else:
            z = (runs - expected_runs) / std_dev

        p_value = 2 * (1 - _norm_cdf(abs(z)))
        runs_results.append({'num': num, 'runs': runs, 'z': z, 'p_value': p_value})

    avg_p = sum(r['p_value'] for r in runs_results) / len(runs_results)
    significant_count = sum(1 for r in runs_results if r['p_value'] < 0.05)

    return {
        'test': 'runs_test',
        'avg_p_value': round(avg_p, 4),
        'significant_at_0.05': significant_count,
        'total_tests': len(runs_results),
        'details': runs_results,
    }


def ljung_box_test(numbers_list: List[List[int]], max_lag: int = 20) -> Dict:
    results = {}

    for lag in range(1, max_lag + 1):
        autocorr_sum = 0
        total_draws = len(numbers_list)

        for num in range(1, 81):
            present = [1 if num in nums else 0 for nums in numbers_list]
            n = len(present)

            mean = sum(present) / n
            var = sum((x - mean) ** 2 for x in present) / n

            if var == 0:
                continue

            autocorr = 0
            for i in range(n - lag):
                autocorr += (present[i] - mean) * (present[i + lag] - mean)
            autocorr /= (n * var)
            autocorr_sum += autocorr ** 2

        q_stat = total_draws * (total_draws + 2) * autocorr_sum / (total_draws - lag)
        df = 80 * lag
        p_value = 1 - _chi2_cdf(q_stat, df)

        results[f'lag_{lag}'] = {
            'q_stat': round(q_stat, 4),
            'df': df,
            'p_value': round(p_value, 6),
        }

    return {
        'test': 'ljung_box_autocorrelation',
        'max_lag': max_lag,
        'results': results,
    }


def pair_cooccurrence_test(numbers_list: List[List[int]]) -> Dict:
    total_draws = len(numbers_list)
    pair_freq = Counter()

    for nums in numbers_list:
        nums_sorted = sorted(nums)
        for i in range(len(nums_sorted)):
            for j in range(i + 1, len(nums_sorted)):
                pair_freq[(nums_sorted[i], nums_sorted[j])] += 1

    total_pairs = sum(pair_freq.values())
    num_pairs = len(pair_freq)
    expected = total_pairs / num_pairs if num_pairs > 0 else 0

    chi2 = 0
    for pair, count in pair_freq.items():
        chi2 += (count - expected) ** 2 / expected

    df = num_pairs - 1 if num_pairs > 1 else 0
    p_value = 1 - _chi2_cdf(chi2, df) if df > 0 else 1.0

    return {
        'test': 'pair_cooccurrence',
        'chi2': round(chi2, 4),
        'df': df,
        'p_value': round(p_value, 6),
        'total_pairs': num_pairs,
        'expected_per_pair': round(expected, 2),
        'min_cooccurrence': min(pair_freq.values()),
        'max_cooccurrence': max(pair_freq.values()),
    }


def zone_distribution_test(numbers_list: List[List[int]], zone_size: int = 10) -> Dict:
    num_zones = 80 // zone_size
    zone_freq = Counter()

    for nums in numbers_list:
        for num in nums:
            zone = (num - 1) // zone_size + 1
            zone_freq[zone] += 1

    total = sum(zone_freq.values())
    expected = total / num_zones

    chi2 = 0
    for zone in range(1, num_zones + 1):
        observed = zone_freq.get(zone, 0)
        chi2 += (observed - expected) ** 2 / expected

    df = num_zones - 1
    p_value = 1 - _chi2_cdf(chi2, df)

    return {
        'test': f'zone_distribution_{zone_size}',
        'chi2': round(chi2, 4),
        'df': df,
        'p_value': round(p_value, 6),
        'expected_per_zone': round(expected, 2),
        'zone_freq': dict(zone_freq),
    }


def _chi2_cdf(x: float, df: int) -> float:
    if x <= 0:
        return 0.0
    try:
        from scipy.stats import chi2
        return chi2.cdf(x, df)
    except ImportError:
        k = df / 2.0
        t = x / 2.0
        if df == 1:
            return math.erf(math.sqrt(t)) / 2 + 0.5
        return _gamma_lower_incomplete_iterative(k, t) / math.gamma(k)


def _gamma_lower_incomplete_iterative(s: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if s <= 0:
        return _gamma_lower_incomplete_iterative(s + 1, x) / x

    result = 0.0
    term = math.exp(-x) * x ** s / math.gamma(s + 1)
    result += term

    for n in range(1, 1000):
        term *= x / (s + n)
        result += term
        if abs(term) < 1e-15:
            break

    return result * math.gamma(s)


def _norm_cdf(x: float) -> float:
    return (1 + math.erf(x / math.sqrt(2))) / 2


def run_all_diagnostics(numbers_list: List[List[int]]) -> Dict:
    results = {}

    results['chi_square'] = chi_square_test(numbers_list)
    results['runs_test'] = runs_test(numbers_list)
    results['ljung_box'] = ljung_box_test(numbers_list)
    results['pair_cooccurrence'] = pair_cooccurrence_test(numbers_list)
    results['zone_10'] = zone_distribution_test(numbers_list, zone_size=10)
    results['zone_8'] = zone_distribution_test(numbers_list, zone_size=8)

    return results