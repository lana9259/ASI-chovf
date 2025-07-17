import ccxt
import pandas as pd
import numpy as np
import time
import json
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator
import ta
import statistics
from enum import Enum

def generate_dynamic_rr(entry_price, stop_loss, target_price):
    risk = abs(entry_price - stop_loss)
    reward = abs(target_price - entry_price)
    rr_ratio = round(reward / risk, 3) if risk != 0 else None
    return {
        "entry": round(entry_price, 2),
        "sl": round(stop_loss, 2),
        "tp": round(target_price, 2),
        "rr_ratio": rr_ratio
    }

def compute_curvature(prices):
    first_deriv = np.gradient(prices)
    second_deriv = np.gradient(first_deriv)
    return second_deriv

def is_real_time_fractal_peak(i, curvature, threshold=0.001):
    return curvature[i] < -threshold and curvature[i-1] > curvature[i] < curvature[i+1]

def is_real_time_fractal_valley(i, curvature, threshold=0.001):
    return curvature[i] > threshold and curvature[i-1] < curvature[i] > curvature[i+1]

def fibonacci_extension(low_point, high_point, retrace_point, ratio=1.618):
    return retrace_point + (high_point - low_point) * ratio

def is_bullish_engulfing(curr, prev):
    return prev['close'] < prev['open'] and \
           curr['close'] > curr['open'] and \
           curr['close'] > prev['open'] and \
           curr['open'] < prev['close']

def is_bearish_engulfing(curr, prev):
    return prev['close'] > prev['open'] and \
           curr['close'] < curr['open'] and \
           curr['close'] < prev['open'] and \
           curr['open'] > prev['close']

def is_pin_bar(candle):
    body = abs(candle['close'] - candle['open'])
    upper_shadow = candle['high'] - max(candle['close'], candle['open'])
    lower_shadow = min(candle['close'], candle['open']) - candle['low']
    return (upper_shadow > body * 2) or (lower_shadow > body * 2)

def is_doji(candle):
    body = abs(candle['close'] - candle['open'])
    total_range = candle['high'] - candle['low']
    return body < total_range * 0.1

def detect_real_time_pivot(price_series, sensitivity=1.5):
    pivots = [0] * len(price_series)
    for i in range(2, len(price_series) - 2):
        prev_slope = price_series[i] - price_series[i - 1]
        next_slope = price_series[i + 1] - price_series[i]
        curvature = next_slope - prev_slope
        if curvature > sensitivity:
            pivots[i] = -1
        elif curvature < -sensitivity:
            pivots[i] = 1
    return pivots

def ultra_adaptive_ema(prices, length=21, sensitivity=1.0):
    ema = []
    curvature = compute_curvature(prices)
    atr_like = np.std(prices[-length:])
    for i in range(len(prices)):
        weight = min(max(abs(curvature[i]) / (atr_like + 1e-6), 0.01), 1.0)
        alpha = (2 / (length + 1)) * weight * sensitivity
        if i == 0:
            ema.append(prices[0])
        else:
            ema.append(alpha * prices[i] + (1 - alpha) * ema[i-1])
    return np.array(ema)

def ultra_adaptive_hma(prices, length=21, sensitivity=1.0):
    curvature = compute_curvature(prices)
    weight = np.clip(np.abs(curvature) / (np.std(prices[-length:]) + 1e-6), 0.1, 1.0)
    half_len = int(length / 2)
    sqrt_len = int(np.sqrt(length))
    wma1 = np.convolve(prices * weight, np.ones(half_len) / half_len, mode='same')
    wma2 = np.convolve(prices * weight, np.ones(length) / length, mode='same')
    raw_hma = 2 * wma1 - wma2
    hma = np.convolve(raw_hma, np.ones(sqrt_len) / sqrt_len, mode='same')
    return hma

def realtime_peak_valley_v2(price_series, curvature_series, threshold=0.002, window=2):
    pivot_labels = ['none'] * len(price_series)
    for i in range(window, len(price_series) - window):
        left = price_series[i - window:i]
        right = price_series[i + 1:i + 1 + window]
        center = price_series[i]
        if all(center > x for x in left + right) and curvature_series[i] < -threshold:
            pivot_labels[i] = 'high'
        elif all(center < x for x in left + right) and curvature_series[i] > threshold:
            pivot_labels[i] = 'low'
    return pivot_labels

def adaptive_fractal_filter(price_series, window=3, deviation_threshold=0.5):
    pivots = []
    for i in range(window, len(price_series) - window):
        local_range = max(price_series[i - window:i + window + 1]) - min(price_series[i - window:i + window + 1])
        local_std = np.std(price_series[i - window:i + window + 1])
        if local_range > 0 and local_std / local_range < deviation_threshold:
            pivots.append(i)
    return pivots

def validate_wave3_strength(df, wave1, wave3):
    rsi = df['rsi']
    adx = df['adx']
    plus_di = df['plus_di']
    minus_di = df['minus_di']
    
    start1, end1 = wave1['start'], wave1['end']
    start3, end3 = wave3['start'], wave3['end']

    avg_rsi_1 = np.mean(rsi.iloc[start1:end1+1])
    avg_rsi_3 = np.mean(rsi.iloc[start3:end3+1])

    avg_adx_1 = np.mean(adx.iloc[start1:end1+1])
    avg_adx_3 = np.mean(adx.iloc[start3:end3+1])

    strong_trend = all(plus_di.iloc[i] > minus_di.iloc[i] for i in range(start3, end3))

    rsi_trending_up = rsi[end3] > rsi[start3]
    adx_trending_up = adx[end3] > adx[start3]

    return (
        avg_rsi_3 > avg_rsi_1 and
        avg_adx_3 > avg_adx_1 and
        strong_trend and
        rsi_trending_up and
        adx_trending_up
    )

def is_not_impulsive_advanced(w1, w2, w3, w4, w5, rsi, adx):
    w3_vs_w1 = (w3['length'] / w1['length']) if w1['length'] != 0 else 0
    w5_vs_w1 = (w5['length'] / w1['length']) if w1['length'] != 0 else 0
    fib_ok = 1.382 <= w3_vs_w1 <= 2.618 and 0.618 <= w5_vs_w1 <= 1.618
    overlap_ok = w4['low'] > w1['high']
    momentum_ok = adx[w3['end']] > 25
    divergence_ok = rsi[w5['end']] < rsi[w3['end']]
    structure_ok = w1['subwaves'] == 5 and w3['subwaves'] == 5 and w5['subwaves'] == 5
    is_impulse = fib_ok and overlap_ok and momentum_ok and structure_ok
    return not is_impulse

def advanced_price_action_validation(df, i, signal_type="high"):
    if i < 1 or i >= len(df):
        return False
    current = df.iloc[i]
    prev = df.iloc[i - 1]
    bullish_engulfing = prev['close'] < prev['open'] and current['close'] > current['open'] and current['close'] > prev['open'] and current['open'] < prev['close']
    bearish_engulfing = prev['close'] > prev['open'] and current['close'] < current['open'] and current['close'] < prev['open'] and current['open'] > prev['close']
    upper_shadow = current['high'] - max(current['close'], current['open'])
    lower_shadow = min(current['close'], current['open']) - current['low']
    body = abs(current['close'] - current['open'])
    bullish_pinbar = lower_shadow > body * 2 and current['close'] > current['open']
    bearish_pinbar = upper_shadow > body * 2 and current['close'] < current['open']
    wick_rejection = (upper_shadow > 1.5 * body and signal_type == "high") or (lower_shadow > 1.5 * body and signal_type == "low")
    ema_trend_up = df['adaptive_ema'].iloc[i] > df['adaptive_ema'].iloc[i - 1]
    ema_trend_down = df['adaptive_ema'].iloc[i] < df['adaptive_ema'].iloc[i - 1]
    hma_trend_up = df['adaptive_hma'].iloc[i] > df['adaptive_hma'].iloc[i - 1]
    hma_trend_down = df['adaptive_hma'].iloc[i] < df['adaptive_hma'].iloc[i - 1]
    confluence_up = ema_trend_up and hma_trend_up
    confluence_down = ema_trend_down and hma_trend_down
    if signal_type == "low":
        return bullish_engulfing or bullish_pinbar or wick_rejection or confluence_up
    else:
        return bearish_engulfing or bearish_pinbar or wick_rejection or confluence_down

def get_higher_tf_data(df, interval="15min"):
    df_resampled = df.resample(interval, on='timestamp').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    df_resampled.reset_index(inplace=True)
    return df_resampled

def detect_elliott_waves(df):
    df = df.copy()
    df['adaptive_ema'] = ultra_adaptive_ema(df['close'].values, length=21, sensitivity=1.0)
    df['adaptive_hma'] = ultra_adaptive_hma(df['close'].values, length=9, sensitivity=1.0)
    df['curvature'] = compute_curvature(df['close'].values)
    df['pivot'] = realtime_peak_valley_v2(df['close'].values, df['curvature'].values, threshold=0.0025, window=2)
    df['pivot_confirmed'] = None
    for i in range(len(df)):
        if df['pivot'].iloc[i] == 'high' and advanced_price_action_validation(df, i, "high"):
            df.at[i, 'pivot_confirmed'] = 'high'
        elif df['pivot'].iloc[i] == 'low' and advanced_price_action_validation(df, i, "low"):
            df.at[i, 'pivot_confirmed'] = 'low'
    pivot_points = df[df['pivot_confirmed'].notna()].reset_index()
    pivot_prices = pivot_points.apply(
        lambda row: df['high'][row['index']] if row['pivot_confirmed'] == 'high' else df['low'][row['index']],
        axis=1).tolist()
    pivot_indexes = pivot_points['index'].tolist()
    waves = []
    for i in range(len(pivot_prices) - 6):
        p0, p1, p2, p3, p4, p5 = pivot_prices[i:i+6]
        i0, i1, i2, i3, i4, i5 = pivot_indexes[i:i+6]
        r12 = retrace_pct(p0, p1, p2)
        r34 = retrace_pct(p2, p3, p4)
        if (0.3 < r12 < 0.7 and 0.3 < r34 < 0.7 and
            is_valid_impulse_lengths(p0, p1, p2, p3, p4) and
            no_overlap(p0, p1, p3, p4)):
            wave = {
                'start_index': i0,
                'end_index': i5,
                'timestamp': df['timestamp'].iloc[i5],
                'wave': '1-5'
            }
            waves.append(wave)
            
    wave_patterns = []  
    
    double_three = detect_double_three(wave_patterns)
    triple_three = detect_triple_three(wave_patterns)

    if double_three:
        print("✅ Double Three شناسایی شد")
    if triple_three:
        print("✅ Triple Three شناسایی شد")
        
    return pd.DataFrame(waves)

def map_htf_wave_to_ltf(df, htf_waves):
    htf_wave_at_time = []
    for ts in df['timestamp']:
        valid_waves = htf_waves[htf_waves['timestamp'] <= ts]
        if not valid_waves.empty:
            last_wave = valid_waves.iloc[-1]
            htf_wave_at_time.append(last_wave['wave'])
        else:
            htf_wave_at_time.append(None)
    df['htf_wave'] = htf_wave_at_time
    return df

def is_wave_confirmed(df, wave_column='wave', htf_column='htf_wave'):
    confirmed = []
    for i in range(len(df)):
        local_wave = df[wave_column].iloc[i] if wave_column in df.columns else None
        higher_wave = df[htf_column].iloc[i]
        if higher_wave and local_wave and local_wave == higher_wave:
            confirmed.append(True)
        else:
            confirmed.append(False)
    df['confirmed_wave'] = confirmed
    return df

def validate_nested_structure(df, wave_num):
    if wave_num == 3:
        subwaves = detect_elliott_waves(df)
        return len(subwaves) >= 5
    elif wave_num in (2, 4):
        subwaves = detect_elliott_waves(df)
        return len(subwaves) >= 3
    return False

class Wave:
    def __init__(self, label, start_index, end_index, wave_type, fib_ratio=None, time_ratio=None, subwaves=None):
        self.label = label
        self.start_index = start_index
        self.end_index = end_index
        self.wave_type = wave_type
        self.fib_ratio = fib_ratio
        self.time_ratio = time_ratio
        self.subwaves = subwaves if subwaves else []

def detect_wxy_wxyz(price_data, start, end):
    segment = price_data[start:end+1]
    if len(segment) < 9:
        return None
    pivots = detect_real_time_pivot(segment, sensitivity=np.std(segment)*1.5)
    piv_locs = np.where(pivots != 0)[0]
    if len(piv_locs) < 6:
        return None
    w = piv_locs[0]
    x = piv_locs[1]
    y = piv_locs[2]
    if detect_correction_type(segment[w:x+1]) == "Zigzag" and \
       detect_correction_type(segment[x:y+1]) in ["Flat", "Zigzag"]:
        return Wave(
            label="WXY",
            start_index=start + w,
            end_index=start + y,
            wave_type="WXY",
            subwaves=[]
        )
    if len(piv_locs) >= 6:
        w, x, y, x2, z = piv_locs[:5]
        if detect_correction_type(segment[w:x+1]) == "Zigzag" and \
           detect_correction_type(segment[x:y+1]) == "Flat" and \
           detect_correction_type(segment[x2:z+1]) in ["Triangle", "Zigzag"]:
            return Wave(
                label="WXYXZ",
                start_index=start + w,
                end_index=start + z,
                wave_type="WXYXZ",
                subwaves=[]
            )
    return None

def detect_nested_elliott_waves(price_data, start, end, level=0):
    waves = []
    if end - start < 10:
        return waves
    for i in range(start, end - 4):
        for j in range(i + 2, end - 2):
            for k in range(j + 1, end):
                combo_wave = detect_wxy_wxyz(price_data, i, k)
                if combo_wave:
                    waves.append(combo_wave)
                    continue
                wave1 = price_data[i]
                wave3 = price_data[j]
                wave5 = price_data[k]
                fib1 = abs(wave3 - wave1) / abs(wave5 - wave3) if abs(wave5 - wave3) != 0 else 0
                if 0.5 < fib1 < 2.0:
                    time1 = j - i
                    time2 = k - j
                    time_ratio = time2 / time1 if time1 != 0 else 0
                    wave = Wave(
                        label=f"Wave_Level{level}_{len(waves)}",
                        start_index=i,
                        end_index=k,
                        wave_type="impulse",
                        fib_ratio=fib1,
                        time_ratio=time_ratio,
                        subwaves=detect_nested_elliott_waves(price_data, i, k, level + 1)
                    )
                    waves.append(wave)
    return waves

def validate_wave_with_indicators(wave, price_data, rsi_data, adx_data):
    divergence_passed = check_rsi_divergence(wave, rsi_data)
    momentum_strong = check_adx_strength(wave, adx_data)
    return divergence_passed and momentum_strong

def wave_to_json(wave):
    return {
        "label": wave.label,
        "start": wave.start_index,
        "end": wave.end_index,
        "type": "DoubleThree" if wave.wave_type == "WXY" else "TripleThree" if wave.wave_type == "WXYXZ" else wave.wave_type,
        "fib_ratio": wave.fib_ratio,
        "time_ratio": wave.time_ratio,
        "subwaves": [wave_to_json(sw) for sw in wave.subwaves]
    }

def check_rsi_divergence(wave, rsi_data):
    start_rsi = rsi_data[wave.start_index]
    end_rsi = rsi_data[wave.end_index]
    if wave.wave_type == "impulse":
        return end_rsi < start_rsi
    else:
        return end_rsi > start_rsi

def check_adx_strength(wave, adx_data):
    adx_values = adx_data[wave.start_index:wave.end_index+1]
    avg_adx = np.mean(adx_values)
    return avg_adx > 25

def calculate_wave_complexity(wave_data):
    if len(wave_data) < 3:
        return 0
    price_range = max(wave_data) - min(wave_data)
    variation = sum(abs(wave_data[i] - wave_data[i-1]) for i in range(1, len(wave_data))
    return variation / price_range

def detect_correction_type(wave_data):
    if len(wave_data) < 5:
        return "Unknown"
    peak1 = wave_data[0]
    trough = min(wave_data[1:-1])
    peak2 = wave_data[-1]
    if abs(peak2 - peak1) < 0.15 * abs(peak1 - trough):
        return "Flat"
    elif wave_data[1] < wave_data[2] > wave_data[3] < wave_data[4]:
        return "Triangle"
    else:
        return "Zigzag"

def check_alternation(wave2_data, wave4_data):
    type2 = detect_correction_type(wave2_data)
    type4 = detect_correction_type(wave4_data)
    time2 = len(wave2_data)
    time4 = len(wave4_data)
    complexity2 = calculate_wave_complexity(wave2_data)
    complexity4 = calculate_wave_complexity(wave4_data)
    alternation_passed = (
        type2 != type4 or
        abs(time4 - time2) > 3 or
        complexity4 > complexity2 * 1.5
    )
    return {
        "type2": type2,
        "type4": type4,
        "time2": time2,
        "time4": time4,
        "complexity2": round(complexity2, 2),
        "complexity4": round(complexity4, 2),
        "alternation_passed": alternation_passed
    }

def get_slope(wave):
    return (wave['end_price'] - wave['start_price']) / (wave['end_time'] - wave['start_time'])

def is_ending_diagonal(waves):
    if len(waves) != 5:
        return False
    w1, w2, w3, w4, w5 = waves
    overlap_1_4 = w4['end_price'] > w1['start_price'] and w4['start_price'] < w1['end_price']
    length_ok = all(abs(w['end_price'] - w['start_price']) < 1.5 * abs(w1['end_price'] - w1['start_price']) for w in [w3, w5])
    slope_similarity = all(abs(get_slope(w)) - abs(get_slope(w1)) < 0.2 for w in [w3, w5])
    return overlap_1_4 and length_ok and slope_similarity

def is_leading_diagonal(waves):
    if len(waves) != 5:
        return False
    w1, w2, w3, w4, w5 = waves
    overlap_1_4 = w4['end_price'] > w1['start_price'] and w4['start_price'] < w1['end_price']
    shortness = all(abs(w['end_price'] - w['start_price']) < abs(w1['end_price'] - w1['start_price']) * 1.5 for w in [w3, w5])
    wave_3_not_longest = abs(w3['end_price'] - w3['start_price']) < abs(w1['end_price'] - w1['start_price']) * 1.618
    return overlap_1_4 and shortness and wave_3_not_longest

def is_multi_tf_confirmed(major_wave, df_1m, df_5m):
    t_start = df_5m['timestamp'].iloc[major_wave['points'][0]]
    t_end = df_5m['timestamp'].iloc[major_wave['points'][-1]]
    df_sub = df_1m[(df_1m['timestamp'] >= t_start) & (df_1m['timestamp'] <= t_end)]
    if len(df_sub) < 10:
        return False
    nested_waves = detect_nested_elliott_waves(df_sub['close'].values, 0, len(df_sub)-1)
    return len(nested_waves) > 0 and validate_nested(nested_waves)

def validate_nested(subwaves):
    return len(subwaves) > 0

def get_wave_direction(waves):
    if len(waves) < 2:
        return None
    return "up" if waves[-1]["end_price"] > waves[-2]["end_price"] else "down"

def confirm_contextual_alignment(dir_1m, dir_5m, dir_15m):
    return dir_1m == dir_5m == dir_15m

def is_impulse_wave(waves):
    if len(waves) < 5:
        return False
    return (
        waves[2]["length"] > waves[0]["length"] and
        waves[3]["end_price"] > waves[1]["end_price"] and
        waves[4]["end_price"] > waves[2]["end_price"]
    )

def detect_combined_divergence(df, indexes, wave_type='peak'):
    divergences = []
    for i in range(1, len(indexes)):
        idx1 = indexes[i-1]
        idx2 = indexes[i]
        price1 = df['close'][idx1]
        price2 = df['close'][idx2]
        rsi1 = df['rsi'][idx1]
        rsi2 = df['rsi'][idx2]
        adx1 = df['adx'][idx1]
        adx2 = df['adx'][idx2]

        if wave_type == 'peak' and price2 > price1 and rsi2 < rsi1 and adx2 < adx1:
            divergences.append(idx2)
        if wave_type == 'valley' and price2 < price1 and rsi2 > rsi1 and adx2 < adx1:
            divergences.append(idx2)
    return divergences

class WavePatternType(Enum):
    IMPULSE = "Impulse"
    ZIGZAG = "Zigzag"
    FLAT = "Flat"
    TRIANGLE = "Triangle"
    DOUBLE_THREE = "Double Three"
    TRIPLE_THREE = "Triple Three"

class CompositeCorrection:
    def __init__(self, waves, type):
        self.waves = waves
        self.type = type

def detect_double_three(waves):
    if len(waves) < 5:
        return None

    for i in range(1, len(waves) - 3):
        w = waves[i-1]
        x = waves[i]
        y = waves[i+1]

        if is_correction(w) and is_connector(x) and is_correction(y):
            return CompositeCorrection([w, x, y], WavePatternType.DOUBLE_THREE)

    return None

def detect_triple_three(waves):
    if len(waves) < 7:
        return None

    for i in range(1, len(waves) - 5):
        w = waves[i-1]
        x = waves[i]
        y = waves[i+1]
        z1 = waves[i+2]
        z2 = waves[i+3]

        if is_correction(w) and is_connector(x) and is_correction(y) and is_connector(z1) and is_correction(z2):
            return CompositeCorrection([w, x, y, z1, z2], WavePatternType.TRIPLE_THREE)

    return None

def is_correction(wave):
    return wave.pattern_type in [WavePatternType.ZIGZAG, WavePatternType.FLAT, WavePatternType.TRIANGLE]

def is_connector(wave):
    return wave.length_pct < 38.2 and wave.duration_pct < 0.382

def is_zigzag_structure(subwaves):
    return len(subwaves) == 3 and is_impulse(subwaves[0]) and is_correction(subwaves[1]) and is_impulse(subwaves[2])

def is_flat_structure(subwaves):
    return len(subwaves) == 3 and is_correction(subwaves[0]) and is_correction(subwaves[1]) and is_impulse(subwaves[2])

def is_triangle_structure(subwaves):
    return len(subwaves) == 5 and all(is_correction(w) for w in subwaves)

def is_impulse(wave):
    return wave.pattern_type == WavePatternType.IMPULSE

def detect_higher_tf_wave_structure(df_15m):
    pivot_indexes = df_15m[df_15m['pivot_confirmed'].notna()]['index'].tolist()
    pivot_prices = df_15m.apply(
        lambda row: row['high'] if row['pivot_confirmed'] == 'high' else row['low'], axis=1
    ).tolist()

    if len(pivot_indexes) < 6:
        return None

    waves = []
    for i in range(len(pivot_indexes) - 6):
        i0, i1, i2, i3, i4, i5 = pivot_indexes[i:i+6]
        p0, p1, p2, p3, p4, p5 = pivot_prices[i:i+6]
        
        r12 = retrace_pct(p0, p1, p2)
        r34 = retrace_pct(p2, p3, p4)
        if 0.3 < r12 < 0.7 and 0.3 < r34 < 0.7 and no_overlap(p0, p1, p3, p4):
            wave_structure = {
                'wave': '1-5',
                'indexes': [i0, i1, i2, i3, i4, i5],
                'prices': [p0, p1, p2, p3, p4, p5],
                'direction': 'up' if p5 > p0 else 'down'
            }
            waves.append(wave_structure)
    return waves[-1] if waves else None

def infer_context_from_higher_tf(higher_tf_wave, current_index):
    i0, i1, i2, i3, i4, i5 = higher_tf_wave['indexes']
    
    if i2 <= current_index <= i3:
        return 'wave_3_high_tf'
    elif i4 <= current_index <= i5:
        return 'wave_5_high_tf'
    elif i1 <= current_index <= i2:
        return 'wave_2_correction'
    elif i3 <= current_index <= i4:
        return 'wave_4_correction'
    else:
        return 'unknown'

def realtime_wave_detection(symbol='BTC/USDT'):
    
    def compute_fractal_energy(price_series, window=14):
        high = pd.Series(price_series).rolling(window=window).max()
        low = pd.Series(price_series).rolling(window=window).min()
        range_ = high - low
        energy = range_ / (pd.Series(price_series).rolling(window=window).std() + 1e-8)
        return energy.fillna(0).values

    while True:
        df_1m = fetch_ohlcv(symbol, '1m', limit=1000)
        df_5m = fetch_ohlcv(symbol, '5m', limit=1000)
        df_15m = fetch_ohlcv(symbol, '15m', limit=1000)

        if df_1m.empty or df_5m.empty or df_15m.empty:
            print("Error fetching data, retrying...")
            time.sleep(15)
            continue

        waves_1m = detect_elliott_waves(df_1m)
        waves_5m = detect_elliott_waves(df_5m)
        waves_15m = detect_elliott_waves(df_15m)
        dir_1m = get_wave_direction(waves_1m)
        dir_5m = get_wave_direction(waves_5m)
        dir_15m = get_wave_direction(waves_15m)
        
        df_15m_from_5m = get_higher_tf_data(df_5m, interval="15min")
        htf_waves = detect_elliott_waves(df_15m_from_5m)
        df_5m = map_htf_wave_to_ltf(df_5m, htf_waves)
        df_5m = is_wave_confirmed(df_5m)

        df_5m['adaptive_ema'] = ultra_adaptive_ema(df_5m['close'].values, length=21, sensitivity=1.0)
        df_5m['adaptive_hma'] = ultra_adaptive_hma(df_5m['close'].values, length=9, sensitivity=1.0)

        close_prices = df_5m['close'].values
        
        def compute_advanced_curvature(price_series):
            hma = ultra_adaptive_hma(price_series, length=9, sensitivity=1.0)
            curvature = np.gradient(np.gradient(hma))  # مشتق دوم
            return curvature

        curvature = compute_advanced_curvature(df_5m['close'].values)
        
        abs_curv = np.abs(curvature)
        if len(close_prices) > 4:
            valid_abs_curv = abs_curv[2:-2]
            if len(valid_abs_curv) > 0:
                threshold_value = np.percentile(valid_abs_curv, 75)
            else:
                threshold_value = 0.001
        else:
            threshold_value = 0.001

        df_5m['pivot_confirmed'] = None
        
        fractal_energy = compute_fractal_energy(df_5m['close'].values)

        adaptive_threshold = np.percentile(np.abs(curvature[2:-2]), 75)
        for i in range(2, len(df_5m)-2):
            c = curvature[i]
            f_energy = fractal_energy[i]
            
            is_peak = c < -adaptive_threshold and f_energy > 1.0
            is_valley = c > adaptive_threshold and f_energy > 1.0
            
            if is_peak:
                df_5m.at[i, 'pivot_confirmed'] = 'high'
            elif is_valley:
                df_5m.at[i, 'pivot_confirmed'] = 'low'

        df_5m['rsi'] = RSIIndicator(close=df_5m['close'], window=14).rsi()
        adx_i = ADXIndicator(df_5m['high'], df_5m['low'], df_5m['close'], window=14)
        df_5m['adx'] = adx_i.adx()
        df_5m['plus_di'] = adx_i.plus_di()
        df_5m['minus_di'] = adx_i.minus_di()

        pivot_points_5m = df_5m[df_5m['pivot_confirmed'].notna()].reset_index()
        valley_indexes = [row['index'] for _, row in pivot_points_5m.iterrows() if row['pivot_confirmed'] == 'low']
        peak_indexes = [row['index'] for _, row in pivot_points_5m.iterrows() if row['pivot_confirmed'] == 'high']
        wave3_divs = detect_combined_divergence(df_5m, valley_indexes, wave_type='valley')
        wave5_divs = detect_combined_divergence(df_5m, peak_indexes, wave_type='peak')
        confirmed_wave3 = [idx for idx in valley_indexes if idx in wave3_divs]
        confirmed_wave5 = [idx for idx in peak_indexes if idx in wave5_divs]

        waves = []
        confirmed_trades = []
        pivot_prices_5m = pivot_points_5m.apply(
            lambda row: df_5m['high'][row['index']] if row['pivot_confirmed'] == 'high' else df_5m['low'][row['index']],
            axis=1).tolist()
        pivot_indexes_5m = pivot_points_5m['index'].tolist()

        for i in range(len(pivot_prices_5m) - 6):
            p0, p1, p2, p3, p4, p5 = pivot_prices_5m[i:i + 6]
            i0, i1, i2, i3, i4, i5 = pivot_indexes_5m[i:i + 6]
            r12 = retrace_pct(p0, p1, p2)
            r34 = retrace_pct(p2, p3, p4)
            wave2_data = df_5m['close'].iloc[i1:i2+1].values
            wave4_data = df_5m['close'].iloc[i3:i4+1].values
            alternation_result = check_alternation(wave2_data, wave4_data)
            alternation_passed = alternation_result["alternation_passed"]
            is_impulse_length_valid = is_valid_impulse_lengths(p0, p1, p2, p3, p4)
            is_no_overlap_valid = no_overlap(p0, p1, p3, p4)
            is_strong_wave3_valid = is_strong_wave3(df_5m, i2, i3)
            w1_dict = {'length': abs(p1 - p0), 'high': p1, 'subwaves': 5}
            w2_dict = {'subwaves': 3}
            w3_dict = {'length': abs(p3 - p2), 'subwaves': 5, 'end': i3}
            w4_dict = {'low': p4, 'subwaves': 3}
            w5_dict = {'length': abs(p5 - p4), 'subwaves': 5, 'end': i5}
            rsi_list = df_5m['rsi'].tolist()
            adx_list = df_5m['adx'].tolist()
            is_entire_impulse_not_valid = is_not_impulsive_advanced(
                w1_dict, w2_dict, w3_dict, w4_dict, w5_dict, rsi_list, adx_list)
            wave1 = {"start_index": i0, "end_index": i1}
            wave3 = {"start_index": i2, "end_index": i3}
            wave5 = {"start_index": i4, "end_index": i5}
            time_wave1 = calculate_wave_duration(wave1)
            time_wave3 = calculate_wave_duration(wave3)
            time_wave5 = calculate_wave_duration(wave5)
            total_impulse_time = i5 - i0
            time_ratios = {}
            if time_wave1 and time_wave3:
                time_ratios["wave3_vs_wave1"] = time_wave3 / time_wave1
            if time_wave1 and time_wave5:
                time_ratios["wave5_vs_total"] = time_wave5 / total_impulse_time if total_impulse_time else None
            valid_time = True
            if time_ratios.get("wave3_vs_wave1") is not None:
                ratio3 = time_ratios["wave3_vs_wave1"]
                if not (0.618 <= ratio3 <= 2.0):
                    valid_time = False
            if time_ratios.get("wave5_vs_total") is not None:
                ratio5 = time_ratios["wave5_vs_total"]
                if not (0.3 <= ratio5 <= 0.9):
                    valid_time = False
            wave1_diag = {'start_price': p0, 'end_price': p1, 'start_time': i0, 'end_time': i1}
            wave2_diag = {'start_price': p1, 'end_price': p2, 'start_time': i1, 'end_time': i2}
            wave3_diag = {'start_price': p2, 'end_price': p3, 'start_time': i2, 'end_time': i3}
            wave4_diag = {'start_price': p3, 'end_price': p4, 'start_time': i3, 'end_time': i4}
            wave5_diag = {'start_price': p4, 'end_price': p5, 'start_time': i4, 'end_time': i5}
            waves_for_diagonal = [wave1_diag, wave2_diag, wave3_diag, wave4_diag, wave5_diag]
            diagonal_type = None
            if is_ending_diagonal(waves_for_diagonal):
                diagonal_type = "Ending Diagonal"
            elif is_leading_diagonal(waves_for_diagonal):
                diagonal_type = "Leading Diagonal"
            major_wave = {
                'wave': '1-5',
                'points': [i0, i1, i2, i3, i4, i5],
                'prices': [p0, p1, p2, p3, p4, p5],
                'retraces': {'1-2': round(r12, 2), '3-4': round(r34, 2)},
                'alternation': alternation_passed,
                'alternation_details': alternation_result,
                'time_ratios': time_ratios,
                'durations': {
                    'wave1': time_wave1,
                    'wave3': time_wave3,
                    'wave5': time_wave5,
                    'total_impulse': total_impulse_time
                },
                'valid_time_structure': valid_time,
                'diagonal': diagonal_type
            }
            wave1_dict = {'start': i0, 'end': i1}
            wave3_dict = {'start': i2, 'end': i3}
            wave3_strength_valid = validate_wave3_strength(df_5m, wave1_dict, wave3_dict)
            contextual_alignment = confirm_contextual_alignment(dir_1m, dir_5m, dir_15m)
            impulse_valid = is_impulse_wave(waves_15m)
            
            higher_tf_wave = detect_higher_tf_wave_structure(df_15m)
            if higher_tf_wave:
                context = infer_context_from_higher_tf(higher_tf_wave, i5)
                major_wave['context_15m'] = context
                if context == 'wave_3_high_tf' or context == 'wave_5_high_tf':
                    contextually_valid = True
                else:
                    contextually_valid = False
            else:
                contextually_valid = False
                major_wave['context_15m'] = None
            
            major_wave['contextual_validation'] = contextually_valid
            major_wave['higher_tf_context'] = context if higher_tf_wave else None
            
            if (0.3 < r12 < 0.7 and
                0.3 < r34 < 0.7 and
                alternation_passed and
                is_impulse_length_valid and
                is_no_overlap_valid and
                is_strong_wave3_valid and
                (not is_entire_impulse_not_valid) and
                valid_time and
                wave3_strength_valid and
                i2 in confirmed_wave3 and
                i5 in confirmed_wave5 and
                contextual_alignment and
                impulse_valid and
                contextually_valid):
                multi_tf_confirmed = is_multi_tf_confirmed(major_wave, df_1m, df_5m)
                major_wave['multi_tf_confirmed'] = multi_tf_confirmed
                mtf_confirmed = df_5m['confirmed_wave'].iloc[i5]
                nested_valid = validate_nested_structure(df_5m.iloc[i0:i5], 3)
                if multi_tf_confirmed and mtf_confirmed and nested_valid:
                    fib_projections = project_fibonacci_zone_advanced(p0, p1, p3)
                    all_fib_levels = fib_projections['wave1_proj'] + fib_projections['wave3_ext']
                    confirmed_levels = check_fib_reaction(df_5m, all_fib_levels, i5)
                    div_wave3 = i2 in wave3_divs
                    div_wave5 = i5 in wave5_divs
                    if confirmed_levels:
                        trade_zone = define_trade_zone(p4, confirmed_levels[0])
                    else:
                        trade_zone = None
                    major_wave['fib_levels'] = {
                        'projected': [round(lvl, 2) for lvl in fib_projections['wave1_proj']],
                        'extended': [round(lvl, 2) for lvl in fib_projections['wave3_ext']],
                        'confirmed': confirmed_levels
                    }
                    major_wave['divergence'] = {'wave3': div_wave3, 'wave5': div_wave5}
                    major_wave['trade_zone'] = trade_zone
                    close_prices = df_5m['close'].values
                    nested_waves = detect_nested_elliott_waves(close_prices, i0, i5)
                    validated_waves = []
                    for wave in nested_waves:
                        if validate_wave_with_indicators(wave, df_5m['close'], df_5m['rsi'], df_5m['adx']):
                            validated_waves.append(wave)
                    for wave in validated_waves:
                        entry = df_5m['close'].iloc[wave.end_index]
                        stop = df_5m['close'].iloc[wave.start_index]
                        target = entry + (entry - stop) * 1.618 if wave.direction == 'up' else entry - (stop - entry) * 1.618
                        wave.rr = generate_dynamic_rr(entry, stop, target)
                    major_wave['nested_waves'] = [wave_to_json(w, include_rr=True) for w in validated_waves]
                    if i5 == len(df_5m) - 1:
                        for wave in validated_waves:
                            if wave.end_index == len(df_5m) - 1:
                                signal = {
                                    "entry": df_5m['close'].iloc[wave.end_index],
                                    "stop": df_5m['close'].iloc[wave.start_index],
                                    "type": wave.wave_type,
                                    "r:r": 1.618
                                }
                                print(f"🚀 Real-Time Signal: {signal}")
                    wave1_len = abs(p1 - p0)
                    wave3_len = abs(p3 - p2)
                    wave4_end_price = p4
                    wave5_target_0618 = p3 + wave1_len * 0.618
                    wave5_target_1 = p3 + wave1_len * 1.0
                    wave5_target_1618 = p3 + wave1_len * 1.618
                    entry_zone_bottom = wave4_end_price
                    entry_zone_top = wave5_target_0618
                    expected_reversal_price = fibonacci_extension(
                        low_point=p0, 
                        high_point=p3, 
                        retrace_point=p4, 
                        ratio=1.618
                    )
                    ema_val = df_5m['adaptive_ema'].values
                    hma_val = df_5m['adaptive_hma'].values
                    hma_cross = (hma_val[-2] < ema_val[-2] and hma_val[-1] > ema_val[-1]) or \
                                (hma_val[-2] > ema_val[-2] and hma_val[-1] < ema_val[-1])
                    current_candle = df_5m.iloc[-1]
                    prev_candle = df_5m.iloc[-2]
                    bullish_pattern = is_bullish_engulfing(current_candle, prev_candle) or is_pin_bar(current_candle) or is_doji(current_candle)
                    bearish_pattern = is_bearish_engulfing(current_candle, prev_candle) or is_pin_bar(current_candle) or is_doji(current_candle)
                    tolerance = expected_reversal_price * 0.005
                    current_price = df_5m['close'].iloc[-1]
                    if (current_price >= expected_reversal_price - tolerance and
                        current_price <= expected_reversal_price + tolerance and
                        hma_cross and 
                        bullish_pattern):
                        entry_price = current_price
                        stop_loss = wave4_end_price - (wave1_len * 0.382)
                        take_profit = entry_price + (entry_price - stop_loss) * 1.618
                        buy_signal = {
                            "entry": entry_price,
                            "sl": stop_loss,
                            "tp": take_profit,
                            "type": "BUY",
                            "wave": "5_end",
                            "confirmed_by": ["PRZ", "EMA/HMA", "Candlestick"]
                        }
                        print(f"✅ سیگنال خرید در پایان موج ۵ صادر شد! Entry: {entry_price:.2f}, TP: {take_profit:.2f}, SL: {stop_loss:.2f}, R:R = 1.618")
                        major_wave['wave5_entry'] = buy_signal
                    elif (current_price >= expected_reversal_price - tolerance and
                          current_price <= expected_reversal_price + tolerance and
                          hma_cross and 
                          bearish_pattern):
                        entry_price = current_price
                        stop_loss = wave4_end_price + (wave1_len * 0.382)
                        take_profit = entry_price - (stop_loss - entry_price) * 1.618
                        sell_signal = {
                            "entry": entry_price,
                            "sl": stop_loss,
                            "tp": take_profit,
                            "type": "SELL",
                            "wave": "5_end",
                            "confirmed_by": ["PRZ", "EMA/HMA", "Candlestick"]
                        }
                        print(f"✅ سیگنال فروش در پایان موج ۵ صادر شد! Entry: {entry_price:.2f}, TP: {take_profit:.2f}, SL: {stop_loss:.2f}, R:R = 1.618")
                        major_wave['wave5_entry'] = sell_signal
                    else:
                        major_wave['wave5_entry'] = None
                    major_wave['wave5_forecast'] = {
                        'target_0618': wave5_target_0618,
                        'target_1': wave5_target_1,
                        'target_1618': wave5_target_1618,
                        'entry_zone': (entry_zone_bottom, entry_zone_top),
                        'expected_reversal_price': expected_reversal_price,
                        'tolerance': tolerance,
                        'hma_cross': hma_cross,
                        'bullish_pattern': bullish_pattern,
                        'bearish_pattern': bearish_pattern
                    }
                    wave4_end_index = i4
                    wave4_end_price = p4
                    early_entry_ready = True
                    wave1_len = abs(p1 - p0)
                    is_uptrend = p1 > p0
                    if is_uptrend:
                        wave5_proj_price = wave4_end_price + wave1_len * 0.618
                        stop_loss = min(p2, p4)
                    else:
                        wave5_proj_price = wave4_end_price - wave1_len * 0.618
                        stop_loss = max(p2, p4)
                    entry_signal = {
                        'entry_price': wave4_end_price,
                        'tp': wave5_proj_price,
                        'sl': stop_loss,
                        'rr_ratio': abs(wave5_proj_price - wave4_end_price) / abs(wave4_end_price - stop_loss),
                        'wave': 'early_wave5'
                    }
                    major_wave['early_entry'] = entry_signal
                    
                    idx = i4
                    is_wave_4_end = wave_type == 'impulse' and subwave == 4
                    
                    is_engulfing_val = is_bullish_engulfing(df_5m.iloc[idx], df_5m.iloc[idx-1]) if idx >= 1 else False
                    is_pinbar_val = is_pin_bar(df_5m.iloc[idx])
                    is_price_action = is_engulfing_val or is_pinbar_val
                    
                    hma_val = df_5m['adaptive_hma'].iloc[idx]
                    ema_val = df_5m['adaptive_ema'].iloc[idx]
                    if idx >= 1:
                        hma_cross_up = (hma_val > ema_val) and (df_5m['adaptive_hma'].iloc[idx-1] <= df_5m['adaptive_ema'].iloc[idx-1])
                    else:
                        hma_cross_up = False
                    
                    wave3_len = abs(p3 - p2)
                    wave1_len = abs(p1 - p0)
                    wave3_momentum = df_5m['adx'].iloc[i3]
                    wave1_momentum = df_5m['adx'].iloc[i1]
                    strong_wave3 = abs(wave3_len) > abs(wave1_len) and wave3_momentum > wave1_momentum
                    
                    entry_signal = is_wave_4_end and is_price_action and hma_cross_up and strong_wave3
                    
                    if entry_signal:
                        entry_price = df_5m['close'].iloc[idx]
                        stop_loss = p4
                        take_profit = entry_price + 1.618 * (p3 - p0)
                        
                        trade_signal = {
                            "entry_idx": int(idx),
                            "entry_price": float(entry_price),
                            "stop_loss": float(stop_loss),
                            "take_profit": float(take_profit),
                            "type": "buy"
                        }
                        confirmed_trades.append(trade_signal)
                    
                    if wave3_strength_valid:
                        if direction == 'up':
                            wave3_entry_price = p2
                            wave3_sl = min(p1, p2) - abs(p3 - p2) * 0.382
                            wave3_tp = p2 + abs(p3 - p2) * 1.618
                        else:
                            wave3_entry_price = p2
                            wave3_sl = max(p1, p2) + abs(p3 - p2) * 0.382
                            wave3_tp = p2 - abs(p3 - p2) * 1.618
                        major_wave['wave3_rr'] = generate_dynamic_rr(wave3_entry_price, wave3_sl, wave3_tp)
                    
                    if contextually_valid and confirmed_levels:
                        if direction == 'up':
                            wave5_entry_price = current_price
                            wave5_sl = wave4_end_price - (wave1_len * 0.382)
                            wave5_tp = wave5_entry_price + (wave5_entry_price - wave5_sl) * 1.618
                        else:
                            wave5_entry_price = current_price
                            wave5_sl = wave4_end_price + (wave1_len * 0.382)
                            wave5_tp = wave5_entry_price - (wave5_sl - wave5_entry_price) * 1.618
                        major_wave['wave5_rr'] = generate_dynamic_rr(wave5_entry_price, wave5_sl, wave5_tp)
                    
                    major_wave['rr_structures'] = {
                        'wave1': major_wave.get('wave1_rr', None),
                        'wave3': major_wave.get('wave3_rr', None),
                        'wave5': major_wave.get('wave5_rr', None)
                    }
                    
                    waves.append(major_wave)
        
        output = {
            'waves': [
                {
                    **wave,
                    'nested_structure': wave.get('nested_waves', []),
                    'wave5_forecast': wave.get('wave5_forecast', None),
                    'wave5_entry': wave.get('wave5_entry', None),
                    'early_entry': wave.get('early_entry', None),
                    'contextual_validation': wave.get('contextual_validation', None),
                    'higher_tf_context': wave.get('higher_tf_context', None),
                    'rr_structures': wave.get('rr_structures', None)
                } 
                for wave in waves
            ],
            'confirmed_trades': confirmed_trades
        }
        with open('wave_professional_price_action_multitimeframe.json', 'w') as f:
            f.write(json.dumps(output, indent=2))
        print(json.dumps(output, indent=2))
        time.sleep(15)

if __name__ == "__main__":
    realtime_wave_detection()
