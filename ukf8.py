!pip install numpy pandas matplotlib scipy filterpy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints

# =====================================================================
# 0. 基本設定 (Configuration)
# =====================================================================
CONFIG = {
    'C': 2,                 # 分析するコーホート（世代）の数
    'fixed_mu': 2.0,        # インフレの長期的な目標値（アンカー）
    
    # 年齢を「年（歳）」で直接指定
    'ages': [40, 60],       # 各コーホートの年齢 (例: 40歳, 60歳)
    
    # --- gamma (過去のインフレ経験への依存度) の異質性スイッチ ---
    # False: 全世代で同じ gamma_t を推計する (シンプルなモデル)
    # True : 世代ごとに異なる gamma_{c,t} を推計する (複雑だが現実に近いモデル)
    'heterogeneous_gamma': True, 
    
    # --- データ期間の設定 (実データの日付に合わせて変更してください) ---
    'start_date_long': '1954-04-01',  # 高齢層の過去の記憶を計算するため、十分に古い年月を指定
    'start_date_short': '2004-04-01', # アンケート（サーベイ）データが実際に存在する最初の月
    'end_date': '2024-03-01',         # データの終わりの月
    
    'calibrated_theta': 1.5,          # 記憶の減衰パラメータ (過去の経験をどれくらい早く忘れるか)
    
    # --- ファイル出力設定 ---
    'mock_csv': 'mock_data_final.csv',              # 生成した擬似データ
    'output_csv': 'endogenous_variables_final.csv', # 全・内生変数（1年先・5年先予想含む）
    'params_csv': 'estimated_parameters.csv',       # 全推計パラメータ
    
    # --- [Step 1] マクロ推論の推計対象 ---
    # パラメータ配列: [phi(AR1係数), sig_pi(インフレショック), sig_S(観測ノイズ)]
    'init_step1': [0.5, 0.1, 0.2],               
    'bounds_step1': ((0.01, 0.99), (1e-4, 5.0), (1e-4, 5.0)),
}

# --- [Step 2] ミクロ期待形成の推計対象 (コーホート数 C に合わせて動的に生成) ---
# 推計対象パラメータ: [sig_gamma, sig_lambda, sig_Z, sig_lambda_obs, alpha] + [b_1, b_2, ...]
#   alpha: ロジスティック関数のインフレ感応度 (インフレ変動に対する反応の敏感さ)
#   b_c  : 各コーホートの「偶然同じ選択肢に留まる」ベース確率
# ★ 識別トラップ回避: alphaの初期値を5.0と大きめに設定し、b_c の下限を0.05に引き上げ
CONFIG['init_step2'] = [0.05, 0.05, 0.1, 0.05, 5.0] + [0.3] * CONFIG['C']
CONFIG['bounds_step2'] = tuple(
    # sig_lambdaの上限を0.2に厳しく制限し、潜在変数の乱高下（過剰適合）を防ぐ
    [(1e-4, 1.0), (1e-4, 0.2), (1e-4, 5.0), (1e-4, 1.0), (0.0, 50.0)] + 
    [(0.05, 0.95)] * CONFIG['C']
)

# --- 内部計算用の次元設定 ---
# 状態ベクトル X (見えない状態): [pi_t, gamma_1, gamma_2, lambda_1, lambda_2] (C=2なら計5次元)
num_gamma_states = CONFIG['C'] if CONFIG['heterogeneous_gamma'] else 1
DIM_X = 1 + num_gamma_states + CONFIG['C']

# 観測ベクトル Z (手元のデータ): [S_t, Z_1, Z_2, lambda_obs_1, lambda_obs_2] (C=2なら計5次元)
DIM_Z = 1 + CONFIG['C'] + CONFIG['C']

# --- 日付インデックスの作成 ---
global_dates = pd.date_range(start=CONFIG['start_date_long'], end=CONFIG['end_date'], freq='MS')
CONFIG['T_long'] = len(global_dates)
CONFIG['start_idx_short'] = global_dates.get_loc(CONFIG['start_date_short'])
CONFIG['T_short'] = CONFIG['T_long'] - CONFIG['start_idx_short']
dates_short = global_dates[CONFIG['start_idx_short']:]

# =====================================================================
# パラメータ保存ルーチン (論文のTable出力用)
# =====================================================================
def save_all_parameters(config, res1_x, res2_x, output_filename):
    """論文のTable用に、すべての推計パラメータを整理してCSVに出力します"""
    params_dict = {'Category': [], 'Parameter': [], 'Value': [], 'Description': []}
    def add_param(cat, name, val, desc):
        params_dict['Category'].append(cat)
        params_dict['Parameter'].append(name)
        params_dict['Value'].append(round(val, 6))
        params_dict['Description'].append(desc)

    # 1. カリブレート変数
    add_param('Calibrated', 'mu', config['fixed_mu'], '長期インフレ目標')
    add_param('Calibrated', 'theta', config['calibrated_theta'], 'インフレ経験の減衰パラメータ')
    
    # 2. Step 1 推計結果
    opt_phi, opt_sig_pi, opt_sig_S = res1_x
    add_param('Step 1 (Macro)', 'phi', opt_phi, 'インフレの持続性 (AR1)')
    add_param('Step 1 (Macro)', 'sig_pi', opt_sig_pi, 'インフレショック標準偏差')
    add_param('Step 1 (Macro)', 'sig_S', opt_sig_S, 'マクロシグナル観測ノイズ')

    # 3. Step 2 推計結果
    opt_sig_gamma, opt_sig_lambda, opt_sig_Z, opt_sig_lambda_obs, opt_alpha = res2_x[:5]
    opt_b = res2_x[5:5+config['C']]
    
    add_param('Step 2 (Micro)', 'sig_gamma', opt_sig_gamma, '経験ウエイト(gamma)の変動ショック')
    add_param('Step 2 (Micro)', 'sig_lambda', opt_sig_lambda, '粘着情報(lambda)の変動ショック')
    add_param('Step 2 (Micro)', 'sig_Z', opt_sig_Z, 'サーベイ観測ノイズ')
    add_param('Step 2 (Micro)', 'sig_lambda_obs', opt_sig_lambda_obs, 'プロキシデータ観測ノイズ')
    add_param('Step 2 (Micro)', 'alpha', opt_alpha, 'ロジスティック関数のインフレ感応度')
    for i, b_val in enumerate(opt_b):
        add_param('Step 2 (Micro)', f'b_{i+1}', b_val, f'Cohort {i+1} の偶然留まるベース確率')

    df_params = pd.DataFrame(params_dict)
    df_params.to_csv(output_filename, index=False, encoding='utf-8-sig')

# =====================================================================
# 1. モックデータ (シミュレーション用の擬似データ) の生成
# =====================================================================
def generate_mock_data():
    """歴史的ショック、時変の粘着情報、およびロジスティックバイアスを含むプロキシデータを生成します"""
    np.random.seed(42)
    T_long = CONFIG['T_long']
    start_idx = CONFIG['start_idx_short']
    
    true_phi, true_sig_pi, true_sig_S = 0.8, 0.1, 0.2
    true_pi = np.zeros(T_long)
    true_pi[0] = CONFIG['fixed_mu']
    
    # 歴史的なインフレ・デフレショックの期間設定 (識別のため極めて重要)
    idx_shock1_start = global_dates.get_loc('1973-10-01')
    idx_shock1_end   = global_dates.get_loc('1980-12-01')
    idx_shock2_start = global_dates.get_loc('1998-01-01')
    idx_shock2_end   = global_dates.get_loc('2012-12-01')
    
    for t in range(1, T_long):
        # 基本的なAR(1)プロセス
        true_pi[t] = (1 - true_phi) * CONFIG['fixed_mu'] + true_phi * true_pi[t-1] + np.random.normal(0, true_sig_pi)
        # 構造的なショックの付加
        if idx_shock1_start <= t <= idx_shock1_end: true_pi[t] += 0.4
        elif idx_shock2_start <= t <= idx_shock2_end: true_pi[t] -= 0.4
            
    # ★ 識別トラップ回避: マクロのボラティリティは、観測されるシグナル(S)の差分で計算する
    S_data = true_pi + np.random.normal(0, true_sig_S, T_long)
    abs_delta_S = np.abs(np.diff(S_data, prepend=S_data[0]))
    
    # 真の時変 gamma_{c,t} の生成
    true_gamma = np.zeros((T_long, CONFIG['C']))
    true_gamma[0, :] = 0.5
    for t in range(1, T_long):
        for c in range(CONFIG['C']):
            true_gamma[t, c] = np.clip(true_gamma[t-1, c] + np.random.normal(0, 0.03), 0.1, 0.9)
            
    # 真の時変 lambda_{c,t} と、ロジスティックバイアスが乗ったプロキシデータの生成
    true_lambda = np.zeros((T_long, CONFIG['C']))
    true_lambda[0, :] = [0.3, 0.7] # コーホートごとの初期値
    
    # モックの真値設定 (UKFがこれをリカバリーできるかテストする)
    true_alpha = 5.0
    true_b = [0.4, 0.5]
    lambda_obs_data = np.zeros((T_long, CONFIG['C']))
    
    for t in range(1, T_long):
        for c in range(CONFIG['C']):
            # lambda_{c,t} のランダムウォーク推移 (滑らかに動くようにノイズを小さめ0.01にする)
            true_lambda[t, c] = np.clip(true_lambda[t-1, c] + np.random.normal(0, 0.01), 0.1, 0.9)
            
            # ロジスティック関数: インフレ変動が大きいほど、偶然同じ選択肢に留まる確率が減る
            stay_prob = true_b[c] * (2.0 / (1.0 + np.exp(true_alpha * abs_delta_S[t])))
            
            # 外部から観測されるプロキシデータ = 真の粘着性 + (情報を更新したのに偶然留まった割合) + 観測ノイズ
            expected_obs = true_lambda[t, c] + (1 - true_lambda[t, c]) * stay_prob
            lambda_obs_data[t, c] = expected_obs + np.random.normal(0, 0.03)
            
    # モック用の1年先サーベイ(Z)と5年先サーベイ(Z5)を生成
    Z_data = np.full((T_long, CONFIG['C']), np.nan)
    Z5_data = np.full((T_long, CONFIG['C']), np.nan)
    
    for c, age_years in enumerate(CONFIG['ages']):
        L_c = age_years * 12 - 240 
        Z_data[start_idx-1, c] = 2.0 
        Z5_data[start_idx-1, c] = 2.0
        
        for t in range(start_idx, T_long):
            available_L = min(L_c, t)
            if available_L <= 0: continue
            
            # インフレ経験 (Lifetime Experience) の計算
            weights = (age_years * 12 - np.arange(1, available_L + 1)) ** CONFIG['calibrated_theta']
            weights = weights / np.sum(weights)
            true_E_ct = np.sum(weights * true_pi[t - available_L : t][::-1])
            
            # 1年先・5年先の合理的インフレ期待
            true_rational_exp_1y = (1 - true_phi**12) * CONFIG['fixed_mu'] + (true_phi**12) * true_pi[t]
            true_rational_exp_5y = (1 - true_phi**60) * CONFIG['fixed_mu'] + (true_phi**60) * true_pi[t]
            
            # ファンダメンタル予想 (Z*) と 粘着情報のブレンド [1年先]
            Z_star_1y = true_gamma[t, c] * true_E_ct + (1 - true_gamma[t, c]) * true_rational_exp_1y
            Z_data[t, c] = (1 - true_lambda[t, c]) * Z_star_1y + true_lambda[t, c] * Z_data[t-1, c] + np.random.normal(0, 0.05)
            
            # ファンダメンタル予想 (Z*) と 粘着情報のブレンド [5年先]
            Z_star_5y = true_gamma[t, c] * true_E_ct + (1 - true_gamma[t, c]) * true_rational_exp_5y
            Z5_data[t, c] = (1 - true_lambda[t, c]) * Z_star_5y + true_lambda[t, c] * Z5_data[t-1, c] + np.random.normal(0, 0.05)
            
    df_mock = pd.DataFrame({'S_t': S_data}, index=global_dates)
    for c in range(CONFIG['C']):
        df_mock[f'Z_{c+1}'] = Z_data[:, c]
        df_mock[f'Z5_{c+1}'] = Z5_data[:, c]
        df_mock[f'Lambda_obs_{c+1}'] = lambda_obs_data[:, c]
        
    df_short = df_mock.iloc[CONFIG['start_idx_short']:].copy()
    Z_lag_short = np.zeros((CONFIG['T_short'], CONFIG['C']))
    Z_lag_short[0, :] = df_mock[[f'Z_{c+1}' for c in range(CONFIG['C'])]].iloc[CONFIG['start_idx_short']-1].values
    if CONFIG['T_short'] > 1:
        Z_lag_short[1:, :] = df_short[[f'Z_{c+1}' for c in range(CONFIG['C'])]].values[:-1, :]
        
    abs_delta_S_short = abs_delta_S[CONFIG['start_idx_short']:]
        
    return S_data, df_short['S_t'].values, df_short[[f'Z_{c+1}' for c in range(CONFIG['C'])]].values, \
           df_short[[f'Lambda_obs_{c+1}' for c in range(CONFIG['C'])]].values, Z_lag_short, abs_delta_S_short

# =====================================================================
# 2. Step 1: 線形カルマンフィルター (マクロインフレ動学の推論)
# =====================================================================
def setup_linear_kf(phi, sig_pi, sig_S):
    """Step 1 の線形カルマンフィルターを構築します"""
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x, kf.P, kf.F, kf.B, kf.H = np.array([[CONFIG['fixed_mu']]]), np.array([[1.0]]), np.array([[phi]]), np.array([[1.0]]), np.array([[1.0]])
    kf.Q, kf.R = np.array([[sig_pi**2]]), np.array([[sig_S**2]])
    return kf

def nll_step1(params, S_long_obs):
    """Step 1 の尤度関数 (負の対数尤度)"""
    kf = setup_linear_kf(*params)
    u = np.array([[(1 - params[0]) * CONFIG['fixed_mu']]])
    return sum(-(kf.predict(u=u) or kf.update(np.array([[z]])) or kf.log_likelihood) for z in S_long_obs)

def extract_nowcast_path(S_long_obs, opt_params):
    """最適化されたパラメータを用いて、全期間のインフレ推論パス(pi_hat)を抽出します"""
    kf = setup_linear_kf(*opt_params)
    u = np.array([[(1 - opt_params[0]) * CONFIG['fixed_mu']]])
    return np.array([kf.predict(u=u) or kf.update(np.array([[z]])) or kf.x[0, 0] for z in S_long_obs])

def compute_exact_mn_experience(pi_hat_long, theta, ages, start_idx_short):
    """マクロの推論パスを用いて、各コーホートのインフレ経験 E_{c,t} を厳密に計算します"""
    E_pool = np.zeros((CONFIG['T_short'], len(ages)))
    for c, age_years in enumerate(ages):
        age_months = age_years * 12      
        L_c = age_months - 240           
        for t_short in range(CONFIG['T_short']):
            t_long = start_idx_short + t_short
            available_L = min(L_c, t_long) 
            if available_L <= 0:
                E_pool[t_short, c] = pi_hat_long[t_long]
                continue
            weights = (age_months - np.arange(1, available_L + 1)) ** theta
            window = pi_hat_long[t_long - available_L : t_long]
            E_pool[t_short, c] = np.sum((weights / np.sum(weights)) * window[::-1])
    return E_pool

# =====================================================================
# 3. Step 2: 拡張 UKF (非線形状態空間モデル)
# =====================================================================
def fx(x, dt, phi):
    """
    状態推移関数 (Transition Function)
    今月の見えない状態が、来月どう変化するかを定義します。
    """
    x_next = np.empty_like(x)
    # 1. インフレ率は AR(1) プロセスに従って長期目標へ回帰
    x_next[0] = (1 - phi) * CONFIG['fixed_mu'] + phi * x[0]
    
    # 2. 潜在変数 gamma_tilde と lambda_tilde はランダムウォーク (先月の状態をそのまま維持)
    # これにより、データの変化を通じてオプティマイザが「時変するパラメータ」として学習します。
    x_next[1:] = x[1:] 
    return x_next

def hx(x, E_t, Z_lag, phi, abs_delta_S, b_params, alpha):
    """
    非線形観測関数 (Observation Function)
    UKFのオプティマイザから受け取った仮のパラメータを用いて、理論上の観測データを生成します。
    """
    pi_t = x[0]
    z = np.zeros(DIM_Z)
    
    # サーベイ形成の基礎となる「1年先の合理的インフレ期待 (Rational Expectation)」
    rational_exp_1y = (1 - phi**12) * CONFIG['fixed_mu'] + (phi**12) * pi_t
    z[0] = pi_t
    
    for c in range(CONFIG['C']):
        # --- (A) 潜在変数の復元 ---
        # 状態ベクトルから無限大を推移する潜在変数を取り出し、シグモイド関数で 0~1 に押し込む
        gamma_tilde = x[1 + c] if CONFIG['heterogeneous_gamma'] else x[1]
        lambda_tilde = x[1 + num_gamma_states + c]
        
        gamma_c = 1.0 / (1.0 + np.exp(-gamma_tilde))
        lambda_c = 1.0 / (1.0 + np.exp(-lambda_tilde))
        
        # --- (B) サーベイ予想 Z_{c,t} の生成 ---
        # インフレ経験と合理的期待(1年先)のブレンドによるファンダメンタル予想 (Z*)
        Z_star = gamma_c * E_t[c] + (1 - gamma_c) * rational_exp_1y
        # 粘着情報 lambda を加味した最終的なサーベイ予想
        z[1 + c] = (1 - lambda_c) * Z_star + lambda_c * Z_lag[c]
        
        # --- (C) ロジスティックバイアスとプロキシデータの生成 ---
        # 観測されるマクロボラティリティ (abs_delta_S) が大きいほど、偶然同じ選択肢に留まる確率が減衰
        stay_prob = b_params[c] * (2.0 / (1.0 + np.exp(alpha * abs_delta_S)))
        
        # 観測データ(プロキシ)の理論生成 = 純粋なサボり割合 + (情報を更新したけど偶然留まった割合)
        expected_obs_lambda = lambda_c + (1 - lambda_c) * stay_prob
        z[1 + CONFIG['C'] + c] = expected_obs_lambda
        
    return z

def setup_ukf(opt_phi, opt_sig_pi, opt_sig_S, sig_gamma, sig_lambda, sig_Z, sig_lambda_obs):
    """UKFの初期化とプロセス・観測ノイズ行列の設定"""
    points = MerweScaledSigmaPoints(n=DIM_X, alpha=0.1, beta=2., kappa=3-DIM_X)
    ukf = UnscentedKalmanFilter(dim_x=DIM_X, dim_z=DIM_Z, dt=1., fx=fx, hx=hx, points=points)
    
    # シグモイド(0)=0.5 となるため、潜在変数の初期値は 0.0 を設定
    ukf.x = np.array([CONFIG['fixed_mu']] + [0.0] * (DIM_X - 1))
    
    # ★ 識別トラップ回避: 潜在変数(シグモイド前)の初期不確実性を 0.5**2 から 2.0**2 に大幅拡大
    # これにより、UKFが初期値(0.5)に固執せず、データのシグナルに合わせて一瞬で真のλへジャンプできる
    ukf.P = np.diag([0.5**2] + [2.0**2] * (DIM_X - 1))
    
    # プロセスノイズ (gamma と lambda それぞれの変動幅)
    ukf.Q = np.diag([opt_sig_pi**2] + [sig_gamma**2]*num_gamma_states + [sig_lambda**2]*CONFIG['C'])
    # 観測ノイズ (サーベイZ と プロキシデータ それぞれのノイズ)
    ukf.R = np.diag([opt_sig_S**2] + [sig_Z**2]*CONFIG['C'] + [sig_lambda_obs**2]*CONFIG['C'])
    return ukf

def nll_step2(params, S_short_obs, Z_short_obs, lambda_obs_short, Z_lag_short, E_pool, abs_delta_S_short, opt_phi, opt_sig_pi, opt_sig_S):
    """
    Step 2 の尤度関数
    minimize関数がここを何度も呼び出し、最適な alpha や b_c などのパラメータを探索します。
    """
    sig_gamma, sig_lambda, sig_Z, sig_lambda_obs, alpha = params[:5]
    b_params = params[5:5+CONFIG['C']] 
    
    ukf = setup_ukf(opt_phi, opt_sig_pi, opt_sig_S, sig_gamma, sig_lambda, sig_Z, sig_lambda_obs)
    ll_total = 0.0
    
    for t in range(len(S_short_obs)):
        z_t = np.zeros(DIM_Z)
        z_t[0] = S_short_obs[t]
        z_t[1:1+CONFIG['C']] = Z_short_obs[t]
        z_t[1+CONFIG['C']:] = lambda_obs_short[t] # 外部からプロキシデータを供給
        
        ukf.predict(phi=opt_phi)
        # 観測関数 hx の kwargs として推計中の alpha や b_params、マクロボラティリティを渡す
        ukf.update(z_t, E_t=E_pool[t], Z_lag=Z_lag_short[t], phi=opt_phi, 
                   abs_delta_S=abs_delta_S_short[t], b_params=b_params, alpha=alpha)
        ll_total += ukf.log_likelihood
    return -ll_total

# =====================================================================
# 4. メイン実行・内生変数の出力・可視化
# =====================================================================
if __name__ == "__main__":
    # データ準備
    S_data, S_short_obs, Z_short_obs, lambda_obs_short, Z_lag_short, abs_delta_S_short = generate_mock_data()
    
    print("--- 1. Step 1: マクロ推論パスの推計 ---")
    res1 = minimize(nll_step1, CONFIG['init_step1'], args=(S_data,), method='L-BFGS-B', bounds=CONFIG['bounds_step1'])
    opt_phi, opt_sig_pi, opt_sig_S = res1.x
    pi_hat_long = extract_nowcast_path(S_data, res1.x)
    E_pool = compute_exact_mn_experience(pi_hat_long, CONFIG['calibrated_theta'], CONFIG['ages'], CONFIG['start_idx_short'])
    
    print("--- 2. Step 2: UKF 推計 (時変λ と ロジスティックバイアスの識別) ---")
    res2 = minimize(nll_step2, CONFIG['init_step2'], 
                    args=(S_short_obs, Z_short_obs, lambda_obs_short, Z_lag_short, E_pool, abs_delta_S_short, opt_phi, opt_sig_pi, opt_sig_S),
                    method='L-BFGS-B', bounds=CONFIG['bounds_step2'])
    
    # 推計結果を展開
    opt_sig_gamma, opt_sig_lambda, opt_sig_Z, opt_sig_lambda_obs, opt_alpha = res2.x[:5]
    opt_b = res2.x[5:5+CONFIG['C']]
    
    # -----------------------------------------------------------------
    # ★ フィルタリングの再実行と【全・内生変数】の保存
    # -----------------------------------------------------------------
    ukf_final = setup_ukf(opt_phi, opt_sig_pi, opt_sig_S, opt_sig_gamma, opt_sig_lambda, opt_sig_Z, opt_sig_lambda_obs)
    records = []
    total_log_likelihood = 0.0
    
    # 5年先予想の粘着情報計算用に、初期値ラグを設定
    implied_Z5_lag = np.full(CONFIG['C'], 2.0)
    
    for t_short in range(CONFIG['T_short']):
        z_t = np.zeros(DIM_Z)
        z_t[0], z_t[1:1+CONFIG['C']], z_t[1+CONFIG['C']:] = S_short_obs[t_short], Z_short_obs[t_short], lambda_obs_short[t_short]
        
        ukf_final.predict(phi=opt_phi)
        ukf_final.update(z_t, E_t=E_pool[t_short], Z_lag=Z_lag_short[t_short], phi=opt_phi, 
                         abs_delta_S=abs_delta_S_short[t_short], b_params=opt_b, alpha=opt_alpha)
        total_log_likelihood += ukf_final.log_likelihood
        
        pi_t = ukf_final.x[0]
        
        # マクロモデルからインプライされる 1年先 および 5年先 の合理的インフレ期待
        rational_exp_1y = (1 - opt_phi**12) * CONFIG['fixed_mu'] + (opt_phi**12) * pi_t
        rational_exp_5y = (1 - opt_phi**60) * CONFIG['fixed_mu'] + (opt_phi**60) * pi_t
        
        record = {
            'Date': dates_short[t_short],
            'Obs_S': S_short_obs[t_short],
            'State_pi_t': pi_t,
            'Rational_Exp_1y': rational_exp_1y, 
            'Rational_Exp_5y': rational_exp_5y  
        }
        
        # コーホート別の内生変数をすべて展開して記録
        for c in range(CONFIG['C']):
            gamma_tilde = ukf_final.x[1 + c] if CONFIG['heterogeneous_gamma'] else ukf_final.x[1]
            lambda_tilde = ukf_final.x[1 + num_gamma_states + c]
            
            gamma_c = 1.0 / (1.0 + np.exp(-gamma_tilde)) 
            lambda_c = 1.0 / (1.0 + np.exp(-lambda_tilde))
            
            # ロジスティックバイアスとプロキシの理論値
            stay_prob = opt_b[c] * (2.0 / (1.0 + np.exp(opt_alpha * abs_delta_S_short[t_short])))
            expected_obs_lambda = lambda_c + (1 - lambda_c) * stay_prob
            
            # --- 【家計の1年先インフレ予想】の形成 ---
            Z_star_1y = gamma_c * E_pool[t_short, c] + (1 - gamma_c) * rational_exp_1y
            Household_Exp_1y = (1 - lambda_c) * Z_star_1y + lambda_c * Z_lag_short[t_short, c]
            
            # --- 【家計の5年先インフレ予想】の形成 ---
            # ※ インフレ経験 (E_pool) は過去の蓄積なので1年先と共通。合理的期待部分のみ 5y に置き換える
            Z_star_5y = gamma_c * E_pool[t_short, c] + (1 - gamma_c) * rational_exp_5y
            Household_Exp_5y = (1 - lambda_c) * Z_star_5y + lambda_c * implied_Z5_lag[c]
            implied_Z5_lag[c] = Household_Exp_5y # 次期のためにラグを更新
            
            # λ に関連する一連の内生変数をすべて記録
            record[f'State_gamma_{c+1}'] = gamma_c
            record[f'State_lambda_{c+1}'] = lambda_c
            record[f'Household_Exp_1y_{c+1}'] = Household_Exp_1y 
            record[f'Household_Exp_5y_{c+1}'] = Household_Exp_5y 
            record[f'Fund_Z_star_1y_{c+1}'] = Z_star_1y
            record[f'Fund_Z_star_5y_{c+1}'] = Z_star_5y
            record[f'Stay_Prob_{c+1}'] = stay_prob
            record[f'Expected_Obs_Lambda_{c+1}'] = expected_obs_lambda
            record[f'Actual_Obs_Lambda_{c+1}'] = lambda_obs_short[t_short, c]
            record[f'Obs_Z_{c+1}'] = Z_short_obs[t_short, c]
            record[f'Experience_E_{c+1}'] = E_pool[t_short, c] 
            
        records.append(record)
        
    # 内生変数の CSV 出力
    df_endogenous = pd.DataFrame(records).set_index('Date')
    df_endogenous.to_csv(CONFIG['output_csv'])
    
    # 推計パラメータの CSV 出力 (ルーチンの呼び出し)
    save_all_parameters(CONFIG, res1.x, res2.x, CONFIG['params_csv'])
    
    print("\n" + "="*50)
    print(" 📊 MODEL EVALUATION METRICS (In-Sample)")
    print("="*50)
    print(f" Total Log-Likelihood: {total_log_likelihood:.2f}")
    print(f" Estimated Alpha (Logistic Sensitivity) : {opt_alpha:.4f}")
    for c in range(CONFIG['C']):
        print(f" Estimated Base Stay Prob b_{c+1}        : {opt_b[c]:.4f}")
    print("="*50 + "\n")
    
    # -----------------------------------------------------------------
    # ★ グラフの描画 (4つのパネル構成)
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    title_suffix = "Heterogeneous" if CONFIG['heterogeneous_gamma'] else "Homogeneous"
    fig.suptitle(f'Extended UKF with Logistic Bias ({title_suffix} $\gamma$)', fontsize=16)

    # パネル1: マクロインフレ動学
    axes[0].plot(df_endogenous.index, df_endogenous['Obs_S'], label='Observed Signal $S_t$', color='gray', alpha=0.5)
    axes[0].plot(df_endogenous.index, df_endogenous['State_pi_t'], label='Filtered $\pi_t$', color='blue', linewidth=2)
    axes[0].axhline(CONFIG['fixed_mu'], color='red', linestyle='--', label='Target $\mu$')
    axes[0].set_title('Inflation Dynamics')
    axes[0].set_ylabel('Rate (%)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # パネル2: インフレ経験へのウエイト (gamma)
    if CONFIG['heterogeneous_gamma']:
        for c in range(CONFIG['C']):
            axes[1].plot(df_endogenous.index, df_endogenous[f'State_gamma_{c+1}'], label=f'Filtered $\gamma_{{{c+1},t}}$ ({CONFIG["ages"][c]} years)', linewidth=2)
    else:
        axes[1].plot(df_endogenous.index, df_endogenous['State_gamma_1'], label='Filtered $\gamma_t$ (Common)', color='green', linewidth=2)
        
    axes[1].set_title('Time-Varying Weights on Inflation Experience')
    axes[1].set_ylabel('Weight $\gamma$')
    axes[1].set_ylim(-0.1, 1.1) 
    axes[1].legend(loc='best', ncol=2, fontsize='small')
    axes[1].grid(True, alpha=0.3)

    # パネル3: 粘着情報 (lambda) とプロキシデータ
    for c in range(CONFIG['C']):
        # プロキシデータ(ノイズとバイアス込み)を薄い色で、推計された真のλを太い線でプロット
        axes[2].plot(df_endogenous.index, df_endogenous[f'Actual_Obs_Lambda_{c+1}'], color=f'C{c}', alpha=0.3, label=f'Proxy Obs (C{c+1})')
        axes[2].plot(df_endogenous.index, df_endogenous[f'State_lambda_{c+1}'], color=f'C{c}', linewidth=2, label=f'Filtered $\lambda_{{{c+1},t}}$')
        
    axes[2].set_title('Time-Varying Sticky Information ($\lambda_{c,t}$) vs Upward Biased Proxy Data')
    axes[2].set_ylabel('Sticky Info $\lambda$')
    axes[2].set_ylim(0, 1.1)
    axes[2].legend(loc='upper right', ncol=2, fontsize='small')
    axes[2].grid(True, alpha=0.3)

    # パネル4: サーベイ vs インプライド (1年先 & 5年先)
    c_idx = CONFIG['C'] # 最も高齢のコーホートのグラフを表示
    
    # 1年先のプロット
    axes[3].plot(df_endogenous.index, df_endogenous[f'Household_Exp_1y_{c_idx}'], label=f'Household 1y Exp (Cohort {c_idx})', color='darkorange', linewidth=2)
    axes[3].plot(df_endogenous.index, df_endogenous['Rational_Exp_1y'], color='teal', linestyle=':', label='Rational Exp (1y ahead)')
    
    # 5年先のプロット
    axes[3].plot(df_endogenous.index, df_endogenous[f'Household_Exp_5y_{c_idx}'], label=f'Household 5y Exp (Cohort {c_idx})', color='firebrick', linewidth=2)
    axes[3].plot(df_endogenous.index, df_endogenous['Rational_Exp_5y'], color='navy', linestyle=':', label='Rational Exp (5y ahead)')
    
    # インフレ経験のプロット
    axes[3].plot(df_endogenous.index, df_endogenous[f'Experience_E_{c_idx}'], label=f'MN Experience $E_{{{c_idx},t}}$', color='purple', linestyle='-', linewidth=1.5, alpha=0.5)
    
    axes[3].set_title(f'Household 1-year and 5-year Expectations (Cohort {c_idx}: {CONFIG["ages"][c_idx-1]} years old)')
    axes[3].set_ylabel('Rate (%)')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()
