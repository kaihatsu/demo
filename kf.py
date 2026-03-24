import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints

# =====================================================================
# 0. 基本設定 (Configuration)
# =====================================================================
CONFIG = {
    'C': 2,                 # コーホート数 (例: 2なら若年層と高齢層)
    'fixed_mu': 2.0,        # インフレ定常値 (インフレ経験)
    'ages': [480, 720],     # 各コーホートの月齢 (例: 40歳=480ヶ月, 60歳=720ヶ月)
    
    # --- ★ gamma の異質性スイッチ ---
    # False: 全コーホート共通の単一の gamma_t を推計 (次元は 2)
    # True : コーホートごとに独立した gamma_{c,t} を推計 (次元は 1 + C)
    'heterogeneous_gamma': True, 
    
    # --- 期間設定 (日付指定) ---
    'start_date_long': '1954-04-01',  # 高齢層の記憶をカバーするため十分に過去から開始
    'start_date_short': '2004-04-01', # サーベイデータの開始月
    'end_date': '2024-03-01',         # データの終了月
    
    'calibrated_theta': 1.5,          # 記憶の減衰パラメータ (MNに準拠した緩やかな減衰)
    
    # ファイルパス
    'mock_csv': 'mock_data_final.csv',
    'output_csv': 'endogenous_variables_final.csv',
    
    # 最適化の初期推測値と境界
    'init_step1': [0.5, 0.1, 0.2],               # [phi, sigma_pi, sigma_S]
    'bounds_step1': ((0.01, 0.99), (1e-4, 5.0), (1e-4, 5.0)),
    
    'init_step2': [0.05, 0.1],                   # [sigma_gamma, sigma_Z]
    'bounds_step2': ((1e-4, 1.0), (1e-4, 5.0))
}

# --- 次元数と期間の動的計算 ---
DIM_X = 1 + CONFIG['C'] if CONFIG['heterogeneous_gamma'] else 2
DIM_Z = 1 + CONFIG['C']

# pd.date_range を用いて月次(MS)の全日付インデックスを生成
global_dates = pd.date_range(start=CONFIG['start_date_long'], end=CONFIG['end_date'], freq='MS')
CONFIG['T_long'] = len(global_dates)
CONFIG['start_idx_short'] = global_dates.get_loc(CONFIG['start_date_short'])
CONFIG['T_short'] = CONFIG['T_long'] - CONFIG['start_idx_short']
dates_short = global_dates[CONFIG['start_idx_short']:]

# =====================================================================
# 1. モックデータの生成・保存と読み込み
# =====================================================================
def generate_and_save_mock_data():
    print(f"--- 1-a. モックデータの生成と保存 ({CONFIG['start_date_long']} 〜 {CONFIG['end_date']}) ---")
    np.random.seed(42)
    T_long = CONFIG['T_long']
    start_idx = CONFIG['start_idx_short']
    
    true_phi, true_sig_pi, true_sig_S = 0.8, 0.1, 0.2
    true_pi = np.zeros(T_long)
    true_pi[0] = CONFIG['fixed_mu']
    
    # マクロショックの発生期間を取得 (文字列完全一致)
    idx_shock1_start = global_dates.get_loc('1973-10-01')
    idx_shock1_end   = global_dates.get_loc('1980-12-01')
    idx_shock2_start = global_dates.get_loc('1998-01-01')
    idx_shock2_end   = global_dates.get_loc('2012-12-01')
    
    for t in range(1, T_long):
        true_pi[t] = (1 - true_phi) * CONFIG['fixed_mu'] + true_phi * true_pi[t-1] + np.random.normal(0, true_sig_pi)
        if idx_shock1_start <= t <= idx_shock1_end: true_pi[t] += 0.4
        elif idx_shock2_start <= t <= idx_shock2_end: true_pi[t] -= 0.4
            
    S_data = true_pi + np.random.normal(0, true_sig_S, T_long)
    
    # 真の gamma の生成 (DGP: 0.1~0.9 の間にクリップ)
    if CONFIG['heterogeneous_gamma']:
        true_gamma = np.zeros((T_long, CONFIG['C']))
        true_gamma[0, :] = 0.5
        for t in range(1, T_long):
            for c in range(CONFIG['C']):
                true_gamma[t, c] = np.clip(true_gamma[t-1, c] + np.random.normal(0, 0.03), 0.1, 0.9)
    else:
        true_gamma_single = np.zeros(T_long)
        true_gamma_single[0] = 0.5
        for t in range(1, T_long):
            true_gamma_single[t] = np.clip(true_gamma_single[t-1] + np.random.normal(0, 0.03), 0.1, 0.9)
        true_gamma = np.tile(true_gamma_single, (CONFIG['C'], 1)).T
        
    Z_data = np.full((T_long, CONFIG['C']), np.nan)
    
    for c, age in enumerate(CONFIG['ages']):
        L_c = age - 240
        for t in range(start_idx, T_long):
            available_L = min(L_c, t)
            if available_L <= 0: continue
            
            k_arr = np.arange(1, available_L + 1)
            weights = (age - k_arr) ** CONFIG['calibrated_theta']
            weights = weights / np.sum(weights)
            
            window = true_pi[t - available_L : t]
            true_E_ct = np.sum(weights * window[::-1])
            true_rational_exp = (1 - true_phi**12) * CONFIG['fixed_mu'] + (true_phi**12) * true_pi[t]
            
            true_Z = true_gamma[t, c] * true_E_ct + (1 - true_gamma[t, c]) * true_rational_exp
            Z_data[t, c] = true_Z + np.random.normal(0, 0.15)
            
    df_dict = {'S_t': S_data}
    for c in range(CONFIG['C']):
        df_dict[f'Z_{c+1}'] = Z_data[:, c]
        df_dict[f'True_gamma_{c+1}'] = true_gamma[:, c]
        
    df_mock = pd.DataFrame(df_dict, index=global_dates)
    df_mock.index.name = 'Date'
    df_mock.to_csv(CONFIG['mock_csv'])
    print(f"モックデータを '{CONFIG['mock_csv']}' に保存しました。\n")

def load_mock_data():
    print(f"--- 1-b. モックデータの読み込み ---")
    df = pd.read_csv(CONFIG['mock_csv'], parse_dates=['Date'], index_col='Date')
    
    S_long_obs = df['S_t'].values
    df_short = df.iloc[CONFIG['start_idx_short']:].copy()
    S_short_obs = df_short['S_t'].values
    Z_short_obs = df_short[[f'Z_{c+1}' for c in range(CONFIG['C'])]].values
    
    return S_long_obs, S_short_obs, Z_short_obs, df_short

# =====================================================================
# 2. Step 1: 線形KFによる推論パス抽出
# =====================================================================
def setup_linear_kf(phi, sig_pi, sig_S):
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([[CONFIG['fixed_mu']]])
    kf.P = np.array([[1.0]])
    kf.F = np.array([[phi]])
    kf.B = np.array([[1.0]])
    kf.H = np.array([[1.0]])
    kf.Q = np.array([[sig_pi**2]])
    kf.R = np.array([[sig_S**2]])
    return kf

def nll_step1(params, S_long_obs):
    phi, sig_pi, sig_S = params
    kf = setup_linear_kf(phi, sig_pi, sig_S)
    u = np.array([[(1 - phi) * CONFIG['fixed_mu']]])
    ll_total = 0.0
    for z in S_long_obs:
        kf.predict(u=u)
        kf.update(np.array([[z]]))
        ll_total += kf.log_likelihood
    return -ll_total

def extract_nowcast_path(S_long_obs, opt_params):
    phi, sig_pi, sig_S = opt_params
    kf = setup_linear_kf(phi, sig_pi, sig_S)
    u = np.array([[(1 - phi) * CONFIG['fixed_mu']]])
    pi_hats = []
    for z in S_long_obs:
        kf.predict(u=u)
        kf.update(np.array([[z]]))
        pi_hats.append(kf.x[0, 0])
    return np.array(pi_hats)

# =====================================================================
# 3. インフレ経験 E_{c,t} の構築 (トランケーション安全装置付き)
# =====================================================================
def compute_exact_mn_experience(pi_hat_long, theta, ages, start_idx_short):
    T_short = len(pi_hat_long) - start_idx_short
    C = len(ages)
    E_pool = np.zeros((T_short, C))
    
    for c, age in enumerate(ages):
        L_c = age - 240
        for t_short in range(T_short):
            t_long = start_idx_short + t_short
            
            # データ不足時の打ち切り処理
            available_L = min(L_c, t_long)
            if available_L <= 0:
                E_pool[t_short, c] = pi_hat_long[t_long]
                continue
            
            k_arr = np.arange(1, available_L + 1)
            weights = (age - k_arr) ** theta
            weights = weights / np.sum(weights)
            
            window = pi_hat_long[t_long - available_L : t_long]
            E_pool[t_short, c] = np.sum(weights * window[::-1])
            
    return E_pool

# =====================================================================
# 4. Step 2: UKF (シグモイド変換とベストプラクティスP0の実装)
# =====================================================================
def fx(x, dt, phi):
    """状態遷移関数: 潜在変数 x[1:] は無限の範囲をランダムウォークする"""
    x_next = np.empty_like(x)
    x_next[0] = (1 - phi) * CONFIG['fixed_mu'] + phi * x[0]
    x_next[1:] = x[1:] 
    return x_next

def hx(x, E_t, phi):
    """観測関数: 潜在変数 x[1:] をシグモイド変換して 0.0〜1.0 にマッピング"""
    pi_t = x[0]
    z = np.zeros(DIM_Z)
    rational_exp = (1 - phi**12) * CONFIG['fixed_mu'] + (phi**12) * pi_t
    
    z[0] = pi_t
    for c in range(CONFIG['C']):
        # 潜在変数の取得
        gamma_tilde = x[1 + c] if CONFIG['heterogeneous_gamma'] else x[1]
        
        # ★ シグモイド変換 (実数の gamma に変換)
        gamma_c = 1.0 / (1.0 + np.exp(-gamma_tilde))
        
        z[1 + c] = gamma_c * E_t[c] + (1 - gamma_c) * rational_exp
    return z

def setup_ukf(opt_phi, opt_sig_pi, opt_sig_S, sig_gamma, sig_Z):
    """UKFの初期化"""
    points = MerweScaledSigmaPoints(n=DIM_X, alpha=0.1, beta=2., kappa=3-DIM_X)
    ukf = UnscentedKalmanFilter(dim_x=DIM_X, dim_z=DIM_Z, dt=1., fx=fx, hx=hx, points=points)
    
    # ★ 潜在変数 gamma_tilde の初期値は 0.0 とする (sigmoid(0.0) = 0.5)
    ukf.x = np.array([CONFIG['fixed_mu']] + [0.0] * (DIM_X - 1))
    
    # ★ 初期分散: pi_t は標準偏差0.5、潜在変数 gamma_tilde は標準偏差0.5程度に設定
    ukf.P = np.diag([0.5**2] + [0.5**2] * (DIM_X - 1))
    
    ukf.Q = np.diag([opt_sig_pi**2] + [sig_gamma**2] * (DIM_X - 1))
    ukf.R = np.diag([opt_sig_S**2] + [sig_Z**2] * CONFIG['C'])
    
    return ukf

def nll_step2(params, S_short_obs, Z_short_obs, E_pool, opt_phi, opt_sig_pi, opt_sig_S):
    sig_gamma, sig_Z = params
    T_short = len(S_short_obs)
    ukf = setup_ukf(opt_phi, opt_sig_pi, opt_sig_S, sig_gamma, sig_Z)
    
    ll_total = 0.0
    for t in range(T_short):
        z_t = np.zeros(DIM_Z)
        z_t[0] = S_short_obs[t]
        z_t[1:] = Z_short_obs[t]
        
        ukf.predict(phi=opt_phi)
        ukf.update(z_t, E_t=E_pool[t], phi=opt_phi)
        ll_total += ukf.log_likelihood
        
    return -ll_total

# =====================================================================
# メイン実行・出力・可視化ブロック
# =====================================================================
if __name__ == "__main__":
    # 1. データの生成と読み込み
    generate_and_save_mock_data()
    S_long_obs, S_short_obs, Z_short_obs, df_short = load_mock_data()
    
    # 2. Step 1: マクロ推論抽出
    print("--- 2. Step 1: マクロ推論パスの推計と抽出 ---")
    res1 = minimize(nll_step1, CONFIG['init_step1'], args=(S_long_obs,), method='L-BFGS-B', bounds=CONFIG['bounds_step1'])
    opt_phi, opt_sig_pi, opt_sig_S = res1.x
    print(f"Step 1 完了: phi={opt_phi:.4f}, sig_pi={opt_sig_pi:.4f}, sig_S={opt_sig_S:.4f}\n")
    
    pi_hat_long = extract_nowcast_path(S_long_obs, res1.x)
    
    # 3. 経験プール構築
    print("--- 3. インフレ経験 E_ct の構築 ---")
    E_pool = compute_exact_mn_experience(pi_hat_long, CONFIG['calibrated_theta'], CONFIG['ages'], CONFIG['start_idx_short'])
    print("経験プールの構築完了。\n")
    
    # 4. Step 2: UKFによるTVP推計
    print(f"--- 4. Step 2: UKF 推計 (異質性モデル: {CONFIG['heterogeneous_gamma']}) ---")
    res2 = minimize(nll_step2, CONFIG['init_step2'], 
                    args=(S_short_obs, Z_short_obs, E_pool, opt_phi, opt_sig_pi, opt_sig_S),
                    method='L-BFGS-B', bounds=CONFIG['bounds_step2'])
    opt_sig_gamma, opt_sig_Z = res2.x
    print(f"Step 2 完了: sig_gamma={opt_sig_gamma:.4f}, sig_Z={opt_sig_Z:.4f}\n")
    
    # 5. 内生変数の最終抽出とCSV保存
    print("--- 5. 全内生変数の抽出とファイル保存 ---")
    ukf_final = setup_ukf(opt_phi, opt_sig_pi, opt_sig_S, opt_sig_gamma, opt_sig_Z)
    records = []
    
    for t_short in range(CONFIG['T_short']):
        z_t = np.zeros(DIM_Z)
        z_t[0] = S_short_obs[t_short]
        z_t[1:] = Z_short_obs[t_short]
        
        ukf_final.predict(phi=opt_phi)
        ukf_final.update(z_t, E_t=E_pool[t_short], phi=opt_phi)
        
        implied_z = hx(ukf_final.x, E_pool[t_short], opt_phi)
        
        record = {
            'Date': dates_short[t_short],
            'Obs_S': S_short_obs[t_short],
            'State_pi_t': ukf_final.x[0],
            'Rational_Exp': (1 - opt_phi**12) * CONFIG['fixed_mu'] + (opt_phi**12) * ukf_final.x[0]
        }
        
        for c in range(CONFIG['C']):
            gamma_tilde = ukf_final.x[1 + c] if CONFIG['heterogeneous_gamma'] else ukf_final.x[1]
            
            # ★ 抽出時にシグモイド変換を適用し、実数の gamma を保存
            gamma_val = 1.0 / (1.0 + np.exp(-gamma_tilde))
            
            record[f'State_gamma_{c+1}'] = gamma_val
            record[f'Experience_E_{c+1}'] = E_pool[t_short, c]
            record[f'Obs_Z_{c+1}'] = Z_short_obs[t_short, c]
            record[f'Implied_Z_{c+1}'] = implied_z[1 + c]
            record[f'True_gamma_{c+1}'] = df_short[f'True_gamma_{c+1}'].iloc[t_short]
            
        records.append(record)
        
    df_endogenous = pd.DataFrame(records).set_index('Date')
    df_endogenous.to_csv(CONFIG['output_csv'])
    print(f"内生変数を '{CONFIG['output_csv']}' に保存しました。\n")
    
    # 6. 結果の可視化 (3パネル構成)
    print("--- 6. 推計結果のプロットを表示します ---")
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    title_suffix = "Heterogeneous" if CONFIG['heterogeneous_gamma'] else "Homogeneous"
    fig.suptitle(f'2-Step Approach: Endogenous Variables ({title_suffix} $\gamma$)', fontsize=16)

    # パネル1: インフレ動学
    axes[0].plot(df_endogenous.index, df_endogenous['Obs_S'], label='Observed Signal $S_t$', color='gray', alpha=0.5)
    axes[0].plot(df_endogenous.index, df_endogenous['State_pi_t'], label='Filtered $\pi_t$', color='blue', linewidth=2)
    # 用語指定に基づき Anchor ではなく Inflation Experience を使用
    axes[0].axhline(CONFIG['fixed_mu'], color='red', linestyle='--', label='Inflation Experience $\mu$')
    axes[0].set_title('Inflation Dynamics')
    axes[0].set_ylabel('Rate (%)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # パネル2: ウエイト(gamma)の動学
    if CONFIG['heterogeneous_gamma']:
        for c in range(CONFIG['C']):
            # f-string の中での LaTeX の中括弧エスケープ ({{{...}}}) を適用
            axes[1].plot(df_endogenous.index, df_endogenous[f'State_gamma_{c+1}'], label=f'Filtered $\gamma_{{{c+1},t}}$ (Cohort {c+1})', linewidth=2)
            axes[1].plot(df_endogenous.index, df_endogenous[f'True_gamma_{c+1}'], color='black', linestyle=':', alpha=0.5, label=f'True $\gamma_{{{c+1},t}}$')
        axes[1].set_title('Heterogeneous Weights on Inflation Experience')
    else:
        axes[1].plot(df_endogenous.index, df_endogenous['State_gamma_1'], label='Filtered $\gamma_t$ (Common)', color='green', linewidth=2)
        axes[1].plot(df_endogenous.index, df_endogenous['True_gamma_1'], label='True $\gamma_t$', color='black', linestyle=':', alpha=0.5)
        axes[1].set_title('Homogeneous Weight on Inflation Experience')
        
    axes[1].set_ylabel('Weight $\gamma$')
    axes[1].set_ylim(-0.1, 1.1) # 見切れ防止のためのゆとり
    axes[1].legend(loc='best', ncol=2, fontsize='small') # 凡例の被り防止
    axes[1].grid(True, alpha=0.3)

    # パネル3: 若年層(1) vs 高齢層(C) のコーホート間比較
    axes[2].plot(df_endogenous.index, df_endogenous['Experience_E_1'], label=f'MN Experience (Young, age {CONFIG["ages"][0]})', color='purple', linestyle='-', linewidth=2)
    axes[2].plot(df_endogenous.index, df_endogenous[f'Experience_E_{CONFIG["C"]}'], label=f'MN Experience (Old, age {CONFIG["ages"][-1]})', color='darkorange', linestyle='--', linewidth=2)
    axes[2].plot(df_endogenous.index, df_endogenous['Obs_Z_1'], label='Observed Survey (Young)', color='purple', alpha=0.3)
    axes[2].plot(df_endogenous.index, df_endogenous[f'Obs_Z_{CONFIG["C"]}'], label='Observed Survey (Old)', color='darkorange', alpha=0.3)
    axes[2].set_title('Heterogeneous Inflation Experiences across Cohorts')
    axes[2].set_ylabel('Rate (%)')
    axes[2].legend(loc='best', ncol=2, fontsize='small')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()
