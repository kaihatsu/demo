import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints

# =====================================================================
# 0. 基本設定と期間の動的計算 (Configuration)
# =====================================================================
CONFIG = {
    'C': 2,                 # コーホート数 (例: 3にすれば自動でZ_1, Z_2, Z_3が生成・推計される)
    'fixed_mu': 2.0,        # インフレ目標 (アンカー)
    'ages': [480, 720],     # 各コーホートの月齢 (40歳=480ヶ月, 60歳=720ヶ月)
    
    # --- 期間設定 (日付で指定) ---
    'start_date_long': '1954-04-01',  # 第一段階(超長期データ)の開始月
    'start_date_short': '2004-04-01', # 第二段階(サーベイデータ)の開始月
    'end_date': '2024-03-01',         # データの終了月
    
    'calibrated_theta': 1.0, # 記憶の減衰形状パラメータ (1.0 = 線形減衰)
    
    'mock_csv': 'mock_data_dynamic.csv',
    'output_csv': 'endogenous_variables_dynamic.csv',
    
    'init_step1': [0.5, 0.1, 0.2],               
    'bounds_step1': ((0.01, 0.99), (1e-4, 5.0), (1e-4, 5.0)),
    
    'init_step2': [0.05, 0.1],                   
    'bounds_step2': ((1e-4, 1.0), (1e-4, 5.0))
}

# --- 内生的な期間変数の算出 ---
# pd.date_range を用いて月次(MS: Month Start)の全日付インデックスを生成
global_dates = pd.date_range(start=CONFIG['start_date_long'], end=CONFIG['end_date'], freq='MS')

CONFIG['T_long'] = len(global_dates) # 全期間のデータ数 (例: 840ヶ月)
# 第二段階の開始月が、全期間の中で何番目のインデックスかを特定
CONFIG['start_idx_short'] = global_dates.get_loc(pd.to_datetime(CONFIG['start_date_short']))
CONFIG['T_short'] = CONFIG['T_long'] - CONFIG['start_idx_short'] # 第二段階のデータ数

# 動的なカラム名の生成 (例: ['Z_1', 'Z_2'])
Z_COLS = [f'Z_{c+1}' for c in range(CONFIG['C'])]


# =====================================================================
# 1. モックデータの生成 (構造変化を年代で指定 & ハードコード排除)
# =====================================================================
def generate_and_load_mock_data():
    print(f"--- 1. モックデータの生成 ({CONFIG['start_date_long']} 〜 {CONFIG['end_date']}) ---")
    np.random.seed(42)
    T_long = CONFIG['T_long']
    start_idx = CONFIG['start_idx_short']
    
    true_phi, true_sig_pi, true_sig_S = 0.8, 0.1, 0.2
    true_pi = np.zeros(T_long)
    true_pi[0] = CONFIG['fixed_mu']
    
    # マクロショックの発生期間を日付ベースで内生的に取得
    idx_shock1_start = global_dates.get_loc(pd.to_datetime('1973-10-01'), method='nearest')
    idx_shock1_end   = global_dates.get_loc(pd.to_datetime('1980-12-01'), method='nearest')
    idx_shock2_start = global_dates.get_loc(pd.to_datetime('1998-01-01'), method='nearest')
    idx_shock2_end   = global_dates.get_loc(pd.to_datetime('2012-12-01'), method='nearest')
    
    for t in range(1, T_long):
        true_pi[t] = (1 - true_phi) * CONFIG['fixed_mu'] + true_phi * true_pi[t-1] + np.random.normal(0, true_sig_pi)
        
        # 構造変化（歴史的ショック）の注入
        if idx_shock1_start <= t <= idx_shock1_end:
            true_pi[t] += 0.4  # オイルショック期の高インフレ
        elif idx_shock2_start <= t <= idx_shock2_end:
            true_pi[t] -= 0.4  # 長期デフレ期
            
    S_data = true_pi + np.random.normal(0, true_sig_S, T_long)
    
    true_gamma = np.zeros(T_long)
    true_gamma[0] = 0.5
    for t in range(1, T_long):
        true_gamma[t] = np.clip(true_gamma[t-1] + np.random.normal(0, 0.03), 0.1, 0.9)
        
    Z_data = np.full((T_long, CONFIG['C']), np.nan)
    
    for c, age in enumerate(CONFIG['ages']):
        L_c = age - 240
        k_arr = np.arange(1, L_c + 1)
        weights = (age - k_arr) ** CONFIG['calibrated_theta']
        weights = weights / np.sum(weights)
        
        for t in range(start_idx, T_long):
            window = true_pi[t - L_c : t]
            true_E_ct = np.sum(weights * window[::-1])
            true_rational_exp = (1 - true_phi**12) * CONFIG['fixed_mu'] + (true_phi**12) * true_pi[t]
            true_Z = true_gamma[t] * true_E_ct + (1 - true_gamma[t]) * true_rational_exp
            Z_data[t, c] = true_Z + np.random.normal(0, 0.15)
            
    # データフレームの動的構築 (C=2なら Z_1, Z_2 が、C=3なら Z_1, Z_2, Z_3 が作られる)
    df_dict = {'S_t': S_data, 'True_gamma': true_gamma}
    for c, col_name in enumerate(Z_COLS):
        df_dict[col_name] = Z_data[:, c]
        
    df_mock = pd.DataFrame(df_dict, index=global_dates)
    df_mock.index.name = 'Date'
    df_mock.to_csv(CONFIG['mock_csv'])
    print(f"モックデータ '{CONFIG['mock_csv']}' を保存しました。\n")
    
    # 読み込みとスライス
    df_loaded = pd.read_csv(CONFIG['mock_csv'], parse_dates=['Date'], index_col='Date')
    S_long_obs = df_loaded['S_t'].values
    
    df_short = df_loaded.iloc[start_idx:].copy()
    S_short_obs = df_short['S_t'].values
    Z_short_obs = df_short[Z_COLS].values # ハードコード排除
    
    return S_long_obs, S_short_obs, Z_short_obs, df_short.index, df_short['True_gamma'].values


# =====================================================================
# 2. Step 1: 線形KF (filterpy の仕様に基づく実装)
# =====================================================================
def setup_linear_kf(phi, sig_pi, sig_S):
    """
    filterpy の KalmanFilter の初期化。
    x = Fx + Bu + w (w ~ N(0, Q))
    z = Hx + v (v ~ N(0, R))
    """
    kf = KalmanFilter(dim_x=1, dim_z=1) # 状態も観測も1次元
    kf.x = np.array([[CONFIG['fixed_mu']]]) # 状態ベクトル x の初期値
    kf.P = np.array([[1.0]])                # 事前誤差共分散行列 P (初期の不確実性)
    kf.F = np.array([[phi]])                # 状態遷移マトリクス F
    kf.B = np.array([[1.0]])                # 制御入力マトリクス B
    kf.H = np.array([[1.0]])                # 観測マトリクス H
    kf.Q = np.array([[sig_pi**2]])          # プロセスノイズの共分散 Q
    kf.R = np.array([[sig_S**2]])           # 観測ノイズの共分散 R
    return kf

def nll_step1(params, S_long_obs):
    phi, sig_pi, sig_S = params
    kf = setup_linear_kf(phi, sig_pi, sig_S)
    u = np.array([[(1 - phi) * CONFIG['fixed_mu']]]) # コントロール入力ベクトル u
    
    ll_total = 0.0
    for z in S_long_obs:
        # predict: 事前状態 x_{t|t-1} と 事前共分散 P_{t|t-1} を計算
        kf.predict(u=u)
        # update: イノベーション(z - Hx)を計算し、カルマンゲインを用いて事後状態 x_{t|t} と P_{t|t} を更新
        kf.update(np.array([[z]]))
        # 対数尤度 (予測誤差の正規分布における確率密度) を累積
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
        pi_hats.append(kf.x[0, 0]) # 事後状態の期待値を保存
    return np.array(pi_hats)


# =====================================================================
# 3. インフレ経験 E_{c,t} の正確なプーリング
# =====================================================================
def compute_exact_mn_experience(pi_hat_long, theta, ages, start_idx_short):
    T_short = len(pi_hat_long) - start_idx_short
    C = len(ages)
    E_pool = np.zeros((T_short, C))
    
    for c, age in enumerate(ages):
        L_c = age - 240
        k_arr = np.arange(1, L_c + 1)
        weights = (age - k_arr) ** theta
        weights = weights / np.sum(weights)
        
        for t_short in range(T_short):
            t_long = start_idx_short + t_short
            window = pi_hat_long[t_long - L_c : t_long]
            E_pool[t_short, c] = np.sum(weights * window[::-1])
            
    return E_pool


# =====================================================================
# 4. Step 2: UKF (Unscented Kalman Filter の詳細設計)
# =====================================================================
dim_x = 2
# 観測変数の次元: S_t(1つ) + サーベイ(C個)
dim_z = 1 + CONFIG['C']

def fx(x, dt, phi):
    """
    UKFの状態遷移関数 f(x)。
    Unscented Transform において、複数のシグマポイントがこの関数を通過することで、
    非線形変換後の事後分布の平均と分散が近似される。
    """
    x_next = np.empty_like(x)
    x_next[0] = (1 - phi) * CONFIG['fixed_mu'] + phi * x[0]
    x_next[1] = x[1]
    return x_next

def hx(x, E_t, phi):
    """
    UKFの観測関数 h(x)。
    状態(x)から期待される観測理論値を返す。
    E_t のように時間経過で変わる外生変数は、ukf.update() の kwargs を経由して毎期受け取る。
    """
    pi_t, gamma_t = x[0], x[1]
    z = np.zeros(dim_z)
    rational_exp = (1 - phi**12) * CONFIG['fixed_mu'] + (phi**12) * pi_t
    
    z[0] = pi_t
    for c in range(CONFIG['C']):
        # Z_c の理論値 (動的コーホート処理)
        z[1 + c] = gamma_t * E_t[c] + (1 - gamma_t) * rational_exp
    return z

def nll_step2(params, S_short_obs, Z_short_obs, E_pool, opt_phi, opt_sig_pi, opt_sig_S):
    sig_gamma, sig_Z = params
    T_short = len(S_short_obs)
    
    # MerweScaledSigmaPoints: シグマポイント（代表点）の配置を決めるアルゴリズム。
    # alpha=0.1 (分布の広がり), beta=2 (ガウス分布に最適), kappa (スケーリング)
    points = MerweScaledSigmaPoints(n=dim_x, alpha=0.1, beta=2., kappa=3-dim_x)
    
    ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=1., fx=fx, hx=hx, points=points)
    ukf.x = np.array([CONFIG['fixed_mu'], 0.5])
    ukf.P *= 0.1
    ukf.Q = np.diag([opt_sig_pi**2, sig_gamma**2])
    
    # R行列の動的構築 (シグナル分散1つ + サーベイ分散C個)
    ukf.R = np.diag([opt_sig_S**2] + [sig_Z**2] * CONFIG['C'])
    
    ll_total = 0.0
    for t in range(T_short):
        z_t = np.zeros(dim_z)
        z_t[0] = S_short_obs[t]
        z_t[1:] = Z_short_obs[t]
        
        ukf.predict(phi=opt_phi)
        # E_t=E_pool[t] として外生変数を渡す。これは filterpy 内部を通ってそのまま hx に届く。
        ukf.update(z_t, E_t=E_pool[t], phi=opt_phi)
        ll_total += ukf.log_likelihood
        
    return -ll_total


# =====================================================================
# 5. メイン実行プロセスと動的プロット
# =====================================================================
if __name__ == "__main__":
    S_long_obs, S_short_obs, Z_short_obs, dates_short, true_gamma_short = generate_and_load_mock_data()
    
    print("--- 2. Step 1: インフレ動学パラメータの推計 ---")
    res1 = minimize(nll_step1, CONFIG['init_step1'], args=(S_long_obs,),
                    method='L-BFGS-B', bounds=CONFIG['bounds_step1'])
    opt_phi, opt_sig_pi, opt_sig_S = res1.x
    print(f"推計完了: phi={opt_phi:.4f}, sig_pi={opt_sig_pi:.4f}, sig_S={opt_sig_S:.4f}\n")
    
    pi_hat_long = extract_nowcast_path(S_long_obs, res1.x)
    
    print(f"--- 3. インフレ経験の正確な構築 (θ = {CONFIG['calibrated_theta']}) ---")
    E_pool = compute_exact_mn_experience(pi_hat_long, CONFIG['calibrated_theta'], 
                                         CONFIG['ages'], CONFIG['start_idx_short'])
    
    print("--- 4. Step 2: 経験ウエイト動学パラメータの推計 (UKF) ---")
    res2 = minimize(nll_step2, CONFIG['init_step2'], 
                    args=(S_short_obs, Z_short_obs, E_pool, opt_phi, opt_sig_pi, opt_sig_S),
                    method='L-BFGS-B', bounds=CONFIG['bounds_step2'])
    opt_sig_gamma, opt_sig_Z = res2.x
    print(f"推計完了: sig_gamma={opt_sig_gamma:.4f}, sig_Z={opt_sig_Z:.4f}\n")
    
    print("--- 5. 内生変数の抽出とCSV保存 ---")
    points = MerweScaledSigmaPoints(n=dim_x, alpha=0.1, beta=2., kappa=3-dim_x)
    ukf_final = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=1., fx=fx, hx=hx, points=points)
    ukf_final.x = np.array([CONFIG['fixed_mu'], 0.5])
    ukf_final.P *= 0.1
    ukf_final.Q = np.diag([opt_sig_pi**2, opt_sig_gamma**2])
    ukf_final.R = np.diag([opt_sig_S**2] + [opt_sig_Z**2] * CONFIG['C'])
    
    records = []
    T_short = CONFIG['T_short']
    
    for t_short in range(T_short):
        z_t = np.zeros(dim_z)
        z_t[0] = S_short_obs[t_short]
        z_t[1:] = Z_short_obs[t_short]
        
        ukf_final.predict(phi=opt_phi)
        ukf_final.update(z_t, E_t=E_pool[t_short], phi=opt_phi)
        
        filt_pi, filt_gamma = ukf_final.x
        implied_z = hx(ukf_final.x, E_pool[t_short], opt_phi)
        
        record = {
            'Date': dates_short[t_short],
            'Obs_S': S_short_obs[t_short],
            'State_pi_t': filt_pi,
            'State_gamma_t': filt_gamma,
            'True_gamma_t': true_gamma_short[t_short]
        }
        # C個のコーホートについて動的に辞書に登録
        for c in range(CONFIG['C']):
            record[f'Experience_E_{c+1}'] = E_pool[t_short, c]
            record[f'Obs_Z_{c+1}'] = Z_short_obs[t_short, c]
            record[f'Implied_Z_{c+1}'] = implied_z[1+c]
            
        records.append(record)
        
    df_endogenous = pd.DataFrame(records).set_index('Date')
    df_endogenous.to_csv(CONFIG['output_csv'])
    print(f"内生変数を '{CONFIG['output_csv']}' に保存しました。\n")
    
    # 6. 結果の可視化
    print("--- 6. 結果のプロットを表示します ---")
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle('2-Step Approach with Endogenous Dates & Dynamic Cohorts', fontsize=16)

    # パネル1
    axes[0].plot(df_endogenous.index, df_endogenous['Obs_S'], label='Observed Signal $S_t$', color='gray', alpha=0.5)
    axes[0].plot(df_endogenous.index, df_endogenous['State_pi_t'], label='Filtered $\pi_t$', color='blue', linewidth=2)
    axes[0].axhline(CONFIG['fixed_mu'], color='red', linestyle='--', label='Anchor $\mu$')
    axes[0].set_title('Inflation Dynamics')
    axes[0].set_ylabel('Rate (%)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # パネル2
    axes[1].plot(df_endogenous.index, df_endogenous['True_gamma_t'], label='True $\gamma_t$ (DGP)', color='black', linestyle='--', alpha=0.7)
    axes[1].plot(df_endogenous.index, df_endogenous['State_gamma_t'], label='Filtered $\gamma_t$', color='green', linewidth=2)
    axes[1].set_title('Weight on Inflation Experience ($\gamma_t$)')
    axes[1].set_ylabel('Weight $\gamma_t$')
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # パネル3: 若年層(C=1) と 高齢層(C=C) の比較を動的に描画
    # 最も若いコーホート (c=1)
    axes[2].plot(df_endogenous.index, df_endogenous['Experience_E_1'], 
                 label=f'MN Experience (Young, age {CONFIG["ages"][0]})', color='purple', linestyle='-', linewidth=2)
    # 最も高い年齢のコーホート (c=C)
    axes[2].plot(df_endogenous.index, df_endogenous[f'Experience_E_{CONFIG["C"]}'], 
                 label=f'MN Experience (Old, age {CONFIG["ages"][-1]})', color='darkorange', linestyle='--', linewidth=2)
    
    axes[2].plot(df_endogenous.index, df_endogenous[f'Obs_Z_{CONFIG["C"]}'], 
                 label=f'Observed Survey (Old)', color='gray', alpha=0.3)
                 
    axes[2].set_title('Heterogeneous Inflation Experiences across Cohorts')
    axes[2].set_ylabel('Rate (%)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
