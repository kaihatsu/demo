import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints

# =====================================================================
# 0. 基本設定とカリブレーション
# =====================================================================
CONFIG = {
    'C': 2,                 # コーホート数
    'fixed_mu': 2.0,        # インフレ定常値 (目標値)
    
    # 年齢設定 (月齢): 40歳(480ヶ月)と60歳(720ヶ月)
    # ※60歳の場合、20歳(240ヶ月)以降の記憶の長さは L_c = 480ヶ月 となる
    'ages': [480, 720],     
    
    # 期間設定 (月次)
    'T_long': 840,          # 第一段階の超長期データ長 (70年分)
    'T_short': 240,         # 第二段階の家計サーベイデータ長 (直近20年分)
    
    # カリブレートされるパラメータ
    'calibrated_theta': 3.0, # 記憶の減衰形状パラメータ (MNの定式化に準拠)
    
    # ファイルパス
    'mock_csv': 'mock_data.csv',
    'output_csv': 'endogenous_variables.csv',
    
    # 最適化の初期推測値と境界
    'init_step1': [0.5, 0.1, 0.2],               # [phi, sigma_pi, sigma_S]
    'bounds_step1': ((0.01, 0.99), (1e-4, 5.0), (1e-4, 5.0)),
    
    'init_step2': [0.05, 0.1],                   # [sigma_gamma, sigma_Z]
    'bounds_step2': ((1e-4, 1.0), (1e-4, 5.0))
}

# サーベイデータが開始する超長期データ上のインデックス
CONFIG['start_idx_short'] = CONFIG['T_long'] - CONFIG['T_short']

# =====================================================================
# 1. モックデータの生成・保存・読み込み (欠損値を含む現実的なデータセット)
# =====================================================================
def generate_and_load_mock_data():
    print("--- 1. モックデータの生成と保存 ---")
    np.random.seed(42)
    T_long = CONFIG['T_long']
    start_idx = CONFIG['start_idx_short']
    
    # 真のインフレ率 pi_t の生成 (真のphi=0.8)
    true_phi, true_sig_pi, true_sig_S = 0.8, 0.1, 0.2
    true_pi = np.zeros(T_long)
    true_pi[0] = CONFIG['fixed_mu']
    for t in range(1, T_long):
        true_pi[t] = (1 - true_phi) * CONFIG['fixed_mu'] + true_phi * true_pi[t-1] + np.random.normal(0, true_sig_pi)
        
    # 全期間(70年分)のシグナル S_t
    S_data = true_pi + np.random.normal(0, true_sig_S, T_long)
    
    # サーベイデータ Z_c,t (直近20年分のみ存在し、それ以前はNaNとする)
    Z_data = np.full((T_long, CONFIG['C']), np.nan)
    for c in range(CONFIG['C']):
        # 真値周辺にノイズを乗せてダミー生成
        Z_data[start_idx:, c] = true_pi[start_idx:] + np.random.normal(0, 0.3, CONFIG['T_short']) 
        
    # CSVに書き出し (NaNが含まれる現実的なフォーマット)
    df_mock = pd.DataFrame({'S_t': S_data, 'Z_1': Z_data[:, 0], 'Z_2': Z_data[:, 1]})
    df_mock.to_csv(CONFIG['mock_csv'], index_label='Time')
    print(f"モックデータ '{CONFIG['mock_csv']}' を生成しました。\n")
    
    # --- ファイルから再読み込み ---
    print("--- データをファイルから読み込みます ---")
    df_loaded = pd.read_csv(CONFIG['mock_csv'], index_col='Time')
    
    # 第一段階用: S_t は全期間取得
    S_long_obs = df_loaded['S_t'].values
    # 第二段階用: Z_c,t と S_t を NaN が無い期間(後半20年)のみスライスして取得
    df_short = df_loaded.iloc[start_idx:].copy()
    S_short_obs = df_short['S_t'].values
    Z_short_obs = df_short[['Z_1', 'Z_2']].values
    
    return S_long_obs, S_short_obs, Z_short_obs

# =====================================================================
# 2. Step 1: 線形KFによるインフレ動学推計 (filterpy詳細解説付き)
# =====================================================================
def setup_linear_kf(phi, sig_pi, sig_S):
    """
    filterpy.kalman.KalmanFilter の標準的なセットアップ。
    状態方程式: x = Fx + Bu + w (w ~ N(0, Q))
    観測方程式: z = Hx + v (v ~ N(0, R))
    """
    kf = KalmanFilter(dim_x=1, dim_z=1)
    
    kf.x = np.array([[CONFIG['fixed_mu']]]) # x: 状態ベクトルの初期値 (1x1行列)
    kf.P = np.array([[1.0]])                # P: 初期状態の不確実性 (共分散行列)
    kf.F = np.array([[phi]])                # F: 状態遷移マトリクス (AR(1)係数)
    
    # Bとuを用いた定数項 (1-phi)*mu の実装
    # filterpyでは kf.predict(u=...) と渡すことで、Fx + Bu が計算される
    kf.B = np.array([[1.0]])                # B: コントロール入力マトリクス
    
    kf.H = np.array([[1.0]])                # H: 観測マトリクス (S_t = pi_t)
    kf.Q = np.array([[sig_pi**2]])          # Q: プロセスノイズの共分散行列
    kf.R = np.array([[sig_S**2]])           # R: 観測ノイズの共分散行列
    
    return kf

def nll_step1(params, S_long_obs):
    phi, sig_pi, sig_S = params
    kf = setup_linear_kf(phi, sig_pi, sig_S)
    u = np.array([[(1 - phi) * CONFIG['fixed_mu']]]) # コントロール入力ベクトル
    
    ll_total = 0.0
    for z in S_long_obs:
        kf.predict(u=u)
        kf.update(np.array([[z]]))
        ll_total += kf.log_likelihood       # 各ステップの対数尤度を累積
    return -ll_total

def extract_nowcast_path(S_long_obs, opt_params):
    phi, sig_pi, sig_S = opt_params
    kf = setup_linear_kf(phi, sig_pi, sig_S)
    u = np.array([[(1 - phi) * CONFIG['fixed_mu']]])
    
    pi_hats = []
    for z in S_long_obs:
        kf.predict(u=u)
        kf.update(np.array([[z]]))
        pi_hats.append(kf.x[0, 0])          # 事後推定値 pi_{t|t} を抽出
    return np.array(pi_hats)

# =====================================================================
# 3. インフレ経験 E_{c,t} の正確なプーリング (Malmendier & Nagel)
# =====================================================================
def compute_exact_mn_experience(pi_hat_long, theta, ages, start_idx_short):
    T_short = len(pi_hat_long) - start_idx_short
    C = len(ages)
    E_pool = np.zeros((T_short, C))
    
    for c, age in enumerate(ages):
        L_c = age - 240 # 20歳(240ヶ月)以降の記憶の窓幅
        
        # k=1 から L_c までの配列を作成し、MNウエイトを計算
        k_arr = np.arange(1, L_c + 1)
        weights = (L_c - k_arr) ** theta
        weights = weights / np.sum(weights) # 合計を1に規格化
        
        for t_short in range(T_short):
            t_long = start_idx_short + t_short
            
            # 過去 L_c ヶ月分の pi_hat を抽出し、最新(1ヶ月前)が先頭になるよう逆順にする
            window = pi_hat_long[t_long - L_c : t_long]
            window_reversed = window[::-1]
            
            # ウエイトを掛けて加重平均 (内積)
            E_pool[t_short, c] = np.sum(weights * window_reversed)
            
    return E_pool

# =====================================================================
# 4. Step 2: UKFの定義と実行 (filterpy詳細解説付き)
# =====================================================================
dim_x = 2                # 状態変数: [pi_t, gamma_t]
dim_z = 1 + CONFIG['C']  # 観測変数: [S_t, Z_1, Z_2]

def fx(x, dt, phi):
    """
    UKFの状態遷移関数。
    filterpyの仕様により dt (時間ステップ) が必須引数となる。
    外生引数 phi は、ukf.predict(phi=...) の形で渡される。
    """
    x_next = np.empty_like(x)
    x_next[0] = (1 - phi) * CONFIG['fixed_mu'] + phi * x[0]
    x_next[1] = x[1] # ランダムウォーク
    return x_next

def hx(x, E_t, phi):
    """
    UKFの非線形観測関数。
    外生変数 E_t と phi は、ukf.update(z, E_t=..., phi=...) の形で渡される。
    """
    pi_t, gamma_t = x[0], x[1]
    z = np.zeros(dim_z)
    phi_12 = phi ** 12
    rational_exp = (1 - phi_12) * CONFIG['fixed_mu'] + phi_12 * pi_t
    
    z[0] = pi_t
    for c in range(CONFIG['C']):
        z[1 + c] = gamma_t * E_t[c] + (1 - gamma_t) * rational_exp
    return z

def nll_step2(params, S_short_obs, Z_short_obs, E_pool, opt_phi, opt_sig_pi, opt_sig_S):
    sig_gamma, sig_Z = params
    T_short = len(S_short_obs)
    
    # MerweScaledSigmaPoints: UKFで用いる代表点(シグマポイント)の生成器
    points = MerweScaledSigmaPoints(n=dim_x, alpha=0.1, beta=2., kappa=3-dim_x)
    
    # UnscentedKalmanFilterの初期化。定義した fx と hx を渡す。
    ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=1., fx=fx, hx=hx, points=points)
    
    ukf.x = np.array([CONFIG['fixed_mu'], 0.5])
    ukf.P *= 0.1
    
    # Q: プロセスノイズ (Step 1の固定値と今回の推計対象を組み合わせる)
    ukf.Q = np.diag([opt_sig_pi**2, sig_gamma**2])
    # R: 観測ノイズ (同様に組み合わせる)
    ukf.R = np.diag([opt_sig_S**2] + [sig_Z**2] * CONFIG['C'])
    
    ll_total = 0.0
    for t in range(T_short):
        z_t = np.zeros(dim_z)
        z_t[0] = S_short_obs[t]
        z_t[1:] = Z_short_obs[t]
        
        # predictステップ: fx が呼び出され、事前状態が計算される
        ukf.predict(phi=opt_phi)
        
        # updateステップ: hx が呼び出され、観測データ z_t と比較して事後状態が更新される
        # kwargs として E_t=E_pool[t] を渡すことで、その期の経験が hx の中で利用される
        ukf.update(z_t, E_t=E_pool[t], phi=opt_phi)
        
        ll_total += ukf.log_likelihood
        
    return -ll_total

# =====================================================================
# 5. メイン実行プロセス
# =====================================================================
if __name__ == "__main__":
    # 1. データ準備
    S_long_obs, S_short_obs, Z_short_obs = generate_and_load_mock_data()
    
    # 2. Step 1 (インフレ動学パラメータ推計)
    print("--- 2. Step 1: インフレ動学パラメータの推計 (超長期データ) ---")
    res1 = minimize(nll_step1, CONFIG['init_step1'], args=(S_long_obs,),
                    method='L-BFGS-B', bounds=CONFIG['bounds_step1'])
    opt_phi, opt_sig_pi, opt_sig_S = res1.x
    print(f"推計完了: phi={opt_phi:.4f}, sig_pi={opt_sig_pi:.4f}, sig_S={opt_sig_S:.4f}\n")
    
    pi_hat_long = extract_nowcast_path(S_long_obs, res1.x)
    
    # 3. 経験プールの構築
    print(f"--- 3. インフレ経験の正確な構築 (θ = {CONFIG['calibrated_theta']}) ---")
    E_pool = compute_exact_mn_experience(pi_hat_long, CONFIG['calibrated_theta'], 
                                         CONFIG['ages'], CONFIG['start_idx_short'])
    print("経験プールの構築完了。\n")
    
    # 4. Step 2 (経験ウエイトパラメータ推計)
    print("--- 4. Step 2: 経験ウエイト動学パラメータの推計 (サーベイデータ期) ---")
    res2 = minimize(nll_step2, CONFIG['init_step2'], 
                    args=(S_short_obs, Z_short_obs, E_pool, opt_phi, opt_sig_pi, opt_sig_S),
                    method='L-BFGS-B', bounds=CONFIG['bounds_step2'])
    opt_sig_gamma, opt_sig_Z = res2.x
    print(f"推計完了: sig_gamma={opt_sig_gamma:.4f}, sig_Z={opt_sig_Z:.4f}\n")
    
    # 5. 内生変数の最終抽出とCSV保存
    print("--- 5. 内生変数の抽出とCSV保存 ---")
    points = MerweScaledSigmaPoints(n=dim_x, alpha=0.1, beta=2., kappa=3-dim_x)
    ukf_final = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=1., fx=fx, hx=hx, points=points)
    ukf_final.x = np.array([CONFIG['fixed_mu'], 0.5])
    ukf_final.P *= 0.1
    ukf_final.Q = np.diag([opt_sig_pi**2, opt_sig_gamma**2])
    ukf_final.R = np.diag([opt_sig_S**2] + [opt_sig_Z**2] * CONFIG['C'])
    
    records = []
    T_short = CONFIG['T_short']
    start_idx = CONFIG['start_idx_short']
    
    for t_short in range(T_short):
        t_long = start_idx + t_short # 実時間インデックス (例: 600番目〜)
        
        z_t = np.zeros(dim_z)
        z_t[0] = S_short_obs[t_short]
        z_t[1:] = Z_short_obs[t_short]
        
        ukf_final.predict(phi=opt_phi)
        ukf_final.update(z_t, E_t=E_pool[t_short], phi=opt_phi)
        
        filt_pi, filt_gamma = ukf_final.x
        implied_z = hx(ukf_final.x, E_pool[t_short], opt_phi)
        
        record = {
            'Time_Index': t_long,
            'Obs_S': S_short_obs[t_short],
            'State_pi_t': filt_pi,
            'State_gamma_t': filt_gamma,
        }
        for c in range(CONFIG['C']):
            record[f'Experience_E_{c+1}'] = E_pool[t_short, c]
            record[f'Obs_Z_{c+1}'] = Z_short_obs[t_short, c]
            record[f'Implied_Z_{c+1}'] = implied_z[1+c]
            
        records.append(record)
        
    df_endogenous = pd.DataFrame(records).set_index('Time_Index')
    df_endogenous.to_csv(CONFIG['output_csv'])
    print(f"内生変数を '{CONFIG['output_csv']}' に保存しました。\n")
    
    # 6. 結果の可視化
    print("--- 6. 結果のプロットを表示します ---")
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle('2-Step Approach with Malmendier & Nagel Experience Weights', fontsize=16)

    # パネル1: シグナルと推計されたインフレ率 (後半20年)
    axes[0].plot(df_endogenous.index, df_endogenous['Obs_S'], label='Observed Signal $S_t$', color='gray', alpha=0.5)
    axes[0].plot(df_endogenous.index, df_endogenous['State_pi_t'], label='Filtered $\pi_t$ (Latent Inflation)', color='blue', linewidth=2)
    axes[0].axhline(CONFIG['fixed_mu'], color='red', linestyle='--', label='Steady State $\mu$ (2.0)')
    axes[0].set_title('Step 1/2 Output: Inflation Dynamics (Survey Period)')
    axes[0].set_ylabel('Rate (%)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # パネル2: インフレ経験へのウエイト gamma_t
    axes[1].plot(df_endogenous.index, df_endogenous['State_gamma_t'], label='Filtered $\gamma_t$ (Weight on Experience)', color='green', linewidth=2)
    axes[1].set_title('Step 2 Output: Time-varying Weight on Inflation Experience')
    axes[1].set_ylabel('Weight $\gamma_t$')
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # パネル3: 経験とインフレ予想 (高齢コーホート2の例)
    c_idx = 2
    axes[2].plot(df_endogenous.index, df_endogenous[f'Obs_Z_{c_idx}'], label=f'Observed Survey $Z_{{{c_idx},t}}$', color='orange', alpha=0.4)
    axes[2].plot(df_endogenous.index, df_endogenous[f'Implied_Z_{c_idx}'], label=f'Model Implied $Z_{{{c_idx},t}}$', color='darkorange', linewidth=2)
    axes[2].plot(df_endogenous.index, df_endogenous[f'Experience_E_{c_idx}'], label=f'Calculated MN Experience $E_{{{c_idx},t}}$', color='purple', linestyle=':', linewidth=2)
    axes[2].set_title(f'Step 2 Output: Survey Expectations vs Implied (Cohort {c_idx})')
    axes[2].set_xlabel('Time Index (Months)')
    axes[2].set_ylabel('Rate (%)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
