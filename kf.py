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
    
    # ★ 変更点: 年齢を「年（歳）」で直接指定できるようにしました
    'ages': [40, 60],       # 各コーホートの年齢 (例: 40歳, 60歳)
    
    # --- gamma (過去の経験への依存度) の異質性スイッチ ---
    # False: 全世代で同じ gamma_t を推計する (シンプルなモデル)
    # True : 世代ごとに異なる gamma_{c,t} を推計する (複雑だが現実に近いモデル)
    'heterogeneous_gamma': True, 
    
    # --- データ期間の設定 (実データの日付に合わせて変更してください) ---
    'start_date_long': '1954-04-01',  # 高齢層の過去の記憶を計算するため、十分に古い年月を指定
    'start_date_short': '2004-04-01', # アンケート（サーベイ）データが実際に存在する最初の月
    'end_date': '2024-03-01',         # データの終わりの月
    
    'calibrated_theta': 1.5,          # 記憶が薄れるスピード (1.0~3.0程度。数字が大きいほど最近の記憶を重視)
    
    'mock_csv': 'mock_data_final.csv',               # モックデータの保存先
    'output_csv': 'endogenous_variables_final.csv',  # 推計結果の保存先
    
    # オプティマイザ（最尤推計）の初期値と、探索する数値の範囲(下限, 上限)
    'init_step1': [0.5, 0.1, 0.2],               # 第一段階: [インフレの持続性, インフレショックの大きさ, 観測ノイズ]
    'bounds_step1': ((0.01, 0.99), (1e-4, 5.0), (1e-4, 5.0)),
    
    'init_step2': [0.05, 0.1],                   # 第二段階: [gammaが変動する大きさ, サーベイの観測ノイズ]
    'bounds_step2': ((1e-4, 1.0), (1e-4, 5.0))
}

# --- 内部計算用の設定 (自動計算) ---
# 推計する「見えない状態（State）」の数。
# 全世代共通なら [インフレ率, 共通のgamma] の2つ。世代別なら [インフレ率, gamma_1, gamma_2...] と増える。
DIM_X = 1 + CONFIG['C'] if CONFIG['heterogeneous_gamma'] else 2
# 「観測できるデータ（Observation）」の数。 [マクロのインフレシグナル, サーベイ_1, サーベイ_2...] の合計。
DIM_Z = 1 + CONFIG['C']

# 期間ごとのデータ数を計算 (pandasの機能を使って日付から正確なインデックスを割り出します)
global_dates = pd.date_range(start=CONFIG['start_date_long'], end=CONFIG['end_date'], freq='MS')
CONFIG['T_long'] = len(global_dates)
CONFIG['start_idx_short'] = global_dates.get_loc(CONFIG['start_date_short'])
CONFIG['T_short'] = CONFIG['T_long'] - CONFIG['start_idx_short']
dates_short = global_dates[CONFIG['start_idx_short']:]

# =====================================================================
# 1. モックデータ (シミュレーション用の擬似データ) の生成と読み込み
# =====================================================================
def generate_mock_data():
    """実際のデータがない場合に、テスト推計を行うためのダミーデータを生成する関数です"""
    np.random.seed(42)
    T_long = CONFIG['T_long']
    start_idx = CONFIG['start_idx_short']
    
    # 1. 真のマクロインフレ率の生成
    true_phi, true_sig_pi, true_sig_S = 0.8, 0.1, 0.2
    true_pi = np.zeros(T_long)
    true_pi[0] = CONFIG['fixed_mu']
    
    # 歴史的なショック（オイルショックやデフレ）を発生させる日付を指定
    idx_shock1_start = global_dates.get_loc('1973-10-01')
    idx_shock1_end   = global_dates.get_loc('1980-12-01')
    idx_shock2_start = global_dates.get_loc('1998-01-01')
    idx_shock2_end   = global_dates.get_loc('2012-12-01')
    
    for t in range(1, T_long):
        # 今月のインフレ = (目標値に引っ張られる力) + (先月のインフレの影響) + (予測不能なショック)
        true_pi[t] = (1 - true_phi) * CONFIG['fixed_mu'] + true_phi * true_pi[t-1] + np.random.normal(0, true_sig_pi)
        if idx_shock1_start <= t <= idx_shock1_end: true_pi[t] += 0.4
        elif idx_shock2_start <= t <= idx_shock2_end: true_pi[t] -= 0.4
            
    # S_data: 私たちが実際に観測できる「ノイズまみれのマクロ指標 (例: CPI)」
    S_data = true_pi + np.random.normal(0, true_sig_S, T_long)
    
    # 2. 真の gamma (経験への依存度) のパスを生成
    if CONFIG['heterogeneous_gamma']:
        true_gamma = np.zeros((T_long, CONFIG['C']))
        true_gamma[0, :] = 0.5
        for t in range(1, T_long):
            for c in range(CONFIG['C']):
                # 世代ごとに独立してフラフラと変動(ランダムウォーク)させる。0.1〜0.9の範囲に制限。
                true_gamma[t, c] = np.clip(true_gamma[t-1, c] + np.random.normal(0, 0.03), 0.1, 0.9)
    else:
        true_gamma_single = np.zeros(T_long)
        true_gamma_single[0] = 0.5
        for t in range(1, T_long):
            true_gamma_single[t] = np.clip(true_gamma_single[t-1] + np.random.normal(0, 0.03), 0.1, 0.9)
        true_gamma = np.tile(true_gamma_single, (CONFIG['C'], 1)).T
        
    Z_data = np.full((T_long, CONFIG['C']), np.nan)
    
    # 3. 真のサーベイデータの生成
    for c, age_years in enumerate(CONFIG['ages']):
        # ★年齢(年)を月数に変換
        age_months = age_years * 12
        # L_c: この世代が「20歳以降」に経験した月数
        L_c = age_months - 240 
        
        for t in range(start_idx, T_long):
            # データが足りない場合は、存在する分だけで計算を打ち切る(安全装置)
            available_L = min(L_c, t)
            if available_L <= 0: continue
            
            # 最近の記憶ほど重みが大きくなるウエイトを計算
            k_arr = np.arange(1, available_L + 1)
            weights = (age_months - k_arr) ** CONFIG['calibrated_theta']
            weights = weights / np.sum(weights) # 合計が1になるように調整(規格化)
            
            # 過去のインフレ率をウエイトで加重平均して「経験(E_ct)」を作る
            window = true_pi[t - available_L : t]
            true_E_ct = np.sum(weights * window[::-1])
            
            # 完璧に合理的な予測 (12ヶ月先)
            true_rational_exp = (1 - true_phi**12) * CONFIG['fixed_mu'] + (true_phi**12) * true_pi[t]
            
            # 観測方程式: (経験への依存) + (合理的期待への依存)
            true_Z = true_gamma[t, c] * true_E_ct + (1 - true_gamma[t, c]) * true_rational_exp
            # アンケート特有のばらつき(ノイズ)を加える
            Z_data[t, c] = true_Z + np.random.normal(0, 0.15)
            
    # データを束ねてCSVに保存
    df_dict = {'S_t': S_data}
    for c in range(CONFIG['C']):
        df_dict[f'Z_{c+1}'] = Z_data[:, c]
        df_dict[f'True_gamma_{c+1}'] = true_gamma[:, c]
        
    df_mock = pd.DataFrame(df_dict, index=global_dates)
    df_mock.index.name = 'Date'
    df_mock.to_csv(CONFIG['mock_csv'])
    
    df_short = df_mock.iloc[CONFIG['start_idx_short']:].copy()
    return S_data, df_short['S_t'].values, df_short[[f'Z_{c+1}' for c in range(CONFIG['C'])]].values, df_short

# =====================================================================
# 2. Step 1: 線形カルマンフィルター (マクロインフレ率の推論)
# =====================================================================
def setup_linear_kf(phi, sig_pi, sig_S):
    """
    filterpy の KalmanFilter を設定します。
    ここでは、見えない「真のインフレ率」を、ノイズまみれの「CPIなどのデータ」からあぶり出します。
    """
    kf = KalmanFilter(dim_x=1, dim_z=1)
    
    kf.x = np.array([[CONFIG['fixed_mu']]]) # x: 探したい「真のインフレ率」の初期値
    kf.P = np.array([[1.0]])                # P: 推計に対する「自信のなさ（誤差の大きさ）」。最初は自信がないので1.0と大きめにする。
    
    # モデルの骨組み（動学）を定義する行列群
    kf.F = np.array([[phi]]) # F: 状態遷移 (先月から今月へ、どれくらい持続するか)
    kf.B = np.array([[1.0]]) # B: コントロール入力への反応 (目標値への回帰力)
    kf.H = np.array([[1.0]]) # H: 観測行列 (真のインフレ率が、実際のデータにどう現れるか。ここでは1対1)
    
    # ノイズ（ばらつき）の大きさを定義する行列群
    kf.Q = np.array([[sig_pi**2]]) # Q: プロセスノイズ (経済に起きる本当のショックの大きさ)
    kf.R = np.array([[sig_S**2]])  # R: 観測ノイズ (CPIという統計指標自体が持つ測定誤差の大きさ)
    
    return kf

def nll_step1(params, S_long_obs):
    """データに最も当てはまるパラメータ(phi, sig_pi, sig_S)を探すための「マイナス対数尤度」を計算します"""
    kf = setup_linear_kf(*params)
    u = np.array([[(1 - params[0]) * CONFIG['fixed_mu']]]) # 目標値への引力
    ll_total = 0.0
    for z in S_long_obs:
        kf.predict(u=u)                     # 1. 過去の情報から「今月はどうなるか」を予測する
        kf.update(np.array([[z]]))          # 2. 「実際のデータ(z)」を見て、予測を修正する
        ll_total += kf.log_likelihood       # 3. その予測がどれくらい当たっていたか(尤度)を足し合わせる
    return -ll_total

def extract_nowcast_path(S_long_obs, opt_params):
    """最適化されたパラメータを使って、過去から現在までの「真のインフレ率の推論パス」を抽出します"""
    kf = setup_linear_kf(*opt_params)
    u = np.array([[(1 - opt_params[0]) * CONFIG['fixed_mu']]])
    pi_hats = []
    for z in S_long_obs:
        kf.predict(u=u)
        kf.update(np.array([[z]]))
        pi_hats.append(kf.x[0, 0]) # 修正された推計結果(x)を保存
    return np.array(pi_hats)

# =====================================================================
# 3. インフレ経験 E_{c,t} の構築
# =====================================================================
def compute_exact_mn_experience(pi_hat_long, theta, ages, start_idx_short):
    """各世代が過去に経験してきたインフレの「重み付け平均」を計算します"""
    E_pool = np.zeros((CONFIG['T_short'], len(ages)))
    
    for c, age_years in enumerate(ages):
        age_months = age_years * 12      # ★年齢を月数に変換
        L_c = age_months - 240           # 20歳以降の経験月数
        
        for t_short in range(CONFIG['T_short']):
            t_long = start_idx_short + t_short
            available_L = min(L_c, t_long) # データが足りない場合の安全装置
            
            if available_L <= 0:
                E_pool[t_short, c] = pi_hat_long[t_long]
                continue
                
            weights = (age_months - np.arange(1, available_L + 1)) ** theta
            window = pi_hat_long[t_long - available_L : t_long]
            E_pool[t_short, c] = np.sum((weights / np.sum(weights)) * window[::-1])
            
    return E_pool

# =====================================================================
# 4. Step 2: UKF (Unscented Kalman Filter) - 経験ウエイトの推計
# =====================================================================
def fx(x, dt, phi):
    """
    UKFの状態遷移関数。見えない状態(x)が時間とともにどう変化するかを定義します。
    x[0] : インフレ率
    x[1:] : 潜在変数としての gamma (無限大〜マイナス無限大の範囲を動く)
    """
    x_next = np.empty_like(x)
    # インフレ率はAR(1)プロセス(過去の値と目標値の間を動く)
    x_next[0] = (1 - phi) * CONFIG['fixed_mu'] + phi * x[0]
    # gamma は「ランダムウォーク(先月の値から予測不能な方向へフラフラ動く)」と仮定
    x_next[1:] = x[1:] 
    return x_next

def hx(x, E_t, phi):
    """
    UKFの観測関数。見えない状態(x)から、手元のデータ(サーベイ予想など)がどう作られるかを定義します。
    ここで「変数同士の掛け算」が発生するため、線形KFではなく非線形なUKFが必要になります。
    """
    pi_t = x[0]
    z = np.zeros(DIM_Z)
    
    # インフレ率から計算される「12ヶ月先の合理的な予測」
    rational_exp = (1 - phi**12) * CONFIG['fixed_mu'] + (phi**12) * pi_t
    z[0] = pi_t
    
    for c in range(CONFIG['C']):
        # 異質性スイッチに応じて、使うべき gamma を選ぶ
        gamma_tilde = x[1 + c] if CONFIG['heterogeneous_gamma'] else x[1]
        
        # ★ シグモイド変換の魔法:
        # カルマンフィルターは変数が無限大まで動くと勘違いして推計を暴走させます。
        # そこで、推計された潜在変数(gamma_tilde)を、必ず 0.0〜1.0 の間に収まるように変換します。
        gamma_c = 1.0 / (1.0 + np.exp(-gamma_tilde))
        
        # 実際のアンケート結果(z)は、経験(E_t)と合理的期待のブレンドである
        z[1 + c] = gamma_c * E_t[c] + (1 - gamma_c) * rational_exp
    return z

def setup_ukf(opt_phi, opt_sig_pi, opt_sig_S, sig_gamma, sig_Z):
    """
    UKFの初期設定を行います。
    UKFは、状態の「代表点(シグマポイント)」をいくつかバラ撒いて、それらが観測関数(hx)を通った後に
    どんな分布になるかを計算することで、非線形な関係を極めて正確に近似します。
    """
    # MerweScaledSigmaPoints: 代表点(シグマポイント)をどう配置するかを決めるアルゴリズム
    points = MerweScaledSigmaPoints(n=DIM_X, alpha=0.1, beta=2., kappa=3-DIM_X)
    ukf = UnscentedKalmanFilter(dim_x=DIM_X, dim_z=DIM_Z, dt=1., fx=fx, hx=hx, points=points)
    
    # 初期値の設定。潜在変数(gamma_tilde)の初期値は0.0 (シグモイド変換するとピッタリ0.5になります)
    ukf.x = np.array([CONFIG['fixed_mu']] + [0.0] * (DIM_X - 1))
    
    # P: 推計に対する自信のなさ。
    # pi_t は標準偏差0.5(分散0.25)、gamma_tilde は標準偏差0.5(分散0.25) くらいからスタートさせる。
    ukf.P = np.diag([0.5**2] + [0.5**2] * (DIM_X - 1))
    
    # Q: 変数そのものが毎月どれくらい激しく変動するか (プロセスノイズ)
    ukf.Q = np.diag([opt_sig_pi**2] + [sig_gamma**2] * (DIM_X - 1))
    # R: 観測データにどれくらいノイズが混じっているか (観測ノイズ)
    ukf.R = np.diag([opt_sig_S**2] + [sig_Z**2] * CONFIG['C'])
    return ukf

def nll_step2(params, S_short_obs, Z_short_obs, E_pool, opt_phi, opt_sig_pi, opt_sig_S):
    """第二段階（gammaの動学推計）の最適化のための関数です"""
    sig_gamma, sig_Z = params
    ukf = setup_ukf(opt_phi, opt_sig_pi, opt_sig_S, sig_gamma, sig_Z)
    ll_total = 0.0
    
    # 毎月ごとに予測(Predict)と修正(Update)を繰り返して尤度を計算します
    for t in range(len(S_short_obs)):
        z_t = np.zeros(DIM_Z)
        z_t[0], z_t[1:] = S_short_obs[t], Z_short_obs[t]
        
        ukf.predict(phi=opt_phi)
        ukf.update(z_t, E_t=E_pool[t], phi=opt_phi)
        ll_total += ukf.log_likelihood
    return -ll_total

# =====================================================================
# 5. メイン実行・評価モジュール・可視化
# =====================================================================
if __name__ == "__main__":
    # --- データ準備 ---
    S_long_obs, S_short_obs, Z_short_obs, df_short = generate_mock_data()
    
    # --- Step 1 実行 ---
    print("--- 1. Step 1: マクロ推論パスの推計 ---")
    res1 = minimize(nll_step1, CONFIG['init_step1'], args=(S_long_obs,), method='L-BFGS-B', bounds=CONFIG['bounds_step1'])
    opt_phi, opt_sig_pi, opt_sig_S = res1.x
    pi_hat_long = extract_nowcast_path(S_long_obs, res1.x)
    E_pool = compute_exact_mn_experience(pi_hat_long, CONFIG['calibrated_theta'], CONFIG['ages'], CONFIG['start_idx_short'])
    
    # --- Step 2 実行 ---
    print(f"--- 2. Step 2: UKF 推計 (世代間異質性: {CONFIG['heterogeneous_gamma']}) ---")
    res2 = minimize(nll_step2, CONFIG['init_step2'], args=(S_short_obs, Z_short_obs, E_pool, opt_phi, opt_sig_pi, opt_sig_S),
                    method='L-BFGS-B', bounds=CONFIG['bounds_step2'])
    opt_sig_gamma, opt_sig_Z = res2.x
    
    # -----------------------------------------------------------------
    # ★ モデル評価 (論文の表に載せるためのスコア計算)
    # -----------------------------------------------------------------
    ukf_final = setup_ukf(opt_phi, opt_sig_pi, opt_sig_S, opt_sig_gamma, opt_sig_Z)
    records = []
    
    total_log_likelihood = 0.0
    sum_sq_error_Z = 0.0
    count_Z = 0
    
    # 推計結果をもう一度最初からなぞって、各種指標を記録します
    for t_short in range(CONFIG['T_short']):
        z_t = np.zeros(DIM_Z)
        z_t[0], z_t[1:] = S_short_obs[t_short], Z_short_obs[t_short]
        
        # [A] 事前予測 (Predict): 「今月のデータを見る前」の予測を立てる
        ukf_final.predict(phi=opt_phi)
        
        # [B] RMSEの計算: 実際のアンケート結果と、予測値のズレ(誤差)を計算する。
        # ※「答え合わせ」をする前の予測誤差を使うのが計量経済学の絶対ルールです。
        z_pred_prior = hx(ukf_final.x, E_pool[t_short], opt_phi)
        for c in range(CONFIG['C']):
            error = z_t[1 + c] - z_pred_prior[1 + c]
            sum_sq_error_Z += error**2
            count_Z += 1
            
        # [C] 修正 (Update): 今月のデータを見て、推計値を微修正する
        ukf_final.update(z_t, E_t=E_pool[t_short], phi=opt_phi)
        total_log_likelihood += ukf_final.log_likelihood
        
        # --- 記録用データの保存 (CSV用) ---
        implied_z_post = hx(ukf_final.x, E_pool[t_short], opt_phi)
        record = {
            'Date': dates_short[t_short],
            'Obs_S': S_short_obs[t_short],
            'State_pi_t': ukf_final.x[0],
            'Rational_Exp': (1 - opt_phi**12) * CONFIG['fixed_mu'] + (opt_phi**12) * ukf_final.x[0]
        }
        
        for c in range(CONFIG['C']):
            # 潜在変数を 0.0〜1.0 の実数に変換して保存
            gamma_tilde = ukf_final.x[1 + c] if CONFIG['heterogeneous_gamma'] else ukf_final.x[1]
            gamma_val = 1.0 / (1.0 + np.exp(-gamma_tilde)) 
            
            record[f'State_gamma_{c+1}'] = gamma_val
            record[f'Experience_E_{c+1}'] = E_pool[t_short, c]
            record[f'Obs_Z_{c+1}'] = Z_short_obs[t_short, c]
            record[f'Implied_Z_{c+1}'] = implied_z_post[1 + c]
            record[f'True_gamma_{c+1}'] = df_short[f'True_gamma_{c+1}'].iloc[t_short]
            
        records.append(record)
        
    df_endogenous = pd.DataFrame(records).set_index('Date')
    df_endogenous.to_csv(CONFIG['output_csv'])
    
    # 評価指標の最終計算
    k_params = len(CONFIG['init_step2']) # モデルの複雑さ(パラメータ数)
    aic = -2 * total_log_likelihood + 2 * k_params
    bic = -2 * total_log_likelihood + k_params * np.log(CONFIG['T_short'])
    rmse_Z = np.sqrt(sum_sq_error_Z / count_Z)
    
    print("\n" + "="*50)
    print(" 📊 MODEL EVALUATION METRICS (In-Sample)")
    print("="*50)
    print(f" Heterogeneous Gamma : {CONFIG['heterogeneous_gamma']}")
    print(f" Total Log-Likelihood: {total_log_likelihood:.2f}")
    print(f" AIC (Akaike Info)   : {aic:.2f}  <-- 同質 vs 異質の比較用")
    print(f" BIC (Bayesian Info) : {bic:.2f}  <-- 同質 vs 異質の比較用")
    print(f" Survey 1-step RMSE  : {rmse_Z:.4f}  <-- 使用するシグナル(CPI等)の比較用")
    print("="*50 + "\n")
    
    # -----------------------------------------------------------------
    # ★ グラフの描画
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    title_suffix = "Heterogeneous" if CONFIG['heterogeneous_gamma'] else "Homogeneous"
    fig.suptitle(f'2-Step Approach: Endogenous Variables ({title_suffix} $\gamma$)', fontsize=16)

    # パネル1: マクロインフレ動学
    axes[0].plot(df_endogenous.index, df_endogenous['Obs_S'], label='Observed Signal $S_t$', color='gray', alpha=0.5)
    axes[0].plot(df_endogenous.index, df_endogenous['State_pi_t'], label='Filtered $\pi_t$', color='blue', linewidth=2)
    axes[0].axhline(CONFIG['fixed_mu'], color='red', linestyle='--', label='Anchor $\mu$')
    axes[0].set_title('Inflation Dynamics')
    axes[0].set_ylabel('Rate (%)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # パネル2: ウエイト(gamma)の動学
    if CONFIG['heterogeneous_gamma']:
        for c in range(CONFIG['C']):
            axes[1].plot(df_endogenous.index, df_endogenous[f'State_gamma_{c+1}'], label=f'Filtered $\gamma_{{{c+1},t}}$ ({CONFIG["ages"][c]} years)', linewidth=2)
            axes[1].plot(df_endogenous.index, df_endogenous[f'True_gamma_{c+1}'], color='black', linestyle=':', alpha=0.5, label=f'True $\gamma_{{{c+1},t}}$')
        axes[1].set_title('Heterogeneous Weights on Inflation Experience')
    else:
        axes[1].plot(df_endogenous.index, df_endogenous['State_gamma_1'], label='Filtered $\gamma_t$ (Common)', color='green', linewidth=2)
        axes[1].plot(df_endogenous.index, df_endogenous['True_gamma_1'], color='black', linestyle=':', alpha=0.5, label='True $\gamma_t$')
        axes[1].set_title('Homogeneous Weight on Inflation Experience')
        
    axes[1].set_ylabel('Weight $\gamma$')
    axes[1].set_ylim(-0.1, 1.1) 
    axes[1].legend(loc='best', ncol=2, fontsize='small')
    axes[1].grid(True, alpha=0.3)

    # パネル3: サーベイ vs インプライド vs 経験
    c_idx = CONFIG['C'] # 最も高齢のコーホートのグラフを表示
    axes[2].plot(df_endogenous.index, df_endogenous[f'Obs_Z_{c_idx}'], label=f'Observed Survey $Z_{{{c_idx},t}}$', color='orange', alpha=0.4)
    axes[2].plot(df_endogenous.index, df_endogenous[f'Implied_Z_{c_idx}'], label=f'Model Implied $Z_{{{c_idx},t}}$', color='darkorange', linewidth=2)
    axes[2].plot(df_endogenous.index, df_endogenous[f'Experience_E_{c_idx}'], label=f'MN Experience $E_{{{c_idx},t}}$', color='purple', linestyle='-', linewidth=2)
    axes[2].plot(df_endogenous.index, df_endogenous['Rational_Exp'], label='Rational Expectation', color='teal', linestyle='--', linewidth=1.5)
    axes[2].set_title(f'Survey Expectations vs Implied (Cohort {c_idx}: {CONFIG["ages"][c_idx-1]} years old)')
    axes[2].set_ylabel('Rate (%)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()
