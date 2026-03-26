import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints

# =====================================================================
# 0. 基本設定 (Configuration)
# =====================================================================
CONFIG = {
    'C': 2,                 
    'fixed_mu': 2.0,        
    'ages': [40, 60],       
    'heterogeneous_gamma': True, 
    
    'start_date_long': '1954-04-01',  
    'start_date_short': '2004-04-01', 
    'end_date': '2024-03-01',         
    
    'calibrated_theta': 1.5,          
    
    'mock_csv': 'mock_data_final.csv',
    'output_csv': 'endogenous_variables_final.csv',
    
    'init_step1': [0.5, 0.1, 0.2],               
    'bounds_step1': ((0.01, 0.99), (1e-4, 5.0), (1e-4, 5.0)),
    
    # ★ Step 2 推計対象:
    # [sig_gamma, sig_lambda, sig_Z, sig_lambda_obs, b_1, b_2, alpha]
    'init_step2': [0.05, 0.05, 0.1, 0.05, 0.2, 0.2, 2.0],                   
    'bounds_step2': ((1e-4, 1.0), (1e-4, 1.0), (1e-4, 5.0), (1e-4, 1.0), 
                     (1e-4, 0.99), (1e-4, 0.99), (1e-4, 20.0))
}

num_gamma_states = CONFIG['C'] if CONFIG['heterogeneous_gamma'] else 1
DIM_X = 1 + num_gamma_states + CONFIG['C']
DIM_Z = 1 + CONFIG['C'] + CONFIG['C']

global_dates = pd.date_range(start=CONFIG['start_date_long'], end=CONFIG['end_date'], freq='MS')
CONFIG['T_long'] = len(global_dates)
CONFIG['start_idx_short'] = global_dates.get_loc(CONFIG['start_date_short'])
CONFIG['T_short'] = CONFIG['T_long'] - CONFIG['start_idx_short']
dates_short = global_dates[CONFIG['start_idx_short']:]

# =====================================================================
# 1. モックデータの生成 (ロジスティック関数の上方バイアスを導入)
# =====================================================================
def generate_mock_data():
    np.random.seed(42)
    T_long = CONFIG['T_long']
    start_idx = CONFIG['start_idx_short']
    
    true_phi, true_sig_pi, true_sig_S = 0.8, 0.1, 0.2
    true_pi = np.zeros(T_long)
    true_pi[0] = CONFIG['fixed_mu']
    
    idx_shock1_start = global_dates.get_loc('1973-10-01')
    idx_shock1_end   = global_dates.get_loc('1980-12-01')
    idx_shock2_start = global_dates.get_loc('1998-01-01')
    idx_shock2_end   = global_dates.get_loc('2012-12-01')
    
    for t in range(1, T_long):
        true_pi[t] = (1 - true_phi) * CONFIG['fixed_mu'] + true_phi * true_pi[t-1] + np.random.normal(0, true_sig_pi)
        if idx_shock1_start <= t <= idx_shock1_end: true_pi[t] += 0.4
        elif idx_shock2_start <= t <= idx_shock2_end: true_pi[t] -= 0.4
            
    S_data = true_pi + np.random.normal(0, true_sig_S, T_long)
    
    # インフレ変動の絶対値 (abs_delta_pi)
    abs_delta_pi_long = np.zeros(T_long)
    abs_delta_pi_long[1:] = np.abs(S_data[1:] - S_data[:-1])
    
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
        
    # 時変 lambda_ct とプロキシ観測の生成
    true_lambda = np.zeros((T_long, CONFIG['C']))
    true_lambda[0, :] = [0.3, 0.7]
    lambda_obs_data = np.zeros((T_long, CONFIG['C']))
    
    # 真の構造パラメータ (バイアス確率 b_c と 感応度 alpha)
    true_b = [0.2, 0.2]
    true_alpha = 2.0
    
    for t in range(1, T_long):
        for c in range(CONFIG['C']):
            true_lambda[t, c] = np.clip(true_lambda[t-1, c] + np.random.normal(0, 0.02), 0.1, 0.9)
            
    for t in range(T_long):
        for c in range(CONFIG['C']):
            # ★ ロジスティック関数による「偶然留まる確率」の計算
            stay_prob = true_b[c] * (2.0 / (1.0 + np.exp(true_alpha * abs_delta_pi_long[t])))
            # 上方バイアスを付与して観測データを生成
            expected_obs = true_lambda[t, c] + (1 - true_lambda[t, c]) * stay_prob
            lambda_obs_data[t, c] = expected_obs + np.random.normal(0, 0.05)
            
    Z_data = np.full((T_long, CONFIG['C']), np.nan)
    
    for c, age_years in enumerate(CONFIG['ages']):
        age_months = age_years * 12
        L_c = age_months - 240 
        Z_data[start_idx-1, c] = 2.0
        
        for t in range(start_idx, T_long):
            available_L = min(L_c, t)
            if available_L <= 0: continue
            
            k_arr = np.arange(1, available_L + 1)
            weights = (age_months - k_arr) ** CONFIG['calibrated_theta']
            weights = weights / np.sum(weights)
            
            window = true_pi[t - available_L : t]
            true_E_ct = np.sum(weights * window[::-1])
            
            true_rational_exp = (1 - true_phi**12) * CONFIG['fixed_mu'] + (true_phi**12) * true_pi[t]
            Z_star = true_gamma[t, c] * true_E_ct + (1 - true_gamma[t, c]) * true_rational_exp
            true_Z = (1 - true_lambda[t, c]) * Z_star + true_lambda[t, c] * Z_data[t-1, c]
            Z_data[t, c] = true_Z + np.random.normal(0, 0.15)
            
    df_dict = {'S_t': S_data}
    for c in range(CONFIG['C']):
        df_dict[f'Z_{c+1}'] = Z_data[:, c]
        df_dict[f'True_gamma_{c+1}'] = true_gamma[:, c]
        df_dict[f'True_lambda_{c+1}'] = true_lambda[:, c]
        df_dict[f'Lambda_obs_{c+1}'] = lambda_obs_data[:, c]
        
    df_mock = pd.DataFrame(df_dict, index=global_dates)
    df_mock.index.name = 'Date'
    df_mock.to_csv(CONFIG['mock_csv'])
    
    df_short = df_mock.iloc[CONFIG['start_idx_short']:].copy()
    abs_delta_pi_short = abs_delta_pi_long[CONFIG['start_idx_short']:]
    
    Z_lag_short = np.zeros((CONFIG['T_short'], CONFIG['C']))
    Z_lag_short[0, :] = df_mock[[f'Z_{c+1}' for c in range(CONFIG['C'])]].iloc[CONFIG['start_idx_short']-1].values
    if CONFIG['T_short'] > 1:
        Z_lag_short[1:, :] = df_short[[f'Z_{c+1}' for c in range(CONFIG['C'])]].values[:-1, :]
        
    return S_data, df_short['S_t'].values, df_short[[f'Z_{c+1}' for c in range(CONFIG['C'])]].values, \
           df_short[[f'Lambda_obs_{c+1}' for c in range(CONFIG['C'])]].values, Z_lag_short, abs_delta_pi_short, df_short

# =====================================================================
# 2. Step 1: 線形カルマンフィルター
# =====================================================================
def setup_linear_kf(phi, sig_pi, sig_S):
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x, kf.P, kf.F, kf.B, kf.H = np.array([[CONFIG['fixed_mu']]]), np.array([[1.0]]), np.array([[phi]]), np.array([[1.0]]), np.array([[1.0]])
    kf.Q, kf.R = np.array([[sig_pi**2]]), np.array([[sig_S**2]])
    return kf

def nll_step1(params, S_long_obs):
    kf = setup_linear_kf(*params)
    u = np.array([[(1 - params[0]) * CONFIG['fixed_mu']]])
    ll_total = sum(kf.predict(u=u) or kf.update(np.array([[z]])) or kf.log_likelihood for z in S_long_obs)
    return -ll_total

def extract_nowcast_path(S_long_obs, opt_params):
    kf = setup_linear_kf(*opt_params)
    u = np.array([[(1 - opt_params[0]) * CONFIG['fixed_mu']]])
    return np.array([kf.predict(u=u) or kf.update(np.array([[z]])) or kf.x[0, 0] for z in S_long_obs])

def compute_exact_mn_experience(pi_hat_long, theta, ages, start_idx_short):
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
# 3. Step 2: UKF (ロジスティック関数による観測モデル)
# =====================================================================
def fx(x, dt, phi):
    x_next = np.empty_like(x)
    x_next[0] = (1 - phi) * CONFIG['fixed_mu'] + phi * x[0]
    x_next[1:] = x[1:] 
    return x_next

def hx(x, E_t, Z_lag, phi, abs_delta_pi, b_params, alpha):
    pi_t = x[0]
    z = np.zeros(DIM_Z)
    rational_exp = (1 - phi**12) * CONFIG['fixed_mu'] + (phi**12) * pi_t
    z[0] = pi_t
    
    for c in range(CONFIG['C']):
        gamma_tilde = x[1 + c] if CONFIG['heterogeneous_gamma'] else x[1]
        lambda_tilde = x[1 + num_gamma_states + c]
        
        gamma_c = 1.0 / (1.0 + np.exp(-gamma_tilde))
        lambda_c = 1.0 / (1.0 + np.exp(-lambda_tilde))
        
        Z_star = gamma_c * E_t[c] + (1 - gamma_c) * rational_exp
        z[1 + c] = (1 - lambda_c) * Z_star + lambda_c * Z_lag[c]
        
        # ★ ロジスティック関数による「偶然同じ選択肢に留まる確率」
        stay_prob = b_params[c] * (2.0 / (1.0 + np.exp(alpha * abs_delta_pi)))
        expected_obs_lambda = lambda_c + (1 - lambda_c) * stay_prob
        
        z[1 + CONFIG['C'] + c] = expected_obs_lambda
        
    return z

def setup_ukf(opt_phi, opt_sig_pi, opt_sig_S, sig_gamma, sig_lambda, sig_Z, sig_lambda_obs):
    points = MerweScaledSigmaPoints(n=DIM_X, alpha=0.1, beta=2., kappa=3-DIM_X)
    ukf = UnscentedKalmanFilter(dim_x=DIM_X, dim_z=DIM_Z, dt=1., fx=fx, hx=hx, points=points)
    
    ukf.x = np.array([CONFIG['fixed_mu']] + [0.0] * (DIM_X - 1))
    ukf.P = np.diag([0.5**2] + [0.5**2] * (DIM_X - 1))
    ukf.Q = np.diag([opt_sig_pi**2] + [sig_gamma**2] * num_gamma_states + [sig_lambda**2] * CONFIG['C'])
    ukf.R = np.diag([opt_sig_S**2] + [sig_Z**2] * CONFIG['C'] + [sig_lambda_obs**2] * CONFIG['C'])
    return ukf

def nll_step2(params, S_short_obs, Z_short_obs, lambda_obs_short, Z_lag_short, E_pool, opt_phi, opt_sig_pi, opt_sig_S, abs_delta_pi_short):
    sig_gamma, sig_lambda, sig_Z, sig_lambda_obs = params[0:4]
    b_params = params[4:4+CONFIG['C']]
    alpha = params[-1]
    
    ukf = setup_ukf(opt_phi, opt_sig_pi, opt_sig_S, sig_gamma, sig_lambda, sig_Z, sig_lambda_obs)
    ll_total = 0.0
    
    for t in range(len(S_short_obs)):
        z_t = np.zeros(DIM_Z)
        z_t[0] = S_short_obs[t]
        z_t[1:1+CONFIG['C']] = Z_short_obs[t]
        z_t[1+CONFIG['C']:] = lambda_obs_short[t] 
        
        ukf.predict(phi=opt_phi)
        ukf.update(z_t, E_t=E_pool[t], Z_lag=Z_lag_short[t], phi=opt_phi, 
                   abs_delta_pi=abs_delta_pi_short[t], b_params=b_params, alpha=alpha)
        ll_total += ukf.log_likelihood
    return -ll_total

# =====================================================================
# 4. メイン実行・評価モジュール・可視化
# =====================================================================
if __name__ == "__main__":
    S_long_obs, S_short_obs, Z_short_obs, lambda_obs_short, Z_lag_short, abs_delta_pi_short, df_short = generate_mock_data()
    
    print("--- 1. Step 1: マクロ推論パスの推計 ---")
    res1 = minimize(nll_step1, CONFIG['init_step1'], args=(S_long_obs,), method='L-BFGS-B', bounds=CONFIG['bounds_step1'])
    opt_phi, opt_sig_pi, opt_sig_S = res1.x
    pi_hat_long = extract_nowcast_path(S_long_obs, res1.x)
    E_pool = compute_exact_mn_experience(pi_hat_long, CONFIG['calibrated_theta'], CONFIG['ages'], CONFIG['start_idx_short'])
    
    print(f"--- 2. Step 2: UKF 推計 (時変 lambda とロジスティックバイアス) ---")
    res2 = minimize(nll_step2, CONFIG['init_step2'], 
                    args=(S_short_obs, Z_short_obs, lambda_obs_short, Z_lag_short, E_pool, opt_phi, opt_sig_pi, opt_sig_S, abs_delta_pi_short),
                    method='L-BFGS-B', bounds=CONFIG['bounds_step2'])
    
    opt_sig_gamma, opt_sig_lambda, opt_sig_Z, opt_sig_lambda_obs = res2.x[0:4]
    opt_b_params = res2.x[4:4+CONFIG['C']]
    opt_alpha = res2.x[-1]
    
    ukf_final = setup_ukf(opt_phi, opt_sig_pi, opt_sig_S, opt_sig_gamma, opt_sig_lambda, opt_sig_Z, opt_sig_lambda_obs)
    records = []
    
    # 5年先予想(60ヶ月)のラグ変数を初期化
    Z_60_lag = np.full(CONFIG['C'], CONFIG['fixed_mu'])
    
    for t_short in range(CONFIG['T_short']):
        z_t = np.zeros(DIM_Z)
        z_t[0] = S_short_obs[t_short]
        z_t[1:1+CONFIG['C']] = Z_short_obs[t_short]
        z_t[1+CONFIG['C']:] = lambda_obs_short[t_short]
        
        ukf_final.predict(phi=opt_phi)
        ukf_final.update(z_t, E_t=E_pool[t_short], Z_lag=Z_lag_short[t_short], phi=opt_phi, 
                         abs_delta_pi=abs_delta_pi_short[t_short], b_params=opt_b_params, alpha=opt_alpha)
        
        implied_z_post = hx(ukf_final.x, E_pool[t_short], Z_lag_short[t_short], opt_phi, abs_delta_pi_short[t_short], opt_b_params, opt_alpha)
        
        # 内生変数として 5年先(60ヶ月先) インフレ予想を計算
        pi_t_post = ukf_final.x[0]
        rational_exp_60 = (1 - opt_phi**60) * CONFIG['fixed_mu'] + (opt_phi**60) * pi_t_post
        
        record = {
            'Date': dates_short[t_short],
            'Obs_S': S_short_obs[t_short],
            'State_pi_t': pi_t_post,
            'Rational_Exp': (1 - opt_phi**12) * CONFIG['fixed_mu'] + (opt_phi**12) * pi_t_post
        }
        
        for c in range(CONFIG['C']):
            gamma_tilde = ukf_final.x[1 + c] if CONFIG['heterogeneous_gamma'] else ukf_final.x[1]
            lambda_tilde = ukf_final.x[1 + num_gamma_states + c]
            
            gamma_val = 1.0 / (1.0 + np.exp(-gamma_tilde)) 
            lambda_val = 1.0 / (1.0 + np.exp(-lambda_tilde)) 
            
            # ★ 5年先インフレ予想の形成ロジック
            Z_star_60 = gamma_val * E_pool[t_short, c] + (1 - gamma_val) * rational_exp_60
            Z_60_current = (1 - lambda_val) * Z_star_60 + lambda_val * Z_60_lag[c]
            record[f'Exp_5Y_{c+1}'] = Z_60_current
            Z_60_lag[c] = Z_60_current # 次期用にラグを更新
            
            record[f'State_gamma_{c+1}'] = gamma_val
            record[f'State_lambda_{c+1}'] = lambda_val
            record[f'Experience_E_{c+1}'] = E_pool[t_short, c]
            record[f'Obs_Z_{c+1}'] = Z_short_obs[t_short, c]
            record[f'Implied_Z_{c+1}'] = implied_z_post[1 + c]
            
        records.append(record)
        
    df_endogenous = pd.DataFrame(records).set_index('Date')
    df_endogenous.to_csv(CONFIG['output_csv'])
    
    print("\n" + "="*50)
    print(" 📊 MODEL EVALUATION METRICS (In-Sample)")
    print("="*50)
    print(f" Estimated alpha (Sensitivity): {opt_alpha:.4f}")
    for c in range(CONFIG['C']):
        print(f" Cohort {c+1} base stay prob b_c: {opt_b_params[c]:.4f}")
    print("="*50 + "\n")
    
    # -----------------------------------------------------------------
    # ★ グラフの描画
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(5, 1, figsize=(12, 18), sharex=True)
    fig.suptitle('Extended UKF: Time-Varying $\gamma_{c,t}$, $\lambda_{c,t}$ & 5-Year Expectations', fontsize=16)

    axes[0].plot(df_endogenous.index, df_endogenous['Obs_S'], label='Observed Signal $S_t$', color='gray', alpha=0.5)
    axes[0].plot(df_endogenous.index, df_endogenous['State_pi_t'], label='Filtered $\pi_t$', color='blue', linewidth=2)
    axes[0].axhline(CONFIG['fixed_mu'], color='red', linestyle='--', label='Inflation Experience $\mu$')
    axes[0].set_title('Inflation Dynamics')
    axes[0].set_ylabel('Rate (%)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    if CONFIG['heterogeneous_gamma']:
        for c in range(CONFIG['C']):
            axes[1].plot(df_endogenous.index, df_endogenous[f'State_gamma_{c+1}'], label=f'Filtered $\gamma_{{{c+1},t}}$', linewidth=2)
    axes[1].set_title('Time-Varying Weight on Inflation Experience ($\gamma_{c,t}$)')
    axes[1].set_ylabel('Weight $\gamma$')
    axes[1].set_ylim(-0.1, 1.1) 
    axes[1].legend(loc='best', ncol=2, fontsize='small')
    axes[1].grid(True, alpha=0.3)

    for c in range(CONFIG['C']):
        axes[2].plot(df_endogenous.index, df_short[f'Lambda_obs_{c+1}'], color=f'C{c}', alpha=0.3, label=f'Obs Signal + Bias (C{c+1})')
        axes[2].plot(df_endogenous.index, df_endogenous[f'State_lambda_{c+1}'], color=f'C{c}', linewidth=2, label=f'Filtered True $\lambda_{{{c+1},t}}$')
    axes[2].set_title('True Sticky Info Parameter ($\lambda_{c,t}$) vs Upward-Biased Proxy')
    axes[2].set_ylabel('Sticky Info $\lambda$')
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].legend(loc='best', ncol=2, fontsize='small')
    axes[2].grid(True, alpha=0.3)

    c_idx = CONFIG['C']
    axes[3].plot(df_endogenous.index, df_endogenous[f'Obs_Z_{c_idx}'], label=f'Observed 1Y Survey $Z_{{{c_idx},t}}$', color='orange', alpha=0.4)
    axes[3].plot(df_endogenous.index, df_endogenous[f'Implied_Z_{c_idx}'], label=f'Model Implied 1Y $Z_{{{c_idx},t}}$', color='darkorange', linewidth=2)
    axes[3].set_title(f'1-Year Survey Expectations vs Implied (Cohort {c_idx})')
    axes[3].set_ylabel('Rate (%)')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)

    # ★ 新規パネル: 5年先インフレ予想
    for c in range(CONFIG['C']):
        axes[4].plot(df_endogenous.index, df_endogenous[f'Exp_5Y_{c+1}'], label=f'Model Implied 5Y Exp (Cohort {c+1})', linewidth=2)
    axes[4].set_title('Endogenous 5-Year (60-Month) Inflation Expectations')
    axes[4].set_ylabel('Rate (%)')
    axes[4].legend(loc='upper right')
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()
