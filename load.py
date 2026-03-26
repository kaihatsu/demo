# =====================================================================
# 本番環境用: 実データ (CSV) の読み込みと前処理ルーチン
# =====================================================================
def load_real_data(csv_filepath=CONFIG['mock_csv']):
    """
    保存されたCSVファイルから実データを読み込み、UKFの推計パイプラインで
    要求されるフォーマットのタプルに変換して返します。
    
    【想定するCSVの要件】
    - インデックス (1列目): 日付 (YYYY-MM-DD形式)
    - S_t: マクロインフレ指標 (T_longの全期間分が必要)
    - Z_1, Z_2 ... : コーホートごとのサーベイ予想 (T_short期間分が必要)
    - Lambda_obs_1, Lambda_obs_2 ... : コーホートごとの粘着情報プロキシデータ (T_short期間分が必要)
    """
    # 1. CSVの読み込み (日付をインデックス（DatetimeIndex）としてパース)
    df_all = pd.read_csv(csv_filepath, index_col=0, parse_dates=True)
    
    # ---------------------------------------------------------
    # (A) Step 1 (マクロ推論) 用の全期間データの抽出
    # ---------------------------------------------------------
    S_data = df_all['S_t'].values
    
    # ロジスティックバイアス計算用のマクロボラティリティ (|Delta S_t|) を全期間で計算
    # 最初の期(t=0)は差分が取れないため、前期と同じ値(差分0)として扱う
    abs_delta_S = np.abs(np.diff(S_data, prepend=S_data[0]))
    
    # ---------------------------------------------------------
    # (B) Step 2 (ミクロ推計) 用の短期データへのスライス
    # ---------------------------------------------------------
    start_idx = CONFIG['start_idx_short']
    df_short = df_all.iloc[start_idx:].copy()
    
    # 短期期間のマクロ指標とボラティリティ
    S_short_obs = df_short['S_t'].values
    abs_delta_S_short = abs_delta_S[start_idx:]
    
    # コーホート数 C に合わせて動的にカラム名を生成し、サーベイ予想データを抽出
    Z_cols = [f'Z_{c+1}' for c in range(CONFIG['C'])]
    Z_short_obs = df_short[Z_cols].values
    
    # コーホートごとの粘着情報プロキシデータを抽出
    lambda_cols = [f'Lambda_obs_{c+1}' for c in range(CONFIG['C'])]
    lambda_obs_short = df_short[lambda_cols].values
    
    # ---------------------------------------------------------
    # (C) 観測方程式の計算に必要な「前期のサーベイ予想 (Z_lag)」の構築
    # ---------------------------------------------------------
    Z_lag_short = np.zeros((CONFIG['T_short'], CONFIG['C']))
    
    # t=0 (短期推計の初月) のラグデータは、全期間データ(df_all)の「短期開始月の1ヶ月前」から取得
    Z_lag_short[0, :] = df_all[Z_cols].iloc[start_idx - 1].values
    
    # t=1 以降は、今月のデータ (Z_short_obs) を1期分ズラして格納
    if CONFIG['T_short'] > 1:
        Z_lag_short[1:, :] = Z_short_obs[:-1, :]
        
    return S_data, S_short_obs, Z_short_obs, lambda_obs_short, Z_lag_short, abs_delta_S_short
