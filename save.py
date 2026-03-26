    # ---------------------------------------------------------
    # ★ データフレームの組み立てと CSV への保存処理
    # ---------------------------------------------------------
    df_mock = pd.DataFrame({'S_t': S_data}, index=global_dates)
    df_mock.index.name = 'Date' # インデックス名を明示
    
    for c in range(CONFIG['C']):
        df_mock[f'Z_{c+1}'] = Z_data[:, c]
        df_mock[f'Z5_{c+1}'] = Z5_data[:, c]
        df_mock[f'Lambda_obs_{c+1}'] = lambda_obs_data[:, c]
        
        # 後で推計精度を確認できるように、真値も保存しておく
        df_mock[f'True_gamma_{c+1}'] = true_gamma[:, c]
        df_mock[f'True_lambda_{c+1}'] = true_lambda[:, c]
        
    # CONFIGで指定したファイル名（mock_data_final.csv）に書き出し
    df_mock.to_csv(CONFIG['mock_csv'], encoding='utf-8-sig')
    print(f"✅ モックデータを '{CONFIG['mock_csv']}' に保存しました。")
    
    # ---------------------------------------------------------
    # 以降は元のコードと同じ（短期データへのスライス処理など）
    # ---------------------------------------------------------
    df_short = df_mock.iloc[CONFIG['start_idx_short']:].copy()
    # ...
