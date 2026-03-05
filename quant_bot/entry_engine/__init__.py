"""
entry_engine — やがみ5条件エントリースコアリングエンジン

使い方:
    from quant_bot.entry_engine.scanner import EntryScanner
    import yaml

    with open('quant_bot/entry_engine/config.yaml') as f:
        config = yaml.safe_load(f)

    scanner = EntryScanner(config)
    for record in scanner.scan_dataframe(df, 'XAU_USD', 'H4'):
        print(record)  # JSONL形式 dict
"""
