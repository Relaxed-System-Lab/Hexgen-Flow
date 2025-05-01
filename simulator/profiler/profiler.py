from simulator.internal.analyzer import ModelAnalyzer

def profile(analyzer: ModelAnalyzer, seq_len: int, batch_size: int, w_bit: int, a_bit: int, kv_bit: int):
    results = analyzer.analyze(seqlen=seq_len, batchsize=batch_size, w_bit=w_bit, a_bit=a_bit, kv_bit=kv_bit, use_flashattention=True)
    print(results)
    analyzer.save_csv('.cache/profile/profile')