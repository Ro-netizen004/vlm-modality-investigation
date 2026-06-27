import json, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

models = [
    'Qwen2-VL-2B-Instruct',
    'llava-v1.6-mistral-7b-hf',
    'Idefics3-8B-Llama3',
    'MiniCPM-V-2_6',
    'InternVL2-8B',
    'llava-onevision-qwen2-7b-ov-hf',
]
benchmarks = ['svamp', 'math', 'aqua_rat', 'mathvista', 'ai2d', 'chartqa', 'scienceqa']

print(f"{'Benchmark':<12} {'Model':<38} {'C1':>5} {'C2':>5} {'Drop':>6} {'Sig':>4} {'TextPref':>9}")
print("-" * 85)
for b in benchmarks:
    for m in models:
        path = os.path.join('results', 'phase3', m, b, 'statistics.json')
        if not os.path.exists(path):
            continue
        with open(path, encoding='utf-8') as f:
            s = json.load(f)
        p = s.get('mcnemar_p', 1.0)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        tp = s.get('text_preference')
        tp_str = f"{tp:.3f}" if tp is not None else "N/A"
        # Protocol A: acc_text/acc_img; Protocol B: acc_text_only/acc_multimodal
        if 'acc_text' in s:
            c1, c2, drop = s['acc_text'], s['acc_img'], s['acc_drop']
        else:
            c1, c2 = s.get('acc_text_only', 0), s.get('acc_multimodal', 0)
            drop = c2 - c1
        print(f"{b:<12} {m:<38} {c1:>5.3f} {c2:>5.3f} {drop:>+6.3f} {sig:>4} {tp_str:>9}")
    print()
