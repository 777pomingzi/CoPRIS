import re
import numpy as np
import pandas as pd

file_path = "/home/test/test06/qzk/verl-partial-agent-loop/output_302029.log"  # 改成你的实际路径

# 正则模式
step_pattern = re.compile(r"step:(\d+)")
gen_pattern = re.compile(r"timing_s/gen:([\d\.]+)")
step_time_pattern = re.compile(r"timing_s/step:([\d\.]+)")

steps, gens, step_times = [], [], []

with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        step_match = step_pattern.search(line)
        gen_match = gen_pattern.search(line)
        step_time_match = step_time_pattern.search(line)
        if step_match and gen_match and step_time_match:
            steps.append(int(step_match.group(1)))
            gens.append(float(gen_match.group(1)))
            step_times.append(float(step_time_match.group(1)))

# 保存为 DataFrame
df = pd.DataFrame({
    "step": steps,
    "timing_s/gen": gens,
    "timing_s/step": step_times
})

print(df)

if gens:
    print(f"\nMean timing_s/gen = {np.mean(gens):.4f}")
    print(f"Mean timing_s/step = {np.mean(step_times):.4f}")

# 导出CSV（可选）
df.to_csv("step_gen_times.csv", index=False)
print("\n已导出到 step_gen_times.csv")
