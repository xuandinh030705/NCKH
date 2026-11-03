import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tutor.problemgen import generate
from tutor.mathcheck import parse_problem, solve, fmt_num

items = generate(1000, seed=7)
bad = []
for it in items:
    s = solve(it.text)
    if s is None:
        bad.append((it.id, it.text, "SOLVE_NONE", it.answer))
        continue
    got = sorted(round(v, 6) for v in s.answers())
    want = sorted(round(v, 6) for v in it.answers())
    if got != want:
        bad.append((it.id, it.text, got, want))
print("n", len(items), "bad", len(bad))
from collections import Counter
print(Counter(b[0].split("-")[1] for b in bad))
for b in bad[:8]:
    print(b)
print("topics", Counter(i.topic for i in items))
