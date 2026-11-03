import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tutor.mathcheck import parse_problem, solve, fmt_num

tests = [
    "Tìm x, biết: x + 18 = 45", "Tìm x, biết: x * 4 = 36", "Tìm x, biết: x*3-4=11",
    "Tìm x, biết: x × 3 − 4 = 11", "Tìm x biết 24 : x = 6", "Tìm x, biết: x : 4 + 5 = 12",
    "Tìm x, biết: x - 8 = 9", "Tìm x, biết: 10 = x + 3", "Tìm x, biết: 3 * x + 7 = 25",
    "Hai bạn An và Bình có tổng cộng 24 cái kẹo. An có nhiều hơn Bình 6 cái kẹo. Hỏi mỗi bạn có bao nhiêu cái kẹo?",
    "Một ô tô đi quãng đường dài 120 km trong thời gian 2 giờ 30 phút. Hãy tính vận tốc của ô tô.",
    "Một cửa hàng có 240 kg gạo, đã bán 25% số gạo. Hỏi cửa hàng còn lại bao nhiêu ki-lô-gam gạo?",
    "Một cửa hàng có 180 kg gạo, trong đó số gạo tẻ gấp 2 lần số gạo nếp. Hỏi mỗi loại có bao nhiêu ki-lô-gam?",
    "Mẹ hơn con 24 tuổi. Tuổi mẹ gấp 3 lần tuổi con. Hỏi mỗi người bao nhiêu tuổi?",
    "Lan mua 5 quyển vở, mỗi quyển giá 8 nghìn đồng. Lan đưa cô bán hàng 50 nghìn đồng. Hỏi cô bán hàng phải trả lại Lan bao nhiêu nghìn đồng?",
]
for t in tests:
    p = parse_problem(t)
    s = solve(t)
    if s is None:
        ans = None
    elif isinstance(s.answer, tuple):
        ans = [fmt_num(v) for v in s.answer]
    else:
        ans = fmt_num(s.answer)
    print(f"{p.kind if p else None!s:12} -> {ans!s:12} {s.unit if s else '':10} | {t[:45]}")
    if s:
        print("     steps:", [(st.desc, st.expr, fmt_num(st.value)) for st in s.steps][:3])
