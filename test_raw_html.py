with open('okooo_page.html', 'r', encoding='utf-8') as f:
    html = f.read()

import re

table_pattern = re.findall(r'<table[^>]*>(.*?)</table>', html, re.DOTALL)
table = table_pattern[1]

rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table, re.DOTALL)

for i, row in enumerate(rows[:10]):
    cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
    if cells and len(cells) >= 3:
        print(f"\n=== 行 {i} ===")
        print(f"第1列原始内容:\n{cells[0][:500]}")
        print(f"\n第2列原始内容:\n{cells[1][:500]}")
        print(f"\n第3列原始内容:\n{cells[2][:1000]}")
        print(f"\n第6列原始内容:\n{cells[5][:500]}")
