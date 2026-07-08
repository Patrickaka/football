with open('okooo_page.html', 'r', encoding='utf-8') as f:
    html = f.read()

import re

print("=== 分析okooo主页结构 ===")

script_tags = re.findall(r'<script[^>]*>(.*?)</script>', html, re.DOTALL)
print(f'找到 {len(script_tags)} 个script标签')

for i, script in enumerate(script_tags):
    if len(script) > 100:
        print(f'\n脚本 {i} (长度: {len(script)})')
        print(script[:500])
        if 'match' in script.lower() or 'odds' in script.lower() or 'id' in script.lower():
            print('>>> 包含match/odds/id关键词')

div_classes = re.findall(r'<div[^>]*class="([^"]+)"', html)
print(f'\n\n找到 {len(set(div_classes))} 种不同的div class')
print(f'前30个: {list(set(div_classes))[:30]}')

table_pattern = re.findall(r'<table[^>]*>(.*?)</table>', html, re.DOTALL)
print(f'\n\n找到 {len(table_pattern)} 个表格')
for i, table in enumerate(table_pattern[:3]):
    tr_count = len(re.findall(r'<tr[^>]*>', table))
    td_count = len(re.findall(r'<td[^>]*>', table))
    print(f'表格 {i}: {tr_count}行, {td_count}列')
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table, re.DOTALL)
    for j, row in enumerate(rows[:3]):
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
        if cells:
            clean = [re.sub(r'<[^>]+>', '', c).strip()[:30] for c in cells]
            print(f'  行 {j}: {clean}')
