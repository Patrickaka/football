#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试排列五数据网站"""

import urllib.request
import re

def main():
    url = 'https://www.8300.cn/kjhhis/5/200.html'
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    html = urllib.request.urlopen(req, timeout=30).read().decode('utf-8', 'ignore')
    
    # 搜索期号
    issues = re.findall(r'(\d{7})期', html)
    if issues:
        print(f'找到 {len(issues)} 个期号')
        print('前10个:', issues[:10])
    
    # 搜索数字球
    balls = re.findall(r'<span class="ball">(\d)</span>', html)
    if balls:
        print(f'\n找到 {len(balls)} 个数字球')
        print('前30个:', balls[:30])
        
        # 每5个一组显示
        print('\n按5个一组分组:')
        for i in range(0, min(30, len(balls)), 5):
            print(f'  {balls[i:i+5]}')

if __name__ == '__main__':
    main()
