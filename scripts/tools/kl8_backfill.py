"""快乐8自动补数脚本 — 循环分批抓取直到800期"""
import sys
import os
import time

# 确保项目路径 — 添加项目根目录而非scripts目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.kl8.fetch import (
    fetch_kl8_history_backfill_batch,
    count_valid_history_periods,
    check_need_backfill,
)

TARGET = 800
MAX_ITERATIONS = 40  # 最多40轮（每轮5页约250期，40轮足够）

def main():
    current = count_valid_history_periods()
    print(f"当前历史期数: {current} / 目标: {TARGET}")

    iteration = 0
    while current < TARGET and iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"\n=== 第 {iteration} 轮补数 (当前 {current} 期) ===")

        result = fetch_kl8_history_backfill_batch(
            target_periods=TARGET,
            pages_per_batch=5,
        )

        if result.get('success'):
            before = result.get('periods_before', 0)
            after = result.get('periods_after', 0)
            print(f"  成功: {before} -> {after} 期")

            if result.get('completed'):
                print(f"\n🎉 补数完成! 已达到 {after} 期目标")
                break
        else:
            error = result.get('error', 'unknown')
            print(f"  失败: {error}")
            # 失败后等30秒再试
            print("  等待30秒后重试...")
            time.sleep(30)
            continue

        current = count_valid_history_periods()

        # 每轮之间等5秒，避免连续请求太密集
        if current < TARGET:
            print(f"  等待5秒后继续下一轮...")
            time.sleep(5)

    final = count_valid_history_periods()
    print(f"\n最终历史期数: {final}")
    if final >= TARGET:
        print("✅ 补数目标达成!")
    else:
        print(f"⚠ 补数未完成，还差 {TARGET - final} 期")

    # 检查状态
    status = check_need_backfill()
    print(f"补数状态: need_backfill={status.get('need_backfill')}, current={status.get('current_periods')}")

if __name__ == '__main__':
    main()
