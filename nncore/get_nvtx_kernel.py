#!/usr/bin/env python3
import sqlite3
import pandas as pd
import sys
import os
from typing import List, Dict, Any


class NVTXKernelAnalyzer:
    def __init__(self, sqlite_file: str) -> None:
        """
        Initialize the NVTXKernelAnalyzer with the path to the SQLite file.

        Args:
            sqlite_file: Exported SQLite file from NVIDIA Nsight profiling tool.
            e.g. nsys export --type sqlite --force-overwrite=true -o profile.sqlite profile.nsys-rep
        """
        self.sqlite_file = sqlite_file
        self.conn = None

    def connect(self) -> None:
        """
        Connect to the SQLite database
        """
        if not os.path.exists(self.sqlite_file):
            raise FileNotFoundError(f"SQLite File Not Found: {self.sqlite_file}")

        self.conn = sqlite3.connect(self.sqlite_file)
        print(f"Connected: {self.sqlite_file}")

    def disconnect(self) -> None:
        """
        Disconnect from the SQLite database
        """
        if self.conn:
            self.conn.close()
            self.conn = None

    def get_nvtx_ranges(self, range_name: str = None) -> pd.DataFrame:
        """
        Get NVTX ranges from the database

        Args:
            range_name: Optional NVTX range name to filter, if None, get all ranges.

        Returns:
            DataFrame of NVTX ranges
        """
        query = """
        SELECT 
            start AS start_ns,
            end AS end_ns,
            ((end - start) * 1e-6) AS duration_ms,
            text AS range_name,
            globalTid,
            eventType
        FROM NVTX_EVENTS
        WHERE eventType = 59  -- 59 stands for NVTXRange Event
        """

        if range_name:
            query += f" AND text = '{range_name}'"

        query += " ORDER BY start"

        return pd.read_sql_query(query, self.conn)

    def get_kernels_in_range(self, range_start: int, range_end: int) -> pd.DataFrame:
        """
        Get Kernel information within a specific NVTX range

        Args:
            range_start: Start time for the range (ns)
            range_end: End time for the range(ns)

        Returns:
            DataFrame of Kernel information within the specified range
        """

        query = """
        SELECT 
            k.start AS start_ns,
            k.end AS end_ns,
            ((k.end - k.start) * 1e-6) AS duration_ms,
            s.value AS kernel_name,
            k.gridX, k.gridY, k.gridZ,
            k.blockX, k.blockY, k.blockZ,
            k.registersPerThread,
            k.staticSharedMemory,
            k.dynamicSharedMemory,
            k.localMemoryPerThread,
            k.localMemoryTotal,
            k.deviceId,
            k.contextId,
            k.streamId,
            k.correlationId
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.mangledName = s.id
        WHERE k.start >= ? AND k.end <= ?
        ORDER BY k.start
        """

        return pd.read_sql_query(query, self.conn, params=(range_start, range_end))

    def analyze_nvtx_range(self, nvtx_range_name: str) -> Dict[str, Any]:
        """
        Analyze kernels within a specific NVTX range

        Args:
            nvtx_range_name: Name of the NVTX range to analyze

        Returns:
            A dictionary containing analysis results
        """
        print(f"Analyzing NVTX Range: '{nvtx_range_name}'")

        # Get NVTX ranges with the specified name
        nvtx_ranges = self.get_nvtx_ranges(nvtx_range_name)

        if nvtx_ranges.empty:
            print(f"Can not find NVTX range'{nvtx_range_name}' in the profile data.")
            available_ranges = self.get_nvtx_ranges()
            if not available_ranges.empty:
                print("\nValid NVTX Ranges:")
                for name in available_ranges["range_name"].unique()[
                    :10
                ]:  # 只显示前10个
                    print(f"  - {name}")
                if len(available_ranges["range_name"].unique()) > 10:
                    print(
                        f"  ... There are {len(available_ranges['range_name'].unique()) - 10} more ranges."
                    )
            return {}

        print(f"Found {len(nvtx_ranges)} instances of NVTX range. ")

        results = {}

        for idx, nvtx_range in nvtx_ranges.iterrows():
            range_id = f"{nvtx_range_name}_{idx+1}"
            print(f"\nAnalyzing {idx+1} instance:")
            print(
                f"  时间范围: {nvtx_range['start_ns']/1e6:.3f} ms - {nvtx_range['end_ns']/1e6:.3f} ms"
            )
            print(f"  持续时间: {nvtx_range['duration_ms']:.3f} ms")

            # 获取该范围内的Kernel
            kernels = self.get_kernels_in_range(
                nvtx_range["start_ns"], nvtx_range["end_ns"]
            )

            if kernels.empty:
                print("  该范围内没有找到任何Kernel")
                results[range_id] = {
                    "nvtx_info": nvtx_range.to_dict(),
                    "kernels": pd.DataFrame(),
                    "summary": {"total_kernels": 0, "total_kernel_time": 0.0},
                }
                continue

            print(f"  找到 {len(kernels)} 个Kernel")

            # 计算统计信息
            total_kernel_time = kernels["duration_ms"].sum()
            kernel_utilization = (total_kernel_time / nvtx_range["duration_ms"]) * 100

            summary = {
                "total_kernels": len(kernels),
                "total_kernel_time": total_kernel_time,
                "nvtx_range_time": nvtx_range["duration_ms"],
                "kernel_utilization_percent": kernel_utilization,
                "avg_kernel_time": kernels["duration_ms"].mean(),
                "max_kernel_time": kernels["duration_ms"].max(),
                "min_kernel_time": kernels["duration_ms"].min(),
            }

            results[range_id] = {
                "nvtx_info": nvtx_range.to_dict(),
                "kernels": kernels,
                "summary": summary,
            }

            # 打印简要统计
            print(f"  Kernel总执行时间: {total_kernel_time:.3f} ms")
            print(f"  GPU利用率: {kernel_utilization:.1f}%")
            print(f"  平均Kernel时间: {summary['avg_kernel_time']:.3f} ms")

        return results

    def generate_report(self, results: Dict[str, Any], output_file: str = None):
        """
        生成详细的分析报告

        Args:
            results: analyze_nvtx_range返回的结果
            output_file: 输出文件路径（可选）
        """
        if not results:
            print("没有数据可生成报告")
            return

        report_lines = []

        for range_id, data in results.items():
            report_lines.append(f"=" * 80)
            report_lines.append(f"NVTX范围: {range_id}")
            report_lines.append(f"=" * 80)

            nvtx_info = data["nvtx_info"]
            report_lines.append(
                f"时间范围: {nvtx_info['start_ns']/1e6:.3f} ms - {nvtx_info['end_ns']/1e6:.3f} ms"
            )
            report_lines.append(f"持续时间: {nvtx_info['duration_ms']:.3f} ms")

            summary = data["summary"]
            report_lines.append(f"\n摘要统计:")
            report_lines.append(f"  Kernel数量: {summary['total_kernels']}")
            report_lines.append(
                f"  Kernel总时间: {summary['total_kernel_time']:.3f} ms"
            )
            report_lines.append(
                f"  GPU利用率: {summary['kernel_utilization_percent']:.1f}%"
            )
            report_lines.append(
                f"  平均Kernel时间: {summary['avg_kernel_time']:.3f} ms"
            )
            report_lines.append(f"  最长Kernel: {summary['max_kernel_time']:.3f} ms")
            report_lines.append(f"  最短Kernel: {summary['min_kernel_time']:.3f} ms")

            kernels = data["kernels"]
            if not kernels.empty:
                report_lines.append(f"\n详细的Kernel信息:")
                report_lines.append("-" * 120)

                # 按执行时间排序
                kernels_sorted = kernels.sort_values("duration_ms", ascending=False)

                for idx, kernel in kernels_sorted.iterrows():
                    report_lines.append(
                        # TODO: Fix this, its mangled name now
                        f"{kernel['kernel_name']:50} | "
                        f"{kernel['duration_ms']:8.3f} ms | "
                        f"Grid: ({kernel['gridX']:4}, {kernel['gridY']:3}, {kernel['gridZ']:2}) | "
                        f"Block: ({kernel['blockX']:3}, {kernel['blockY']:3}, {kernel['blockZ']:3})"
                    )

            report_lines.append("\n")

        # 输出到控制台
        print("\n" + "=" * 80)
        print("详细分析报告")
        print("=" * 80)
        for line in report_lines:
            print(line)

        # 输出到文件
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(report_lines))
            print(f"\n报告已保存到: {output_file}")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


def main():
    if len(sys.argv) != 3:
        print("用法: python analyze_nvtx_kernels.py <sqlite_file> <nvtx_range_name>")
        print('示例: python analyze_nvtx_kernels.py profile.sqlite "MyMatmul"')
        sys.exit(1)

    sqlite_file = sys.argv[1]
    nvtx_range_name = sys.argv[2]

    try:
        with NVTXKernelAnalyzer(sqlite_file) as analyzer:
            # 分析指定范围的Kernel
            results = analyzer.analyze_nvtx_range(nvtx_range_name)

            if results:
                # 生成报告
                output_file = (
                    f"{os.path.splitext(sqlite_file)[0]}_{nvtx_range_name}_report.txt"
                )
                analyzer.generate_report(results, output_file)
            else:
                print(f"没有找到关于 '{nvtx_range_name}' 的分析结果")

    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()