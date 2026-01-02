"""Performance profiler for measuring execution time of RAG pipeline stages."""
import time
from contextlib import contextmanager
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class StageTiming:
    """Timing information for a single stage."""
    stage_name: str
    duration_seconds: float
    start_time: float
    end_time: float
    metadata: Dict = field(default_factory=dict)


class PerformanceProfiler:
    """Profiler for measuring execution time of different stages."""
    
    def __init__(self):
        """Initialize the profiler."""
        self.stages: List[StageTiming] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.active_stages: Dict[str, float] = {}  # stage_name -> start_time
    
    def start(self):
        """Start profiling (marks the beginning of total execution)."""
        self.start_time = time.perf_counter()
        self.stages.clear()
        self.active_stages.clear()
    
    def end(self):
        """End profiling (marks the end of total execution)."""
        self.end_time = time.perf_counter()
    
    @contextmanager
    def stage(self, stage_name: str, metadata: Optional[Dict] = None):
        """
        Context manager for timing a stage.
        
        Usage:
            with profiler.stage("embedding_generation"):
                # code to measure
                pass
        """
        start = time.perf_counter()
        self.active_stages[stage_name] = start
        
        try:
            yield
        finally:
            end = time.perf_counter()
            duration = end - start
            
            stage_timing = StageTiming(
                stage_name=stage_name,
                duration_seconds=duration,
                start_time=start,
                end_time=end,
                metadata=metadata or {}
            )
            self.stages.append(stage_timing)
            
            # Remove from active stages
            if stage_name in self.active_stages:
                del self.active_stages[stage_name]
    
    def get_total_time(self) -> float:
        """Get total execution time in seconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
    
    def get_stage_time(self, stage_name: str) -> float:
        """Get total time for a specific stage (sum if called multiple times)."""
        total = 0.0
        for stage in self.stages:
            if stage.stage_name == stage_name:
                total += stage.duration_seconds
        return total
    
    def get_stage_count(self, stage_name: str) -> int:
        """Get number of times a stage was executed."""
        return sum(1 for stage in self.stages if stage.stage_name == stage_name)
    
    def get_slowest_stage(self) -> Optional[StageTiming]:
        """Get the slowest single stage execution."""
        if not self.stages:
            return None
        return max(self.stages, key=lambda s: s.duration_seconds)
    
    def get_stage_summary(self) -> Dict[str, Dict]:
        """
        Get summary of all stages with aggregated statistics.
        
        Returns:
            Dict mapping stage_name to {
                'total_time': float,
                'count': int,
                'avg_time': float,
                'min_time': float,
                'max_time': float
            }
        """
        summary = defaultdict(lambda: {
            'total_time': 0.0,
            'count': 0,
            'times': []
        })
        
        for stage in self.stages:
            summary[stage.stage_name]['total_time'] += stage.duration_seconds
            summary[stage.stage_name]['count'] += 1
            summary[stage.stage_name]['times'].append(stage.duration_seconds)
        
        # Calculate statistics
        result = {}
        for stage_name, stats in summary.items():
            times = stats['times']
            result[stage_name] = {
                'total_time': stats['total_time'],
                'count': stats['count'],
                'avg_time': stats['total_time'] / stats['count'] if stats['count'] > 0 else 0.0,
                'min_time': min(times) if times else 0.0,
                'max_time': max(times) if times else 0.0
            }
        
        return result
    
    def print_report(self, detailed: bool = False):
        """
        Print a human-readable performance report.
        
        Args:
            detailed: If True, print detailed information for each stage execution
        """
        total_time = self.get_total_time()
        summary = self.get_stage_summary()
        slowest = self.get_slowest_stage()
        
        print("\n" + "=" * 80)
        print("📊 PERFORMANCE PROFILING REPORT")
        print("=" * 80)
        
        # Total time
        print(f"\n⏱️  Total Execution Time: {total_time:.3f} seconds ({total_time*1000:.1f} ms)")
        
        if not summary:
            print("\n⚠️  No stages were profiled.")
            return
        
        # Stage summary table
        print("\n" + "-" * 80)
        print("📋 Stage Summary (sorted by total time)")
        print("-" * 80)
        print(f"{'Stage Name':<35} {'Total (s)':<12} {'Count':<8} {'Avg (s)':<12} {'% of Total':<10}")
        print("-" * 80)
        
        # Sort by total time (descending)
        sorted_stages = sorted(summary.items(), key=lambda x: x[1]['total_time'], reverse=True)
        
        for stage_name, stats in sorted_stages:
            total = stats['total_time']
            count = stats['count']
            avg = stats['avg_time']
            percentage = (total / total_time * 100) if total_time > 0 else 0.0
            
            print(f"{stage_name:<35} {total:<12.3f} {count:<8} {avg:<12.3f} {percentage:<10.1f}%")
        
        # Slowest stage
        if slowest:
            print("\n" + "-" * 80)
            print("🐌 Slowest Single Stage Execution:")
            print("-" * 80)
            print(f"  Stage: {slowest.stage_name}")
            print(f"  Duration: {slowest.duration_seconds:.3f} seconds ({slowest.duration_seconds*1000:.1f} ms)")
            print(f"  Percentage of total: {(slowest.duration_seconds / total_time * 100):.1f}%")
            if slowest.metadata:
                print(f"  Metadata: {slowest.metadata}")
        
        # Detailed view
        if detailed:
            print("\n" + "-" * 80)
            print("📝 Detailed Stage Executions:")
            print("-" * 80)
            for i, stage in enumerate(self.stages, 1):
                print(f"\n{i}. {stage.stage_name}")
                print(f"   Duration: {stage.duration_seconds:.3f} seconds ({stage.duration_seconds*1000:.1f} ms)")
                print(f"   Time: {stage.start_time:.3f} - {stage.end_time:.3f}")
                if stage.metadata:
                    print(f"   Metadata: {json.dumps(stage.metadata, indent=6)}")
        
        print("\n" + "=" * 80)
    
    def get_report_dict(self) -> Dict:
        """Get a dictionary representation of the report."""
        total_time = self.get_total_time()
        summary = self.get_stage_summary()
        slowest = self.get_slowest_stage()
        
        return {
            'total_time_seconds': total_time,
            'total_time_ms': total_time * 1000,
            'stage_summary': summary,
            'slowest_stage': {
                'name': slowest.stage_name if slowest else None,
                'duration_seconds': slowest.duration_seconds if slowest else 0.0,
                'duration_ms': slowest.duration_seconds * 1000 if slowest else 0.0,
                'metadata': slowest.metadata if slowest else {}
            },
            'all_stages': [
                {
                    'name': stage.stage_name,
                    'duration_seconds': stage.duration_seconds,
                    'duration_ms': stage.duration_seconds * 1000,
                    'start_time': stage.start_time,
                    'end_time': stage.end_time,
                    'metadata': stage.metadata
                }
                for stage in self.stages
            ]
        }
    
    def export_json(self, filepath: str):
        """Export the report to a JSON file."""
        report = self.get_report_dict()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"📄 Performance report exported to: {filepath}")


# Global profiler instance (can be used across the application)
_global_profiler: Optional[PerformanceProfiler] = None


def get_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def reset_profiler():
    """Reset the global profiler."""
    global _global_profiler
    _global_profiler = PerformanceProfiler()
















