#!/usr/bin/env python3
"""
Audit Script #1: Benchmark Harness Analysis
Goal: Examine if a runtime's bench harness can generate agentic workloads
       (multi-turn, session, shared prefix, tool-call capable)

Usage: python3 1_bench_harness_audit.py <path_to_repo>
"""

import os
import sys
import re
from pathlib import Path
from collections import defaultdict

# ANSI colors for output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

class BenchHarnessAudit:
    def __init__(self, repo_path):
        self.repo_path = Path(repo_path).resolve()
        if not self.repo_path.exists():
            print(f"{RED}ERROR: Repo path does not exist: {repo_path}{RESET}")
            sys.exit(1)

        self.findings = {
            'harness_locations': [],
            'multiturn_features': [],
            'session_features': [],
            'prefix_features': [],
            'agent_features': [],
            'datasets': [],
            'arrival_models': [],
        }

    def grep(self, pattern, extensions=['.py', '.cpp', '.hpp', '.h', '.sh'],
             max_results=50, case_insensitive=True):
        """Simple grep implementation"""
        results = []
        flags = re.IGNORECASE if case_insensitive else 0
        regex = re.compile(pattern, flags)

        for fpath in self.repo_path.rglob('*'):
            if fpath.is_file() and fpath.suffix in extensions:
                try:
                    content = fpath.read_text(encoding='utf-8', errors='ignore')
                    for line_no, line in enumerate(content.split('\n'), 1):
                        if regex.search(line):
                            results.append({
                                'file': str(fpath.relative_to(self.repo_path)),
                                'line': line_no,
                                'text': line.strip()[:120]
                            })
                            if len(results) >= max_results:
                                return results
                except:
                    pass
        return results

    def find_benchmark_dirs(self):
        """Find all benchmark-related directories"""
        print(f"\n{BLUE}=== Step 1: Locating Benchmark Directories ==={RESET}")
        bench_patterns = ['benchmark', 'benchmarks', 'bench']

        for pattern in bench_patterns:
            for dpath in self.repo_path.rglob(pattern):
                if dpath.is_dir() and not any(x in str(dpath) for x in ['.git', '__pycache__', '.pytest']):
                    print(f"  {GREEN}✓{RESET} {dpath.relative_to(self.repo_path)}/")
                    self.findings['harness_locations'].append(str(dpath.relative_to(self.repo_path)))

        if not self.findings['harness_locations']:
            print(f"  {YELLOW}⚠ No benchmark directories found{RESET}")

    def analyze_workload_capabilities(self):
        """Check if harness can generate multi-turn, session, prefix-sharing, agent workloads"""
        print(f"\n{BLUE}=== Step 2: Workload Shape Analysis ==={RESET}")

        # Keywords for agentic features
        patterns = {
            'multiturn_features': [
                r'multi.?turn', r'conversation', r'dialogue', r'multi.?step',
                r'trajectory', r'multi.?round', r'turn.?\d+'
            ],
            'session_features': [
                r'session', r'stateful', r'conversation.?id', r'thread', r'ctx.?window'
            ],
            'prefix_features': [
                r'prefix.?repeat', r'prefix.?shar', r'prefix.?pool', r'prefix.?cache',
                r'shared.?prefix', r'kv.?cache.?shar', r'block.?hash', r'radix', r'preemp'
            ],
            'agent_features': [
                r'agent', r'tool.?call', r'function.?call', r'act.*observation',
                r'react', r'agent.?loop', r'agentic'
            ]
        }

        for feature_type, regexes in patterns.items():
            for regex in regexes:
                results = self.grep(regex, extensions=['.py', '.yaml', '.yml', '.md', '.json'], max_results=20)
                if results:
                    self.findings[feature_type].extend(results)
                    print(f"  {GREEN}✓{RESET} {feature_type}: {regex}")
                    for r in results[:3]:  # Show first 3 matches
                        print(f"      {r['file']}:{r['line']} → {r['text'][:80]}")
                    if len(results) > 3:
                        print(f"      ... ({len(results)} total matches)")

        # Report what's missing
        missing = [k for k, v in self.findings.items()
                   if k.endswith('_features') and not v]
        if missing:
            print(f"\n  {RED}✗ Missing features:{RESET}")
            for m in missing:
                print(f"      • {m}")

    def analyze_datasets(self):
        """Analyze available dataset backends"""
        print(f"\n{BLUE}=== Step 3: Dataset Backends ==={RESET}")

        # Common dataset patterns
        dataset_patterns = [
            (r'--dataset.?name|dataset\s*=\s*["\']([^"\']+)["\']', 'dataset arg'),
            (r'class\s+(\w*Dataset)', 'Dataset class'),
            (r'"(random|sharegpt|sonnet|hf|alpaca|openorca|wikitext)', 'Dataset name'),
            (r'(prefix.?repeat|multi.?turn|conversation)', 'Multi-turn dataset'),
            (r'(gsm8k|mmlu|arc|hellaswag|truthful)', 'Benchmark dataset'),
        ]

        for pattern, desc in dataset_patterns:
            results = self.grep(pattern, extensions=['.py'], max_results=30)
            if results:
                print(f"  {GREEN}•{RESET} {desc}:")
                unique = set(r['text'][:100] for r in results)
                for text in list(unique)[:5]:
                    print(f"      {text}")
                if len(unique) > 5:
                    print(f"      ... ({len(unique)} total)")
                self.findings['datasets'].extend(results)

    def analyze_arrival_models(self):
        """Check how requests arrive (poisson vs dependency-driven)"""
        print(f"\n{BLUE}=== Step 4: Request Arrival Model ==={RESET}")

        patterns = [
            (r'poisson|request.?rate|--qps|--rps', 'Poisson/QPS (iid single-turn)'),
            (r'multi.?turn|conversation|session|stateful', 'Potential multi-turn (check code)'),
            (r'burstiness|burst|arrival.?pattern', 'Burst pattern'),
            (r'dependent|dependency|tool.?exec|tool.?latency', 'Dependency-driven'),
        ]

        for pattern, desc in patterns:
            results = self.grep(pattern, extensions=['.py', '.cpp'], max_results=20)
            if results:
                print(f"  {GREEN}•{RESET} {desc}:")
                for r in results[:2]:
                    print(f"      {r['file']}:{r['line']}")
                if len(results) > 2:
                    print(f"      ... ({len(results)} total)")
                self.findings['arrival_models'].append((desc, len(results)))

    def verdict(self):
        """Final assessment"""
        print(f"\n{BLUE}=== VERDICT ==={RESET}\n")

        has_multiturn = bool(self.findings['multiturn_features'])
        has_session = bool(self.findings['session_features'])
        has_prefix = bool(self.findings['prefix_features'])
        has_agent = bool(self.findings['agent_features'])

        print(f"  Multi-turn capable:     {GREEN + 'YES' if has_multiturn else RED + 'NO' + RESET}")
        print(f"  Session-aware:          {GREEN + 'YES' if has_session else RED + 'NO' + RESET}")
        print(f"  Prefix-sharing aware:   {GREEN + 'YES' if has_prefix else RED + 'NO' + RESET}")
        print(f"  Agent/tool-call aware:  {GREEN + 'YES' if has_agent else RED + 'NO' + RESET}")
        print()

        score = sum([has_multiturn, has_session, has_prefix, has_agent])
        if score == 4:
            print(f"  {GREEN}★★★★★ Full agentic support detected{RESET}")
        elif score == 3:
            print(f"  {GREEN}★★★★☆ Strong agentic support{RESET}")
        elif score == 2:
            print(f"  {YELLOW}★★★☆☆ Partial support (likely prefix-cache focused){RESET}")
        elif score == 1:
            print(f"  {YELLOW}★★☆☆☆ Minimal agentic features{RESET}")
        else:
            print(f"  {RED}★☆☆☆☆ No agentic features detected (pure single-turn harness){RESET}")

        print(f"\n  {BLUE}→ This is a '{['iid single-turn', 'prefix-cache microbench', 'proto-agentic', 'agentic-aware', 'full agentic'][min(score, 4)]}'  harness{RESET}\n")

    def run(self):
        """Execute the full audit"""
        print(f"{BLUE}{'='*70}")
        print(f"Benchmark Harness Audit: {self.repo_path.name}")
        print(f"{'='*70}{RESET}")

        self.find_benchmark_dirs()
        self.analyze_workload_capabilities()
        self.analyze_datasets()
        self.analyze_arrival_models()
        self.verdict()

        # Summary for easy reference
        print(f"\n{BLUE}=== Raw Findings Summary ==={RESET}")
        for key, val in self.findings.items():
            if val:
                print(f"  {key}: {len(val)} items found")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path_to_repo>")
        print(f"Example: {sys.argv[0]} ../vllm")
        sys.exit(1)

    audit = BenchHarnessAudit(sys.argv[1])
    audit.run()
