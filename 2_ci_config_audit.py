#!/usr/bin/env python3
"""
Audit Script #2: CI Configuration Analysis
Goal: Determine what runtime truly guards in its CI
       (performance workload, accuracy tasks, optimization coverage)

Key insight: CI doesn't lie. If it's not tested daily, it doesn't matter.

Usage: python3 2_ci_config_audit.py <path_to_repo>
"""

import os
import sys
import re
import yaml
from pathlib import Path
from collections import defaultdict

# ANSI colors
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

class CIConfigAudit:
    def __init__(self, repo_path):
        self.repo_path = Path(repo_path).resolve()
        if not self.repo_path.exists():
            print(f"{RED}ERROR: Repo path does not exist: {repo_path}{RESET}")
            sys.exit(1)

        self.findings = {
            'perf_jobs': [],
            'accuracy_jobs': [],
            'agentic_tasks': [],
            'optimization_coverage': defaultdict(list),  # opt_name -> job_names
            'workload_types': defaultdict(list),  # workload -> jobs
        }

    def find_ci_configs(self):
        """Find all CI configuration files"""
        print(f"\n{BLUE}=== Step 1: Locating CI Configurations ==={RESET}")

        ci_dirs = ['.github/workflows', '.buildkite', '.circleci', '.travis.yml', 'azure-pipelines.yml']
        found_configs = []

        for ci_dir in ci_dirs:
            full_path = self.repo_path / ci_dir
            if full_path.exists():
                print(f"  {GREEN}✓{RESET} {ci_dir}/")
                for config_file in full_path.rglob('*'):
                    if config_file.is_file() and config_file.suffix in ['.yml', '.yaml']:
                        print(f"      └─ {config_file.name}")
                        found_configs.append(config_file)

        # Also check in tests/ for CI-adjacent configs
        for fpath in (self.repo_path / 'tests').rglob('*.yml') if (self.repo_path / 'tests').exists() else []:
            if 'ci' in fpath.name.lower() or 'bench' in fpath.name.lower():
                print(f"  {GREEN}✓{RESET} {fpath.relative_to(self.repo_path)}")
                found_configs.append(fpath)

        if not found_configs:
            print(f"  {YELLOW}⚠ No standard CI configs found{RESET}")

        return found_configs

    def parse_yaml_config(self, fpath):
        """Parse YAML CI config safely"""
        try:
            with open(fpath, 'r') as f:
                return yaml.safe_load(f)
        except:
            return None

    def extract_jobs_from_yaml(self, config, filename):
        """Extract job names and their commands from GitHub Actions"""
        jobs = []
        if not config:
            return jobs

        # GitHub Actions format
        if 'jobs' in config:
            for job_name, job_config in config.get('jobs', {}).items():
                steps = job_config.get('steps', [])
                commands = []
                for step in steps:
                    if 'run' in step:
                        commands.append(step['run'])
                jobs.append({
                    'name': job_name,
                    'file': filename,
                    'commands': commands
                })

        return jobs

    def analyze_perf_jobs(self, config_files):
        """Identify performance-related CI jobs"""
        print(f"\n{BLUE}=== Step 2: Performance Testing Jobs ==={RESET}")

        perf_patterns = [
            r'benchmark|bench|perf|throughput|latency|speed|performance',
            r'serving|request.?rate|batch.*size',
            r'profile|profile.*performance'
        ]

        for config_file in config_files:
            content = config_file.read_text(encoding='utf-8', errors='ignore')

            for pattern in perf_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    print(f"  {GREEN}✓{RESET} {config_file.name}")

                    # Extract job names and commands
                    config = self.parse_yaml_config(config_file)
                    jobs = self.extract_jobs_from_yaml(config, config_file.name)

                    for job in jobs:
                        if any(re.search(p, job['name'], re.IGNORECASE) for p in perf_patterns):
                            print(f"      Job: {YELLOW}{job['name']}{RESET}")
                            for cmd in job['commands'][:2]:  # Show first 2 commands
                                print(f"        └─ {cmd[:100]}")
                            self.findings['perf_jobs'].append(job)
                    break  # One match per file is enough

    def analyze_accuracy_jobs(self, config_files):
        """Identify accuracy/evaluation CI jobs"""
        print(f"\n{BLUE}=== Step 3: Accuracy Testing Jobs ==={RESET}")

        accuracy_patterns = [
            (r'accuracy|eval|lm.?eval|correctness|quality', 'General accuracy'),
            (r'gsm8k|mmlu|arc|hellaswag|truthful|lfqa', 'Standard benchmarks'),
            (r'bfcl|gorilla|tool.?call.*eval|function.?call.*test', 'Tool-calling accuracy'),
            (r'tau.?bench|agent.?bench|swe.?bench', 'Agentic benchmarks'),
            (r'perplexity|ppl\b', 'Perplexity tests'),
        ]

        for config_file in config_files:
            content = config_file.read_text(encoding='utf-8', errors='ignore')
            found_any = False

            for pattern, desc in accuracy_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    if not found_any:
                        print(f"  {GREEN}✓{RESET} {config_file.name}")
                        found_any = True

                    # Count occurrences
                    count = len(re.findall(pattern, content, re.IGNORECASE))
                    print(f"      {YELLOW}•{RESET} {desc}: {count} mention(s)")

                    # Extract specifics
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if re.search(pattern, line, re.IGNORECASE):
                            print(f"          Line {i+1}: {line.strip()[:100]}")
                            break

                    self.findings['accuracy_jobs'].append({
                        'file': config_file.name,
                        'pattern': desc,
                        'matches': count
                    })

    def check_agentic_coverage(self, config_files):
        """Check if agentic/tool-calling tasks are covered"""
        print(f"\n{BLUE}=== Step 4: Agentic & Tool-Calling Coverage ==={RESET}")

        agentic_keywords = [
            'bfcl', 'gorilla', 'tool.?call', 'function.?call',
            'tau.?bench', 'agent.?bench', 'react', 'agentic',
            'multi.?turn.*eval', 'trajectory'
        ]

        found_agentic = []
        for config_file in config_files:
            content = config_file.read_text(encoding='utf-8', errors='ignore')
            for keyword in agentic_keywords:
                if re.search(keyword, content, re.IGNORECASE):
                    found_agentic.append({
                        'file': config_file.name,
                        'keyword': keyword
                    })

        if found_agentic:
            print(f"  {GREEN}✓ Agentic tasks found:{RESET}")
            for item in found_agentic:
                print(f"      {item['file']}: {item['keyword']}")
            self.findings['agentic_tasks'] = found_agentic
        else:
            print(f"  {RED}✗ No agentic/tool-calling tasks in CI{RESET}")
            print(f"    This is a {RED}CRITICAL finding{RESET} — the runtime doesn't CI-guard")
            print(f"    whether optimizations (KV cache quantization, eviction, etc.)")
            print(f"    maintain tool-calling accuracy.")

    def check_optimization_coverage(self, config_files):
        """Check which optimizations have explicit accuracy tests"""
        print(f"\n{BLUE}=== Step 5: Optimization-Specific Coverage ==={RESET}")

        optimizations = {
            'prefix_caching': [r'prefix.?cach', r'prompt.?cach', r'kv.?shar'],
            'kv_quantization': [r'kv.?cache.?quantiz', r'kv.?cache.?compres', r'kv.?cache.*int'],
            'cache_eviction': [r'cache.?eviction', r'cache.?budget', r'kv.*evict'],
            'speculative_decoding': [r'speculative', r'spec.?decod', r'draft.?model'],
            'sparse_attention': [r'sparse.?att', r'sparse.*attention'],
            'quantization': [r'int8|int4|awq|gptq|nf4', r'quantiz'],
        }

        for config_file in config_files:
            content = config_file.read_text(encoding='utf-8', errors='ignore')

            for opt_name, patterns in optimizations.items():
                for pattern in patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        self.findings['optimization_coverage'][opt_name].append(config_file.name)
                        break

        if self.findings['optimization_coverage']:
            print(f"  Optimization coverage:")
            for opt, jobs in self.findings['optimization_coverage'].items():
                print(f"    {YELLOW}•{RESET} {opt}: {', '.join(jobs)}")
        else:
            print(f"  {RED}⚠ No explicit optimization-specific CI found{RESET}")
            print(f"    Default config only? This means:")
            print(f"      • Optimizations may have different behavior than baseline")
            print(f"      • Multi-turn/agentic accuracy under optimization is untested")

    def check_non_default_configs(self, config_files):
        """Check if accuracy tests use non-default runtime configs"""
        print(f"\n{BLUE}=== Step 6: Configuration Diversity ==={RESET}")

        config_flags = [
            ('enable_prefix_caching', 'Prefix caching enabled'),
            ('kv_cache_precision', 'KV cache precision'),
            ('cache_eviction', 'Cache eviction'),
            ('enable.*quantiz', 'Quantization enabled'),
            ('max_num_batched_tokens', 'Batch config'),
        ]

        for config_file in config_files:
            content = config_file.read_text(encoding='utf-8', errors='ignore')

            found_configs = []
            for flag, desc in config_flags:
                if re.search(flag, content, re.IGNORECASE):
                    found_configs.append(desc)

            if found_configs:
                print(f"  {GREEN}✓{RESET} {config_file.name}:")
                for cfg in found_configs:
                    print(f"      └─ {cfg}")

    def verdict(self):
        """Final assessment of CI coverage"""
        print(f"\n{BLUE}=== VERDICT ==={RESET}\n")

        has_perf_ci = bool(self.findings['perf_jobs'])
        has_accuracy_ci = bool(self.findings['accuracy_jobs'])
        has_agentic_ci = bool(self.findings['agentic_tasks'])
        has_opt_coverage = bool(self.findings['optimization_coverage'])

        print(f"  Performance CI:                 {GREEN + 'YES' if has_perf_ci else RED + 'NO' + RESET}")
        print(f"  Accuracy/Eval CI:               {GREEN + 'YES' if has_accuracy_ci else RED + 'NO' + RESET}")
        print(f"  Agentic/Tool-call CI:           {GREEN + 'YES' if has_agentic_ci else RED + 'NO ← IMPORTANT' + RESET}")
        print(f"  Optimization-specific CI:       {GREEN + 'YES' if has_opt_coverage else YELLOW + 'PARTIAL/NONE ← CRITICAL' + RESET}")
        print()

        if not has_agentic_ci:
            print(f"  {RED}⚠ MAJOR FINDING:{RESET}")
            print(f"    This runtime has {RED}NO CI-guarded agentic evaluation{RESET}.")
            print(f"    Implications:")
            print(f"      • Tool-call accuracy under optimizations is {RED}UNTESTED{RESET}")
            print(f"      • Agent behavior in multi-turn loops may degrade silently")
            print(f"      • Prefix-cache/KV-quant/eviction safety for agentic unproven")

        if not has_opt_coverage:
            print(f"\n  {YELLOW}⚠ FINDING:{RESET}")
            print(f"    Accuracy tests only cover {YELLOW}DEFAULT configuration{RESET}.")
            print(f"    When optimizations (KV quantization, eviction, etc.) are enabled,")
            print(f"    whether accuracy holds is NOT guaranteed by CI.")

        print()

    def run(self):
        """Execute the full audit"""
        print(f"{BLUE}{'='*70}")
        print(f"CI Configuration Audit: {self.repo_path.name}")
        print(f"{'='*70}{RESET}")
        print(f"{YELLOW}Key principle: CI is truth. If it's not tested daily, assume it doesn't work.{RESET}")

        config_files = self.find_ci_configs()
        if not config_files:
            print(f"\n{YELLOW}No CI configurations found. Skipping analysis.{RESET}")
            return

        self.analyze_perf_jobs(config_files)
        self.analyze_accuracy_jobs(config_files)
        self.check_agentic_coverage(config_files)
        self.check_optimization_coverage(config_files)
        self.check_non_default_configs(config_files)
        self.verdict()

        # Export raw summary
        print(f"\n{BLUE}=== Raw Summary ==={RESET}")
        print(f"  Perf jobs: {len(self.findings['perf_jobs'])}")
        print(f"  Accuracy jobs: {len(self.findings['accuracy_jobs'])}")
        print(f"  Agentic tasks found: {len(self.findings['agentic_tasks'])}")
        print(f"  Optimization coverage: {len(self.findings['optimization_coverage'])} dimensions")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path_to_repo>")
        print(f"Example: {sys.argv[0]} ../vllm")
        sys.exit(1)

    audit = CIConfigAudit(sys.argv[1])
    audit.run()
