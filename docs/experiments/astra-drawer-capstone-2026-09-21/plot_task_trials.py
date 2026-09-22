"""Present complete independent drawer traces without changing acceptance."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--label', required=True)
    a = p.parse_args()
    assert not a.output.exists(), 'Use a fresh presentation output directory'
    spec = json.loads(a.spec.read_text())
    data = []
    for seed in spec['seeds']:
        folder = a.evaluation / f'seed_{seed}'
        trace = folder / 'trace.jsonl'
        report_path = folder / 'trial_report.json'
        report = json.loads(report_path.read_text())
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        assert rows and report['seed'] == seed
        t = np.array([r['time_s'] for r in rows])
        q = np.array([r['q_m'] for r in rows])
        force = np.array([np.linalg.norm(r['payload_drawer_contact_force_n']) / spec['dt_s'] for r in rows])
        offset = np.array([r['payload_pose'][2] - r['drawer_pose'][2] for r in rows])
        penetration = np.array([max([0.] + [-c['separation'] for c in r['contacts']]) for r in rows])
        assert np.isfinite(np.concatenate([t, q, force, offset, penetration])).all()
        data.append((seed, report, trace, report_path, t, q, force, offset, penetration))
    a.output.mkdir(parents=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 6.8), sharex=True)
    records = []
    for seed, report, trace, report_path, t, q, force, offset, penetration in data:
        line, = axes[0, 0].plot(t, q * 100, lw=1.4, label=f'Seed {seed}: {report["status"]}')
        for ax, values in [(axes[0, 1], offset * 100), (axes[1, 0], force), (axes[1, 1], penetration * 1000)]:
            ax.plot(t, values, color=line.get_color(), lw=1.1)
        csv_path = a.output / f'seed_{seed}_measurements.csv'
        with csv_path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time_s', 'drawer_displacement_m', 'payload_relative_z_m', 'payload_contact_force_n_from_impulse', 'max_contact_penetration_m'])
            writer.writerows(zip(t, q, offset, force, penetration))
        compressed = a.output / f'seed_{seed}_trace.jsonl.gz'
        with compressed.open('wb') as f:
            with gzip.GzipFile(filename='', mode='wb', fileobj=f, mtime=0) as z:
                z.write(trace.read_bytes())
        records.append({'seed': seed, 'status': report['status'], 'checks': report['checks'], 'metrics': report['metrics'], 'trace_sha256': digest(trace), 'report_sha256': digest(report_path), 'compressed_trace_sha256': digest(compressed), 'measurements_sha256': digest(csv_path)})
    axes[0, 0].axhline(spec['opening_required_m'] * 100, color='#777', ls='--', lw=1)
    axes[1, 0].axhline(spec['payload']['mass_kg'] * abs(spec['gravity_m_s2'][1]), color='#777', ls='--', lw=1)
    axes[1, 1].axhline(spec['limits']['max_contact_penetration_m'] * 1000, color='#777', ls='--', lw=1)
    for ax, label in zip(axes.flat, ['Drawer displacement (cm)', 'Payload offset in drawer (cm)', 'Payload contact force (N)', 'Contact penetration (mm)']):
        ax.set_ylabel(label)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.15)
    for ax in axes[1]:
        ax.set_xlabel('Simulated time (s)')
    passes = sum(r['status'] == 'PASS' for r in records)
    fig.suptitle(f'{a.label}: {passes}/{len(records)} independent task trials passed', x=.07, ha='left', fontsize=14, fontweight='bold')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.51, .935), ncol=5, frameon=False, fontsize=8)
    fig.text(.07, .014, f'Protocol: {spec["protocol_id"]} | 0.5 kg free payload | bounded external force\nContact impulses divided by simulation step time. Full plotted ranges retained; no physical-hardware validation.', fontsize=8, color='#444')
    fig.tight_layout(rect=(.02, .075, 1, .92))
    fig.savefig(a.output / 'drawer_task.png', dpi=170)
    fig.savefig(a.output / 'drawer_task.svg')
    plt.close(fig)
    (a.output / 'summary.json').write_text(json.dumps({'label': a.label, 'protocol_id': spec['protocol_id'], 'spec_sha256': digest(a.spec), 'all_five_pass': len(records) == 5 and passes == 5, 'scope': 'Presentation of independent native physics traces; acceptance remains the evaluator result.', 'contact_units': 'The frozen force-named field contains N*s impulse; plotted N equals its vector norm divided by dt_s.', 'trials': records}, indent=2) + '\n')
    print(a.output / 'drawer_task.png')


if __name__ == '__main__':
    main()
