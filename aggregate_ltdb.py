"""
Aggregate LTDB evaluation results from parallel group logs.
Run after all ltdb_eval_g*.log files complete.
"""
import re
import glob
import os

LOG_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_log(path):
    """Extract per-record and total TP/FP/FN from a group log."""
    records = {}
    with open(path) as f:
        content = f.read()

    # Per-record lines: "  14046 (24h): TP=XXX  FP=XXX  FN=XXX  ref=XXX  Se=XX%  P+=XX%  F1=XX%"
    pattern = r'  (\d+) \((\d+)h\): TP=(\d+)\s+FP=(\d+)\s+FN=(\d+)\s+ref=(\d+)\s+Se=([\d.]+)%\s+P\+=([\d.]+)%\s+F1=([\d.]+)%'
    for m in re.finditer(pattern, content):
        rec_id, dur, tp, fp, fn, ref, se, pp, f1 = m.groups()
        records[rec_id] = {
            'dur_h': int(dur),
            'TP': int(tp), 'FP': int(fp), 'FN': int(fn),
            'ref': int(ref),
            'Se': float(se), 'Pp': float(pp), 'F1': float(f1),
        }
    return records

def main():
    logs = sorted(glob.glob(os.path.join(LOG_DIR, 'ltdb_eval_g*.log')))
    if not logs:
        print("No ltdb_eval_g*.log files found.")
        return

    all_records = {}
    incomplete_groups = []

    for log in logs:
        records = parse_log(log)
        if not records:
            incomplete_groups.append(log)
            print(f"[INCOMPLETE or no records] {os.path.basename(log)}")
        else:
            all_records.update(records)
            print(f"[OK] {os.path.basename(log)}: {list(records.keys())}")

    if incomplete_groups:
        print(f"\nWARNING: {len(incomplete_groups)} groups not yet complete. Partial results below.")

    if not all_records:
        print("No completed records found.")
        return

    print(f"\n{'='*55}")
    print(f"{'Record':<8} {'Dur':>4}  {'TP':>8} {'FP':>6} {'FN':>6} {'ref':>8}  {'Se%':>7} {'P+%':>7} {'F1%':>7}")
    print(f"{'-'*55}")

    total_TP = total_FP = total_FN = 0
    for rec_id in sorted(all_records):
        r = all_records[rec_id]
        total_TP += r['TP']
        total_FP += r['FP']
        total_FN += r['FN']
        print(f"{rec_id:<8} {r['dur_h']:>4}h  {r['TP']:>8} {r['FP']:>6} {r['FN']:>6} {r['ref']:>8}  "
              f"{r['Se']:>6.2f}% {r['Pp']:>6.2f}% {r['F1']:>6.2f}%")

    Se = total_TP / (total_TP + total_FN + 1e-9)
    Pp = total_TP / (total_TP + total_FP + 1e-9)
    F1 = 2 * Se * Pp / (Se + Pp + 1e-9)

    print(f"{'='*55}")
    print(f"{'TOTAL':<8} {'':>4}  {total_TP:>8} {total_FP:>6} {total_FN:>6}")
    print(f"\n===== LTDB Overall ({len(all_records)} records) =====")
    print(f"  TP={total_TP}  FP={total_FP}  FN={total_FN}")
    print(f"  Se  = {Se*100:.2f}%")
    print(f"  P+  = {Pp*100:.2f}%")
    print(f"  F1  = {F1*100:.2f}%")
    print(f"\n  MIT-BIH v2 (reference): F1=99.69%")
    print(f"  LTDB (this run):         F1={F1*100:.2f}%")
    print(f"  dF1 = {(F1-0.9969)*100:+.2f}pp vs MIT-BIH")

if __name__ == '__main__':
    main()
