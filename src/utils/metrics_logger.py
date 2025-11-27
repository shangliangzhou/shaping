import csv, os, time
from pathlib import Path

COLUMNS = ["timestamp","env","exp","seed","mode","episode","formula",
           "success","ep_len","return_disc","return_undisc",
           "cost","collisions","ldba_accept_visits",
           "shape_sum","shape_prog","noise_p_miss","noise_p_false"]

class SummaryLogger:
    def __init__(self, out_csv):
        self.out_csv = Path(out_csv)
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        if not self.out_csv.exists():
            with self.out_csv.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=COLUMNS).writeheader()

    def log_episode(self, **row):
        row["timestamp"] = int(time.time())
        # 缺省字段补空，保证列对齐
        full = {k: row.get(k, "") for k in COLUMNS}
        with self.out_csv.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writerow(full)
