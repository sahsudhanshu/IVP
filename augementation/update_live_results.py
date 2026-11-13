"""Refresh local reports from the real Kaggle execution stream."""
import csv, json, os, re, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
ENV=os.environ.copy()
ENV["PYTHONPATH"]=str(PROJECT / ".kaggle_cli_deps")
ENV["KAGGLE_CONFIG_DIR"]=str(Path.home()/".kaggle")
ENV["PYTHONIOENCODING"]="utf-8"
KERNEL="sudhanshusah/nwucla-measured-augmentation"
logpath=ROOT/"augmentation_training.log"
records={}
result=None

def render():
    lines=["# Experiment results", "", "## Historical reference from slides 26–31", "", "The previous Inception-v3 experiment used 12 epochs. Slide 27 reports:", "", "| Protocol | Accuracy | Macro-F1 |", "|---|---:|---:|", "| Cross-subject, subject 1 | 54.9% | 0.444 |", "| Cross-view {2,3} → 1 | 52.6% | 0.494 |", "| Cross-view {1,3} → 2 | 46.9% | 0.471 |", "| Cross-view {1,2} → 3 | 36.6% | 0.295 |", "", "The historical test used 418 complete three-view groups and the historical run applied independently sampled frame flips.", "", "## Augmentation experiment — actual recorded progress", "", f"Last refreshed: {datetime.now(timezone.utc).isoformat(timespec='seconds')}", "", "Inception-v3; source views 1 and 2; four snippets; 12 epochs; seed 42. Training uses one flip and one brightness/contrast draw per clip. Development holds out subject 10 from source-view pairs; the final refit uses all source pairs and tests view 3 on the same 418 complete groups.", "", "| Phase | Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy | Validation macro-F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for (phase,epoch),r in sorted(records.items(),key=lambda item:(0 if item[0][0]=="development" else 1,item[0][1])):
        def fmt(k,pct=False):
            v=r.get(k)
            return "—" if v is None else (f"{100*v:.2f}%" if pct else f"{v:.4f}")
        lines.append(f"| {phase} | {epoch} | {fmt('train_loss')} | {fmt('train_accuracy',True)} | {fmt('validation_loss')} | {fmt('validation_accuracy',True)} | {fmt('validation_macro_f1')} |")
    if not records: lines.append("| Awaiting first recorded epoch | — | — | — | — | — | — |")
    lines.extend(["", "## Final cross-view evaluation", ""])
    if result:
        lines.extend([f"Accuracy: {100*result['accuracy']:.2f}%; macro-F1: {result['macro_f1']:.4f}; support: {result['support']}.", "", "Precision, recall, predictions and confusion matrix are available in the downloaded experiment output when present.", "", "The difference from 36.6% is descriptive: the runs use different random histories and augmentation, and the historical run monitored the test set each epoch. This is not a controlled causal estimate."])
    else:
        lines.extend(["**Pending:** final accuracy, precision, recall, macro-F1, confusion matrix and target-view comparison. The run has not logged a completed evaluation.", "", "Epoch validation metrics above are from a source-subject holdout and cannot be compared directly with the historical target-view score."])
    lines.extend(["", f"Run: https://www.kaggle.com/code/{KERNEL}", "", "Detailed time-stamped output: `augmentation_training.log`."])
    (ROOT/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

cmd=[sys.executable,"-m","kaggle","kernels","logs",KERNEL,"-f"]
with logpath.open("w",encoding="utf-8") as f:
    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",env=ENV,bufsize=1)
    render()
    for line in proc.stdout:
        f.write(line); f.flush()
        match=re.search(r"(\{.*\})",line)
        if match:
            try: data=json.loads(match.group(1))
            except json.JSONDecodeError: continue
            if "phase" in data and "epoch" in data: records[(data["phase"],data["epoch"])]=data; render()
            elif "accuracy" in data and "experiment" in data: result=data; render()
    status=proc.wait()
    f.write(f"\n[Log stream ended with code {status}; see Kaggle for run status.]\n")
    render()
    if result:
        completed=subprocess.run([sys.executable,"-m","kaggle","kernels","output",KERNEL,"-p",str(ROOT/"kaggle_output")],env=ENV,check=False)
        files=list((ROOT/"kaggle_output").rglob("test_predictions.csv"))
        if completed.returncode==0 and len(files)==1:
            with files[0].open(newline="",encoding="utf8") as f: rows=list(csv.DictReader(f))
            matrix=[[0]*10 for _ in range(10)]
            for row in rows: matrix[int(row["true"])][int(row["predicted"])]+=1
            precision=[]; recall=[]
            for i in range(10):
                tp=matrix[i][i]; predicted=sum(matrix[j][i] for j in range(10)); actual=sum(matrix[i])
                precision.append(tp/predicted if predicted else 0); recall.append(tp/actual if actual else 0)
            with (ROOT/"RESULTS.md").open("a",encoding="utf8") as f:
                f.write(f"\nMacro-precision: {sum(precision)/10:.4f}; macro-recall: {sum(recall)/10:.4f}. Calculated from saved test predictions across all 10 classes, using zero for undefined values.\n")
                f.write(f"\nChange from historical 36.6%: {100*result['accuracy']-36.6:+.2f} percentage points (descriptive only).\n")
                f.write("\nConfusion matrix and charts are in `kaggle_output/augmentation_results/`; rows are true and columns predicted.\n")
