Demo data for plot_progress.py
================================

1. Install matplotlib (use a venv if your system blocks pip, e.g. PEP 668):

   python3 -m venv .venv && .venv/bin/pip install matplotlib
   .venv/bin/python plot_progress.py -i examples/sample_results.tsv -o examples/output

   Or: pip install matplotlib

2. Generate charts from the bundled sample TSV:

   python plot_progress.py -i examples/sample_results.tsv -o examples/output

3. Output files (saved next to this README when you run step 2):

   examples/output/progress.png
   examples/output/progress_detail.png

The sample mimics results.tsv from autoresearch (commit, mAP, status, LLM description).
