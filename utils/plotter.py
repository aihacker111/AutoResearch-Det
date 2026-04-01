import subprocess
import sys
from config import Config

class Plotter:
    @staticmethod
    def update_plot():
        script = Config.ROOT_DIR / "plot_progress.py"
        if not script.is_file(): return
        
        out_dir = Config.plot_output_dir()
        r = subprocess.run([sys.executable, str(script), "-i", str(Config.RESULTS_FILE), "-o", str(out_dir)], cwd=str(Config.ROOT_DIR), capture_output=True, text=True)
        if r.returncode != 0 and r.stderr:
            print(f"  [plot] warning: plot_progress.py exited {r.returncode}\n{r.stderr[:800]}")