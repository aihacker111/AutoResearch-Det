import subprocess
from datetime import datetime
from config import Config

class GitManager:
    @staticmethod
    def setup_branch() -> str:
        tag = datetime.now().strftime("%b%d").lower()
        branch = f"autoresearch/{tag}"
        r = subprocess.run(["git", "-C", str(Config.ROOT_DIR), "checkout", "-b", branch], capture_output=True)
        if r.returncode != 0:
            subprocess.run(["git", "-C", str(Config.ROOT_DIR), "checkout", branch], capture_output=True)
        return branch

    @staticmethod
    def commit(msg: str) -> str:
        subprocess.run(["git", "-C", str(Config.ROOT_DIR), "add", Config.GIT_TRAIN_PATH], capture_output=True)
        subprocess.run(["git", "-C", str(Config.ROOT_DIR), "commit", "-m", msg], capture_output=True)
        return subprocess.check_output(
            ["git", "-C", str(Config.ROOT_DIR), "rev-parse", "--short", "HEAD"], text=True
        ).strip()

    @staticmethod
    def rollback():
        subprocess.run(["git", "-C", str(Config.ROOT_DIR), "reset", "--hard", "HEAD~1"], capture_output=True)