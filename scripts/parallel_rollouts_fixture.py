import json
import time


def run_episode(item, args, videos, episode_path):
    if item.get("fixture") == "slow":
        time.sleep(0.1)
    if item.get("fixture") == "raise":
        raise RuntimeError("fixture worker failure")
    row = {**item, "case_id": item["id"], "status": "failure", "success": False,
           "worker_setting": args.fixture_setting}
    episode_path.write_text(json.dumps(row))
    return row
