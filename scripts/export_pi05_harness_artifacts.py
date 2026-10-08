#!/usr/bin/env python3
"""Export small hashes binding worker JSON records to videos retained in place."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

SCHEMA="pi05_harness_artifact_inventory.v1"

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def export(worker:Path,output:Path)->Path:
    worker,output=worker.resolve(),output.resolve()
    if output.exists():raise FileExistsError("artifact inventory is create-only")
    controller=worker/"controller.json"
    if not controller.is_file():raise ValueError("worker controller is missing")
    artifacts=[]
    for summary_path in sorted(worker.glob("*/summary.json")):
        summary=json.loads(summary_path.read_text());batch=summary_path.parent
        for row in summary.get("cases",[]):
            episode=(batch/row["episode"]).resolve()
            if not episode.is_relative_to(batch.resolve()) or not episode.is_file():
                raise ValueError("episode path escapes or is missing")
            value=json.loads(episode.read_text());video=value.get("video",{});name=video.get("path")
            if video.get("written") is not True or not isinstance(name,str) or Path(name).name!=name:
                raise ValueError("episode has no valid written-video receipt")
            video_path=batch/"videos"/name
            if not video_path.is_file() or video_path.stat().st_size<=0:
                raise ValueError("retained video is missing or empty")
            artifacts.append({"episode_path":episode.relative_to(worker).as_posix(),
                "episode_sha256":sha256(episode),"video_path":video_path.relative_to(worker).as_posix(),
                "video_sha256":sha256(video_path),"video_size":video_path.stat().st_size})
    document={"schema":SCHEMA,"worker_output_name":worker.name,
              "controller_sha256":sha256(controller),"artifacts":artifacts}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(document,indent=2,sort_keys=True)+"\n")
    return output

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--worker-output",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();print(export(a.worker_output,a.output))
