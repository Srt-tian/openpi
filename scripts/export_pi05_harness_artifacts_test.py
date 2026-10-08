from __future__ import annotations
import json,tempfile,unittest
from pathlib import Path
import export_pi05_harness_artifacts as target

class ArtifactExportTest(unittest.TestCase):
    def test_hashes_episode_and_retained_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"worker0";batch=root/"worker0_control";(batch/"episodes").mkdir(parents=True);(batch/"videos").mkdir()
            (root/"controller.json").write_text('{}');episode=batch/"episodes/000.json";episode.write_text(json.dumps({"video":{"written":True,"path":"000.mp4"}}));(batch/"videos/000.mp4").write_bytes(b"video")
            (batch/"summary.json").write_text(json.dumps({"cases":[{"episode":"episodes/000.json"}]}))
            output=Path(directory)/"receipt.json";target.export(root,output);value=json.loads(output.read_text());row=value["artifacts"][0]
            self.assertEqual((value["worker_output_name"],row["video_size"]),("worker0",5));self.assertEqual(len(row["episode_sha256"]),64)
            with self.assertRaises(FileExistsError):target.export(root,output)

if __name__=="__main__":unittest.main()
