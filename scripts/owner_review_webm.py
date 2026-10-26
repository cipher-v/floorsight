import json, shutil, subprocess
from pathlib import Path
from datetime import datetime, timezone
root=Path('owner-review')
shutil.copyfile(root/'index.html','eval/evidence/owner-review-h264-index.html')
shutil.copyfile(root/'review-manifest.json','eval/evidence/owner-review-h264-manifest.json')
subprocess.run(['ffmpeg','-nostdin','-n','-i',str(root/'media/tracking.mp4'),'-c:v','libvpx-vp9','-threads','4','-crf','32','-b:v','0',str(root/'media/tracking.webm')],check=True)
html=(root/'index.html').read_text().replace('src="media/tracking.mp4"','src="media/tracking.webm"')
html=html.replace('16 seconds, camera 2','VP9 compatibility copy; original H264 retained. 16 seconds, camera 2')
(root/'index.html').write_text(html)
manifest=json.loads((root/'review-manifest.json').read_text())
manifest['version']='opencv-owner-20261017-02'
manifest['updatedAt']=datetime.now(timezone.utc).isoformat()
manifest['media'].append({'path':'media/tracking.webm','type':'video/webm'})
(root/'review-manifest.json').write_text(json.dumps(manifest,indent=2))
