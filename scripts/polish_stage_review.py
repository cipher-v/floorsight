import os,json,shutil
from pathlib import Path
from datetime import datetime,timezone
assert os.environ.get('FLOORSIGHT_GCP_LEASE_CONFIRMED')=='yes'
r=Path('owner-review')
shutil.copytree(r,Path('eval/reviews/owner-review-v03-preserved-20261017'))
p=r/'index.html';s=p.read_text();s=s.replace('</style>','a{color:#8fd6ff;text-underline-offset:3px}</style>');p.write_text(s)
p=r/'review-manifest.json';m=json.loads(p.read_text());m['version']='opencv-owner-20261017-04';m['updatedAt']=datetime.now(timezone.utc).isoformat();p.write_text(json.dumps(m,indent=2))
