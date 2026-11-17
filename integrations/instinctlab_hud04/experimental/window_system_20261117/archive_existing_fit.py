"""Back up the already generated fit; never reruns its solver."""
import hashlib,json
from pathlib import Path
from motion_archive import freeze_file

def main():
    root=Path(__file__).parent;dest=root/'EXISTING_FIT_ARCHIVE_RECEIPT.json'
    if dest.exists():raise FileExistsError('Receipt already exists')
    path=Path('/workspace/hud04-implementation-20261117/step1.joint_fit.npz')
    if hashlib.sha256(path.read_bytes()).hexdigest()!='48f8d43e058420e5ff89861d2336ead3317dc18223c36a7305503f3df54a66eb':raise ValueError('Existing fit checksum')
    receipt=freeze_file(path,Path('/workspace/hud04-backups/sha256'),role='joint_temporal_fit_not_reference',
        source_sha256='56158e8679580910ccff375963d025b138636acfe70f5b1f88381837209b2056',
        model_sha256='865a4344e2f838d939c2549ec0d9e771dc6c2eb745670d96bba78e162d2880d7')
    with dest.open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()