"""Independent new008 read-only verification targets. No old launches."""
from pathlib import Path
ROOT=Path(__file__).parent
SHA='d5cdb5382b5eb910b3273adab662de286e0452376ee9d134124d4ceddd2f30cf'
if __name__=='__main__':
    import verify_trial006_once as V
    V.TRIAL=ROOT/'WHOLEBODY_TRIAL_008';V.SHA=SHA;V.main()
    import audit_trial007_wholebody_once as A
    A.TRIAL=ROOT/'WHOLEBODY_TRIAL_008';A.SHA=SHA;A.main()