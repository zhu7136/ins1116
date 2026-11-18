"""New verification target; no solver and no trial006 reexecution."""
from pathlib import Path
import verify_trial006_once as V
if __name__=='__main__':
    V.TRIAL=Path(__file__).parent/'WHOLEBODY_TRIAL_007'
    V.SHA='76fe75f0fe63b58fd4f4f3e71e535fced0dd0e6f4dc221fab84a46a64da3face'
    V.main()