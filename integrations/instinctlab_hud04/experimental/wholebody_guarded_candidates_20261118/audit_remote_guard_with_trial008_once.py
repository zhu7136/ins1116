"""New final saved008 cross-check; does not replay old audit output."""
import audit_remote_guard_vs_new_candidates_once as A
if __name__=='__main__':
    A.CASES.append(('trial008',A.ROOT/'WHOLEBODY_TRIAL_008','d5cdb5382b5eb910b3273adab662de286e0452376ee9d134124d4ceddd2f30cf'))
    A.OUTPUT_NAME='FINAL_REMOTE_GUARD_WITH_TRIAL008_RECONCILIATION.json'
    A.AUDITED_TRIAL='WHOLEBODY_TRIAL_008'
    A.main()