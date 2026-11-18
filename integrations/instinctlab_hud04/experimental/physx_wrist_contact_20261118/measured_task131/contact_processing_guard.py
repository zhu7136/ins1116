"""CPU-tested proposed initialization repair. NOT executed in task131 or any GPU.

IsaacLab v2.3.2 SimulationContext sets disableContactProcessing=True;
ContactSensor.__init__ sets False. The raw-view probe omitted that transition.
This mismatch is a hypothesis, NOT an established cause of zero GPU readings.
Positive controls and repeated box state observation remain mandatory.
"""
def enable_contact_processing(settings):
    key='/physics/disableContactProcessing'
    before=settings.get_as_bool(key)
    settings.set_bool(key,False)
    after=settings.get_as_bool(key)
    if after is not False:raise RuntimeError('Contact processing setting did not read back False')
    return dict(setting=key,before=before,after=after,gpu_effect_verified=False,
                positive_control_verified=False,accepted_reference=False,training_ready=False)

def measured_classification(verdict):
    if verdict.get('independent_box_contact_positive') is not True:
        return 'INCONCLUSIVE_POSITIVE_CONTROL_FAILED'
    sides=verdict.get('controlled_wrist_body_pair_filter_supported',{})
    if sides=={'left':True,'right':True}:return 'LIMITED_WRIST_PAIR_SUPPORT_ONLY'
    return 'INCONCLUSIVE_WRIST_TREATMENT_CONTROL_NOT_SATISFIED'