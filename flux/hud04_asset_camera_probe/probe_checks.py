"""CPU-testable acceptance checks; all nonfinite input fails closed."""
import math

HEAD_ZERO_TRANSLATIONS=((-0.013,0.0,0.3882),(0.0,0.0,0.0395))

def check_head_zero(positions,quaternions,tolerance=1e-4):
    if len(positions)!=2 or len(quaternions)!=2:raise ValueError('Expected two head transforms')
    for position,quaternion,expected in zip(positions,quaternions,HEAD_ZERO_TRANSLATIONS):
        if len(position)!=3 or len(quaternion)!=4:raise ValueError('Malformed head transform')
        if not all(math.isfinite(x) for x in [*position,*quaternion]):
            raise ValueError('Nonfinite head transform')
        if max(abs(x-y) for x,y in zip(position,expected))>tolerance:
            raise ValueError('Imported head translation disagrees with fixed zero-pose URDF')
        # Both fixed joint origins have rpy0; q and -q are equivalent.
        error=min(max(abs(x-y) for x,y in zip(quaternion,[1,0,0,0])),
                  max(abs(x+y) for x,y in zip(quaternion,[1,0,0,0])))
        if error>tolerance:raise ValueError('Imported head orientation not fixed at model zero pose')
    return True

def require_finite_values(values,label):
    if not values or not all(math.isfinite(x) for x in values):raise ValueError('Nonfinite or empty '+label)
    return True