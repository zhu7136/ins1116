"""Copy documented per-contact material IDs inside an existing callback.

No stage access, SDK import, subscription, material edit or simulation action.
107.3 ContactData.material0/1 are uint64 IDs decoded via intToSdfPath.
Materials need not be in /World. Missing/zero/invalid IDs stay unresolved.
"""
import numbers
from pathlib import PurePosixPath


def material_prim_path(value):
    """Conservative canonical absolute prim-path subset, not a USD resolver."""
    if (not isinstance(value, str) or len(value) > 2048
            or not value.startswith('/') or value == '/' or value.endswith('/')
            or str(PurePosixPath(value)) != value
            or any(c in value for c in '*?[]{}.:\\')
            or any(c.isspace() for c in value)
            or any(c in value for c in ('\x00', '\n', '\r'))):
        raise ValueError('Noncanonical or unsupported material prim path')
    return value


def _uint(value, bits):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or not 0 <= value < 2**bits:
        raise ValueError('Expected uint' + str(bits))
    return int(value)


def _material(point, field, decode_path):
    record = dict(status='UNRESOLVED',native_uint64_id=None,decoded_material_prim_path=None)
    try:
        value = getattr(point, field)
    except AttributeError:
        record['status'] = 'FIELD_UNAVAILABLE'
        return record
    except Exception as exc:
        record.update(status='FIELD_READ_ERROR', error=type(exc).__name__)
        return record
    try:
        value = _uint(value, 64)
        record['native_uint64_id'] = value
    except Exception as exc:
        record.update(status='INVALID_ID', error=type(exc).__name__)
        return record
    if value == 0:
        # Documentation does NOT establish zero as a particular default material.
        record['status'] = 'ZERO_ID_UNRESOLVED'
        return record
    try:
        record['decoded_material_prim_path'] = material_prim_path(str(decode_path(value)))
        record['status'] = 'ID_DECODED_NOT_RUNTIME_MATERIAL_VERIFIED'
    except Exception as exc:
        record.update(status='DECODE_ERROR_OR_UNSUPPORTED_PATH', error=type(exc).__name__)
    return record


def _face(point, field):
    try:
        value = getattr(point, field)
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError('Expected integer face field')
        value = int(value)
        if value.bit_length() > 64:
            raise ValueError('Face field exceeds serialization bit budget')
        # Preserve negative/all-ones values without asserting sentinel semantics.
        return dict(status='RAW_INTEGER_NOT_COOKED_FACE_BINDING',value=value)
    except AttributeError:
        return dict(status='FIELD_UNAVAILABLE',value=None)
    except Exception as exc:
        return dict(status='FIELD_INVALID_OR_READ_ERROR',value=None,error=type(exc).__name__)


def copy_contact_material_links(point, decode_path):
    """Called during native point lifetime; do not infer values from ID equality.

    Side0/side1 preserve the containing native event's actor/collider ordering.
    No reversed-pair canonicalization, face-index reinterpretation or slot join.
    """
    if not callable(decode_path):
        raise ValueError('Explicit runtime decoder required')
    return dict(material0=_material(point,'material0',decode_path),
                material1=_material(point,'material1',decode_path),
                face_index0=_face(point,'face_index0'),
                face_index1=_face(point,'face_index1'),
                runtime_material_values_verified=False,
                material_combine_modes_verified=False,
                effective_pair_friction=None)