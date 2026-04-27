"""ITSM agent tool functions — see itsm_params.py for flow sequences."""

import copy
import hashlib
import json as _json
import re
from datetime import datetime as _dt
from datetime import timedelta

from pydantic import ValidationError

from eva.assistant.tools.itsm_params import (
    AddAffectedUserParams,
    AssignSLATierParams,
    AttachDiagnosticLogParams,
    AttemptAccountUnlockParams,
    AttemptPasswordResetParams,
    CheckDeskAvailabilityParams,
    CheckErgonomicAssessmentParams,
    CheckExistingAccountsParams,
    CheckExistingOutageParams,
    CheckHardwareEntitlementParams,
    CheckParkingAvailabilityParams,
    CheckRoleChangeAuthorizedParams,
    CheckRoomAvailabilityParams,
    CreateIncidentTicketParams,
    EscalateApprovalParams,
    GetApplicationDetailsParams,
    GetAssetRecordParams,
    GetEmployeeAssetsParams,
    GetEmployeeLicensesParams,
    GetEmployeeRecordParams,
    GetGroupDetailsParams,
    GetGroupMembershipsParams,
    GetLicenseCatalogItemParams,
    GetOffboardingRecordParams,
    GetPermissionTemplatesParams,
    GetRequestStatusParams,
    GetTroubleshootingGuideParams,
    InitiateAssetRecoveryParams,
    InitiateAssetReturnParams,
    InitiateOtpAuthParams,
    InitiateRemoteWipeParams,
    LinkKnownErrorParams,
    LookupNewHireParams,
    MarkResolvedParams,
    ProvisionNewAccountParams,
    ReportSecurityIncidentParams,
    RouteApprovalWorkflowParams,
    ScheduleAccessReviewParams,
    ScheduleFieldDispatchParams,
    SendCalendarInviteParams,
    SubmitAccessRemovalParams,
    SubmitAccessRequestParams,
    SubmitDeskAssignmentParams,
    SubmitEquipmentRequestParams,
    SubmitGroupMembershipChangeParams,
    SubmitHardwareRequestParams,
    SubmitLicenseRenewalParams,
    SubmitLicenseRequestParams,
    SubmitMfaResetParams,
    SubmitParkingAssignmentParams,
    SubmitPermissionChangeParams,
    SubmitRoomBookingParams,
    SubmitWaitlistParams,
    TransferToAgentParams,
    ValidateCostCenterParams,
    VerifyCostCenterBudgetParams,
    VerifyEmployeeAuthParams,
    VerifyManagerAuthParams,
    VerifyOtpAuthParams,
    validation_error_response,
)

_DEFAULT_CURRENT_DATE = "2026-08-12"


# Content-hash IDs: every counter-derived identifier (REQ-/INC/CASE-/SEC-/CAL-)
# is now derived from the SHA-1 of the record's content, so identical actions
# produce identical IDs regardless of execution order. The strip regex matches
# both legacy counter form (REQ-FAC-048271) and new content-hash form
# (REQ-FAC-a3f8e1) so a record's hash doesn't depend on whether cross-references
# inside it have already been migrated.
_ID_STRIP_RE = re.compile(r"^(REQ-[A-Z]+-|INC|CASE-[A-Z]+-|SEC-|CAL-)[A-Za-z0-9]+$")

# Fields that tools mutate AFTER the record is created. The ID is computed at
# creation time; later mutations would shift the hash if these were included.
# Excluding them here keeps the ID stable across a record's lifecycle and lets
# the migration (which sees the final post-mutation state) produce the same
# hash the tool produced at creation.
_POST_CREATION_MUTABLE_FIELDS = frozenset(
    {
        # generic lifecycle
        "status",
        "resolved_date",
        # tickets — assign_sla_tier
        "sla_tier",
        "sla_response_hours",
        "sla_resolution_hours",
        # tickets — link_known_error
        "linked_known_error_id",
        # tickets — schedule_field_dispatch
        "dispatch_id",
        "dispatch_date",
        "dispatch_window",
        # tickets — attach_diagnostic_log
        "diagnostic_ref_code",
        "diagnostic_attached",
        # requests — route_approval_workflow
        "approval_routed_to",
        "approval_sla_deadline",
        # requests — escalate_approval
        "escalated_to",
        "escalation_sla_deadline",
        # security cases — initiate_remote_wipe
        "remote_wipe_id",
        # pending_role_change — schedule_access_review
        "access_review_id",
        "review_date",
    }
)


def _strip_ids_for_hash(obj):
    if isinstance(obj, dict):
        return {k: _strip_ids_for_hash(v) for k, v in obj.items() if k not in _POST_CREATION_MUTABLE_FIELDS}
    if isinstance(obj, list):
        return [_strip_ids_for_hash(item) for item in obj]
    if isinstance(obj, str) and _ID_STRIP_RE.match(obj):
        return None
    return obj


def _content_hash(record):
    stripped = _strip_ids_for_hash(record)
    serialized = _json.dumps(stripped, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(serialized.encode()).hexdigest()[:12]


def _make_request_id(record, cat):
    return f"REQ-{cat}-{_content_hash(record)}"


def _make_case_id(record, cat):
    return f"CASE-{cat}-{_content_hash(record)}"


def _make_ticket_number(record):
    return f"INC{_content_hash(record)}"


def _make_security_case_id(record):
    return f"SEC-{_content_hash(record)}"


def _enf(eid):
    return {"status": "error", "error_type": "not_found", "message": f"Employee {eid} not found"}


def _ar(t="employee_auth"):
    return {"status": "error", "error_type": "authentication_required", "message": f"Authentication ({t}) required"}


def _ok(db, k):
    return db.get("session", {}).get(k) is True


def _current_date(db):
    return db.get("_current_date", _DEFAULT_CURRENT_DATE)


def _parse_date(s):
    return _dt.strptime(s, "%Y-%m-%d")


def _catalog_match(name, entry):
    """Case-insensitive exact match against entry.name and entry.name_aliases."""
    target = name.strip().lower()
    if entry.get("name", "").strip().lower() == target:
        return True
    return any(target == a.strip().lower() for a in entry.get("name_aliases", []))


_BUILDING_CODE_RE = re.compile(r"^BLD\d{1,2}$")
_ZONE_CODE_RE = re.compile(r"^PZ[A-Z]$")


def _resolve_facility(kind, value, db):
    """X5-R2: resolve a caller-provided building or zone reference to its canonical code. kind ∈ {'building','zone'}. Returns (code, None) on hit or (None, error_dict) on miss."""
    if value is None:
        return None, None
    if kind == "building":
        table = db.get("facilities", {}).get("buildings", {})
        code_re = _BUILDING_CODE_RE
        label = "Building"
    elif kind == "zone":
        table = db.get("facilities", {}).get("zones", {})
        code_re = _ZONE_CODE_RE
        label = "Parking zone"
    else:
        return None, {
            "status": "error",
            "error_type": "invalid_facility_kind",
            "message": f"Unknown facility kind: {kind}",
        }
    # Direct code match: return as-is if it parses, even if table is empty
    if code_re.match(value):
        return value, None
    # Alias / name match (case-insensitive exact)
    target = value.strip().lower()
    for code, entry in table.items():
        if entry.get("name", "").strip().lower() == target:
            return code, None
        if any(target == a.strip().lower() for a in entry.get("name_aliases", [])):
            return code, None
    known = [entry.get("name") for entry in table.values() if entry.get("name")]
    return None, {
        "status": "error",
        "error_type": "name_not_found",
        "message": f"{label} '{value}' not recognized. Provide a code or a known name. Known names: {', '.join(known) if known else '(none on file)'}.",
    }


def _group_eligible(group, emp):
    """X4: empty list on a dimension = unrestricted; otherwise employee's dept/role must be in the list."""
    depts = group.get("eligible_departments") or []
    roles = group.get("eligible_roles") or []
    if depts and emp.get("department_code") not in depts:
        return False
    if roles and emp.get("role_code") not in roles:
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# AUTH
# ═══════════════════════════════════════════════════════════════════════════════


def verify_employee_auth(params, db, call_index):
    try:
        p = VerifyEmployeeAuthParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, VerifyEmployeeAuthParams)
    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _enf(p.employee_id)
    if emp.get("phone_last_four") != p.phone_last_four:
        return {"status": "error", "error_type": "authentication_failed", "message": "Phone number does not match"}
    db.setdefault("session", {})["employee_auth"] = True
    db["session"]["authenticated_employee_id"] = p.employee_id
    return {
        "status": "success",
        "authenticated": True,
        "employee_id": p.employee_id,
        "first_name": emp.get("first_name"),
        "last_name": emp.get("last_name"),
        "message": f"Employee {p.employee_id} authenticated",
    }


def initiate_otp_auth(params, db, call_index):
    try:
        p = InitiateOtpAuthParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, InitiateOtpAuthParams)
    if not _ok(db, "employee_auth") and not _ok(db, "manager_auth"):
        return _ar()
    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _enf(p.employee_id)
    db.setdefault("session", {}).update({"otp_employee_id": p.employee_id, "otp_issued": True})
    return {
        "status": "success",
        "phone_last_four": emp.get("phone_last_four"),
        "message": f"OTP sent to ***{emp.get('phone_last_four')}",
    }


def verify_otp_auth(params, db, call_index):
    try:
        p = VerifyOtpAuthParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, VerifyOtpAuthParams)
    if not db.get("session", {}).get("otp_issued"):
        return {"status": "error", "error_type": "otp_not_initiated", "message": "Call initiate_otp_auth first"}
    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _enf(p.employee_id)
    if emp.get("otp_code") != p.otp_code:
        return {"status": "error", "error_type": "authentication_failed", "message": "OTP does not match"}
    db["session"]["otp_auth"] = True
    db["session"].pop("otp_employee_id", None)
    db["session"].pop("otp_issued", None)
    return {"status": "success", "authenticated": True, "employee_id": p.employee_id, "message": "OTP verified"}


def verify_manager_auth(params, db, call_index):
    """Performs standard employee verification AND manager authorization in one call."""
    try:
        p = VerifyManagerAuthParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, VerifyManagerAuthParams)
    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _enf(p.employee_id)
    if emp.get("phone_last_four") != p.phone_last_four:
        return {"status": "error", "error_type": "authentication_failed", "message": "Phone number does not match"}
    if emp.get("manager_auth_code") != p.manager_auth_code:
        return {"status": "error", "error_type": "authentication_failed", "message": "Manager auth code does not match"}
    if not emp.get("is_manager"):
        return {"status": "error", "error_type": "not_authorized", "message": f"{p.employee_id} is not a manager"}
    db.setdefault("session", {})["employee_auth"] = True
    db["session"]["authenticated_employee_id"] = p.employee_id
    db["session"]["manager_auth"] = True
    db["session"]["manager_employee_id"] = p.employee_id
    return {
        "status": "success",
        "confirmed": True,
        "employee_id": p.employee_id,
        "first_name": emp.get("first_name"),
        "last_name": emp.get("last_name"),
        "department_code": emp.get("department_code"),
        "direct_reports": emp.get("direct_reports", []),
        "message": "Standard + manager auth confirmed",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED LOOKUPS
# ═══════════════════════════════════════════════════════════════════════════════


def get_employee_record(params, db, call_index):
    try:
        p = GetEmployeeRecordParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, GetEmployeeRecordParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _enf(p.employee_id)
    fields = [
        "employee_id",
        "first_name",
        "last_name",
        "department_code",
        "role_code",
        "hire_date",
        "employment_status",
        "building_code",
        "floor_code",
        "manager_employee_id",
    ]
    return {"status": "success", "employee": {k: emp[k] for k in fields if k in emp}}


def get_asset_record(params, db, call_index):
    try:
        p = GetAssetRecordParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, GetAssetRecordParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    asset = db.get("assets", {}).get(p.asset_tag)
    if not asset:
        return {"status": "error", "error_type": "not_found", "message": f"Asset {p.asset_tag} not found"}
    return {"status": "success", "asset": copy.deepcopy(asset)}


def get_employee_assets(params, db, call_index):
    try:
        p = GetEmployeeAssetsParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, GetEmployeeAssetsParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    assets = [copy.deepcopy(a) for a in db.get("assets", {}).values() if a.get("assigned_employee_id") == p.employee_id]
    return {"status": "success", "employee_id": p.employee_id, "assets": assets, "message": f"{len(assets)} asset(s)"}


# ═══════════════════════════════════════════════════════════════════════════════
# FLOW 1: LOGIN — troubleshooting + direct resolution
# ═══════════════════════════════════════════════════════════════════════════════


def get_troubleshooting_guide(params, db, call_index):
    try:
        p = GetTroubleshootingGuideParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, GetTroubleshootingGuideParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    guide = db.get("troubleshooting_guides", {}).get(p.issue_category)
    if not guide:
        return {"status": "error", "error_type": "guide_not_found", "message": f"No guide for {p.issue_category}"}
    return {
        "status": "success",
        "issue_category": p.issue_category,
        "steps": copy.deepcopy(guide["steps"]),
        "message": f"{len(guide['steps'])} steps",
    }


def attempt_account_unlock(params, db, call_index):
    try:
        p = AttemptAccountUnlockParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, AttemptAccountUnlockParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _enf(p.employee_id)
    acct_status = emp.get("account_status", {}).get(p.target_system)
    if not acct_status:
        return {
            "status": "error",
            "error_type": "no_account",
            "message": f"No account found on {p.target_system} for {p.employee_id}",
        }
    if acct_status.get("locked") is not True:
        return {"status": "error", "error_type": "not_locked", "message": f"Account on {p.target_system} is not locked"}
    # C11: refuse to unlock when the lock is under a security hold
    reason = acct_status.get("lock_reason")
    if reason in ("security_investigation", "admin_hold"):
        return {
            "status": "error",
            "error_type": "cannot_unlock_security_hold",
            "lock_reason": reason,
            "message": f"Account on {p.target_system} is under {reason.replace('_', ' ')}. This cannot be unlocked from the service desk. Offer to transfer the caller to a live agent; security will contact them separately.",
        }
    acct_status["locked"] = False
    acct_status["lock_reason"] = None
    return {
        "status": "success",
        "employee_id": p.employee_id,
        "target_system": p.target_system,
        "resolved": True,
        "message": f"Account unlocked on {p.target_system}",
    }


def attempt_password_reset(params, db, call_index):
    try:
        p = AttemptPasswordResetParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, AttemptPasswordResetParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _enf(p.employee_id)
    acct_status = emp.get("account_status", {}).get(p.target_system)
    if not acct_status:
        return {
            "status": "error",
            "error_type": "no_account",
            "message": f"No account on {p.target_system} for {p.employee_id}",
        }
    if acct_status.get("requires_in_person_reset"):
        return {
            "status": "error",
            "error_type": "in_person_required",
            "message": f"Password reset for {p.target_system} requires in-person verification at the IT security office",
        }
    acct_status["password_expired"] = False
    acct_status["temp_password_issued"] = True
    return {
        "status": "success",
        "employee_id": p.employee_id,
        "target_system": p.target_system,
        "resolved": True,
        "message": "Temporary password sent to phone on file. Caller must change on first login.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# INCIDENT CREATION + SLA (Flows 1-4)
# ═══════════════════════════════════════════════════════════════════════════════


def create_incident_ticket(params, db, call_index):
    try:
        p = CreateIncidentTicketParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, CreateIncidentTicketParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    if p.category in ("login_issue", "network_connectivity") and not p.troubleshooting_completed:
        return {
            "status": "error",
            "error_type": "troubleshooting_required",
            "message": f"Complete troubleshooting before creating {p.category} ticket",
        }
    record = {
        "ticket_number": None,
        "employee_id": p.employee_id,
        "category": p.category,
        "urgency": p.urgency,
        "affected_system": p.affected_system,
        "troubleshooting_completed": p.troubleshooting_completed,
        "status": "open",
        "sla_tier": None,
        "created_date": _current_date(db),
    }
    tn = _make_ticket_number(record)
    record["ticket_number"] = tn
    db.setdefault("tickets", {})[tn] = record
    return {
        "status": "success",
        "ticket_number": tn,
        "category": p.category,
        "urgency": p.urgency,
        "message": f"Ticket created: {tn}",
    }


def assign_sla_tier(params, db, call_index):
    try:
        p = AssignSLATierParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, AssignSLATierParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    ticket = db.get("tickets", {}).get(p.ticket_number)
    if not ticket:
        return {"status": "error", "error_type": "not_found", "message": f"Ticket {p.ticket_number} not found"}
    # Enforce urgency→tier mapping: high→tier_1, medium→tier_2, low→tier_3
    expected_tier = {"high": "tier_1", "medium": "tier_2", "low": "tier_3"}.get(ticket.get("urgency"))
    if expected_tier and p.sla_tier != expected_tier:
        return {
            "status": "error",
            "error_type": "tier_urgency_mismatch",
            "message": f"Ticket urgency is '{ticket['urgency']}' which requires {expected_tier}, not {p.sla_tier}",
        }
    sla = {"tier_1": {"resp": 1, "res": 4}, "tier_2": {"resp": 4, "res": 8}, "tier_3": {"resp": 8, "res": 24}}[
        p.sla_tier
    ]
    ticket.update({"sla_tier": p.sla_tier, "sla_response_hours": sla["resp"], "sla_resolution_hours": sla["res"]})
    return {
        "status": "success",
        "ticket_number": p.ticket_number,
        "sla_tier": p.sla_tier,
        "response_target": f"{sla['resp']}h",
        "resolution_target": f"{sla['res']}h",
        "message": f"SLA {p.sla_tier}: respond {sla['resp']}h, resolve {sla['res']}h",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FLOW 2: OUTAGE
# ═══════════════════════════════════════════════════════════════════════════════


def check_existing_outage(params, db, call_index):
    try:
        p = CheckExistingOutageParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, CheckExistingOutageParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    outage = db.get("active_outages", {}).get(p.service_name)
    if not outage:
        return {
            "status": "success",
            "existing_outage": False,
            "service_name": p.service_name,
            "message": "No active outage found.",
        }
    return {
        "status": "success",
        "existing_outage": True,
        "ticket_number": outage["ticket_number"],
        "service_name": p.service_name,
        "affected_user_count": outage.get("affected_user_count", 0),
        "message": f"Active outage: {outage['ticket_number']}",
    }


def add_affected_user(params, db, call_index):
    try:
        p = AddAffectedUserParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, AddAffectedUserParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    outage = next((o for o in db.get("active_outages", {}).values() if o.get("ticket_number") == p.ticket_number), None)
    if not outage:
        return {"status": "error", "error_type": "ticket_not_found", "message": f"No active outage {p.ticket_number}"}
    affected = outage.setdefault("affected_users", [])
    if p.employee_id not in affected:
        affected.append(p.employee_id)
    outage["affected_user_count"] = len(affected)
    return {
        "status": "success",
        "ticket_number": p.ticket_number,
        "employee_id": p.employee_id,
        "total_affected": len(affected),
        "message": f"Added to {p.ticket_number}",
    }


def link_known_error(params, db, call_index):
    try:
        p = LinkKnownErrorParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, LinkKnownErrorParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    ticket = db.get("tickets", {}).get(p.ticket_number)
    if not ticket:
        return {"status": "error", "error_type": "not_found", "message": f"Ticket {p.ticket_number} not found"}
    ke = db.get("known_errors", {}).get(p.service_name)
    if not ke:
        return {
            "status": "success",
            "linked": False,
            "ticket_number": p.ticket_number,
            "message": "No known error found.",
        }
    ticket["linked_known_error_id"] = ke["known_error_id"]
    return {
        "status": "success",
        "linked": True,
        "ticket_number": p.ticket_number,
        "known_error_id": ke["known_error_id"],
        "workaround": ke.get("workaround"),
        "message": f"Linked {ke['known_error_id']}",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FLOW 3: HARDWARE MALFUNCTION — field dispatch
# ═══════════════════════════════════════════════════════════════════════════════


def schedule_field_dispatch(params, db, call_index):
    try:
        p = ScheduleFieldDispatchParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, ScheduleFieldDispatchParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    ticket = db.get("tickets", {}).get(p.ticket_number)
    if not ticket:
        return {"status": "error", "error_type": "not_found", "message": f"Ticket {p.ticket_number} not found"}
    avail = db.get("field_dispatch_availability", {})
    window = avail.get(p.preferred_date, {}).get(p.time_window)
    if not window or not window.get("available"):
        alts = [
            {"date": d, "time_window": w}
            for d, ws in sorted(avail.items())
            if d >= p.preferred_date
            for w, info in ws.items()
            if info.get("available")
        ][:3]
        return {
            "status": "error",
            "error_type": "no_availability",
            "message": f"No tech on {p.preferred_date} {p.time_window}",
            "alternative_slots": alts,
        }
    dsp = f"DSP-{p.ticket_number[-7:]}"
    window["available"] = False
    ticket.update({"dispatch_id": dsp, "dispatch_date": p.preferred_date, "dispatch_window": p.time_window})
    return {
        "status": "success",
        "dispatch_id": dsp,
        "date": p.preferred_date,
        "time_window": p.time_window,
        "message": f"Dispatch {dsp} confirmed",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FLOW 4: NETWORK — diagnostic log
# ═══════════════════════════════════════════════════════════════════════════════


def attach_diagnostic_log(params, db, call_index):
    try:
        p = AttachDiagnosticLogParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, AttachDiagnosticLogParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    ticket = db.get("tickets", {}).get(p.ticket_number)
    if not ticket:
        return {"status": "error", "error_type": "not_found", "message": f"Ticket {p.ticket_number} not found"}
    ticket.update({"diagnostic_ref_code": p.diagnostic_ref_code, "diagnostic_attached": True})
    return {
        "status": "success",
        "ticket_number": p.ticket_number,
        "diagnostic_ref_code": p.diagnostic_ref_code,
        "message": "Diagnostic attached",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FLOWS 5-6, 19: HARDWARE REQUESTS
# ═══════════════════════════════════════════════════════════════════════════════


def check_hardware_entitlement(params, db, call_index):
    try:
        p = CheckHardwareEntitlementParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, CheckHardwareEntitlementParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _enf(p.employee_id)
    ent = emp.get("hardware_entitlements", {}).get(p.request_type)
    if not ent:
        return {"status": "error", "error_type": "no_entitlement", "message": f"No entitlement for {p.request_type}"}
    if ent.get("pending_request"):
        return {
            "status": "error",
            "error_type": "request_already_pending",
            "message": f"Pending: {ent.get('pending_request_id')}",
        }

    # Compute device age from asset record (D2)
    device_age_months = None
    current_asset_tag = ent.get("current_asset_tag")
    if current_asset_tag:
        asset = db.get("assets", {}).get(current_asset_tag)
        if asset and asset.get("purchase_date"):
            try:
                purchase = _parse_date(asset["purchase_date"])
                now = _parse_date(_current_date(db))
                device_age_months = (now.year - purchase.year) * 12 + (now.month - purchase.month)
                if now.day < purchase.day:
                    device_age_months -= 1
            except Exception:
                pass

    # Eligibility windows
    if p.request_type == "laptop_replacement":
        if device_age_months is not None and device_age_months < 36:
            return {
                "status": "error",
                "error_type": "device_too_new",
                "message": f"Laptop is {device_age_months} months old; minimum 36 months for standard replacement. File a security incident if lost/stolen/damaged.",
            }
    elif p.request_type == "monitor_bundle":
        if device_age_months is not None and device_age_months < 12:
            return {
                "status": "error",
                "error_type": "replaced_too_recently",
                "message": f"Monitor is {device_age_months} months old; minimum 12 months between replacements.",
            }

    asset_details = None
    if current_asset_tag:
        asset = db.get("assets", {}).get(current_asset_tag)
        if asset:
            asset_details = {
                "asset_tag": current_asset_tag,
                "os": asset.get("os"),
                "size": asset.get("size"),
                "model": asset.get("model"),
            }

    return {
        "status": "success",
        "eligible": True,
        "employee_id": p.employee_id,
        "request_type": p.request_type,
        "current_asset_tag": current_asset_tag,
        "device_age_months": device_age_months,
        "current_asset": asset_details,
        "message": "Eligible",
    }


def submit_hardware_request(params, db, call_index):
    try:
        p = SubmitHardwareRequestParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, SubmitHardwareRequestParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    bcode, err = _resolve_facility("building", p.delivery_building, db)
    if err:
        return err
    record = {
        "request_id": None,
        "employee_id": p.employee_id,
        "request_type": p.request_type,
        "justification": p.justification,
        "current_asset_tag": p.current_asset_tag,
        "monitor_size": p.monitor_size.value if p.monitor_size else None,
        "laptop_os": p.laptop_os.value if p.laptop_os else None,
        "laptop_size": p.laptop_size.value if p.laptop_size else None,
        "delivery_building": bcode,
        "delivery_floor": p.delivery_floor,
        "status": "submitted",
        "created_date": _current_date(db),
    }
    rid = _make_request_id(record, "HW")
    record["request_id"] = rid
    db.setdefault("requests", {})[rid] = record
    ent = db.get("employees", {}).get(p.employee_id, {}).get("hardware_entitlements", {}).get(p.request_type)
    if ent:
        ent.update({"pending_request": True, "pending_request_id": rid})
    # Asset lifecycle update for lost/stolen — set on asset record
    if p.justification == "lost_or_stolen" and p.current_asset_tag:
        a = db.get("assets", {}).get(p.current_asset_tag)
        if a:
            a["lifecycle_status"] = "lost_or_stolen"
            a["status"] = "inactive"
    return {"status": "success", "request_id": rid, "request_type": p.request_type, "message": f"Submitted: {rid}"}


def initiate_asset_return(params, db, call_index):
    try:
        p = InitiateAssetReturnParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, InitiateAssetReturnParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    asset = db.get("assets", {}).get(p.asset_tag)
    if not asset:
        return {"status": "error", "error_type": "not_found", "message": f"Asset {p.asset_tag} not found"}
    deadline = (_parse_date(_current_date(db)) + timedelta(days=14)).strftime("%Y-%m-%d")
    rma = f"RMA-{p.asset_tag[-6:]}"
    asset["return_authorization"] = {
        "return_auth_id": rma,
        "request_id": p.request_id,
        "return_deadline": deadline,
        "status": "pending_return",
    }
    return {
        "status": "success",
        "return_auth_id": rma,
        "asset_tag": p.asset_tag,
        "return_deadline": deadline,
        "message": f"RMA {rma}: return by {deadline}",
    }


def verify_cost_center_budget(params, db, call_index):
    """Auto-fetches dept/CC from employee record (G1)."""
    try:
        p = VerifyCostCenterBudgetParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, VerifyCostCenterBudgetParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _enf(p.employee_id)
    cc_code = emp.get("cost_center_code")
    dept = emp.get("department_code")
    if not cc_code:
        return {
            "status": "error",
            "error_type": "cc_not_on_file",
            "message": f"No cost center on file for {p.employee_id}",
        }
    cc = db.get("cost_centers", {}).get(cc_code)
    if not cc:
        return {"status": "error", "error_type": "not_found", "message": f"Cost center {cc_code} not found"}
    if cc.get("department_code") != dept:
        return {
            "status": "error",
            "error_type": "department_mismatch",
            "message": f"{cc_code} belongs to {cc['department_code']}, not {dept}",
        }
    if not cc.get("has_budget", True):
        return {
            "status": "error",
            "error_type": "insufficient_budget",
            "message": f"Cost center {cc_code} has insufficient budget",
            "cost_center_code": cc_code,
            "remaining_budget_usd": cc.get("remaining_budget_usd", 0),
        }
    return {
        "status": "success",
        "cost_center_code": cc_code,
        "department_code": dept,
        "has_budget": True,
        "remaining_budget_usd": cc.get("remaining_budget_usd", 0),
        "message": "Budget verified",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FLOW 7: APP ACCESS
# ═══════════════════════════════════════════════════════════════════════════════


def _find_catalog_entry(entries: dict, name: str):
    """Find a catalog entry by exact (case-insensitive) match on name or aliases."""
    for cid, entry in entries.items():
        if _catalog_match(name, entry):
            return cid, entry
    return None, None


def get_application_details(params, db, call_index):
    try:
        p = GetApplicationDetailsParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, GetApplicationDetailsParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    apps = db.get("software_catalog", {}).get("applications", {})
    cid, app = _find_catalog_entry(apps, p.application_name)
    if not cid:
        return {
            "status": "error",
            "error_type": "not_found",
            "message": f"No application matching '{p.application_name}'. Please confirm the exact product name.",
        }
    payload = copy.deepcopy(app)
    payload["catalog_id"] = cid
    return {"status": "success", "catalog_id": cid, "application": payload}


def submit_access_request(params, db, call_index):
    try:
        p = SubmitAccessRequestParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, SubmitAccessRequestParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    if not _ok(db, "otp_auth"):
        return _ar("otp_auth")
    app = db.get("software_catalog", {}).get("applications", {}).get(p.catalog_id)
    if not app:
        return {"status": "error", "error_type": "not_found", "message": f"App {p.catalog_id} not found"}
    if p.access_level not in app.get("available_access_levels", []):
        return {
            "status": "error",
            "error_type": "invalid_access_level",
            "message": f"'{p.access_level}' not available. Options: {app['available_access_levels']}",
        }
    approval = app.get("requires_manager_approval", False)
    record = {
        "request_id": None,
        "employee_id": p.employee_id,
        "catalog_id": p.catalog_id,
        "application_name": app["name"],
        "access_level": p.access_level,
        "status": "pending_approval" if approval else "approved",
        "requires_manager_approval": approval,
        "created_date": _current_date(db),
    }
    rid = _make_request_id(record, "SW")
    record["request_id"] = rid
    db.setdefault("requests", {})[rid] = record
    return {
        "status": "success",
        "request_id": rid,
        "application_name": app["name"],
        "access_level": p.access_level,
        "requires_approval": approval,
        "message": f"{rid}" + (" (pending approval)" if approval else " (auto-approved)"),
    }


def route_approval_workflow(params, db, call_index):
    try:
        p = RouteApprovalWorkflowParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, RouteApprovalWorkflowParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    req = db.get("requests", {}).get(p.request_id)
    if not req:
        return {"status": "error", "error_type": "not_found", "message": f"Request {p.request_id} not found"}
    approver = db.get("employees", {}).get(p.approver_employee_id)
    if not approver:
        return _enf(p.approver_employee_id)
    deadline = (_parse_date(_current_date(db)) + timedelta(hours=48)).strftime("%Y-%m-%d %H:%M")
    req.update({"approval_routed_to": p.approver_employee_id, "approval_sla_deadline": deadline})
    return {
        "status": "success",
        "request_id": p.request_id,
        "approver_name": f"{approver['first_name']} {approver['last_name']}",
        "approval_deadline": deadline,
        "message": f"Routed to {approver['first_name']} {approver['last_name']}. 48h window.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FLOWS 8-9: LICENSE
# ═══════════════════════════════════════════════════════════════════════════════


def get_license_catalog_item(params, db, call_index):
    try:
        p = GetLicenseCatalogItemParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, GetLicenseCatalogItemParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    licenses = db.get("software_catalog", {}).get("licenses", {})
    cid, lic = _find_catalog_entry(licenses, p.license_name)
    if not cid:
        return {
            "status": "error",
            "error_type": "not_found",
            "message": f"No license matching '{p.license_name}'. Please confirm the exact product name.",
        }
    payload = copy.deepcopy(lic)
    payload["catalog_id"] = cid
    return {"status": "success", "catalog_id": cid, "license": payload}


def validate_cost_center(params, db, call_index):
    """Auto-fetches dept/CC from employee record (G1)."""
    try:
        p = ValidateCostCenterParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, ValidateCostCenterParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _enf(p.employee_id)
    cc_code = emp.get("cost_center_code")
    dept = emp.get("department_code")
    if not cc_code:
        return {
            "status": "error",
            "error_type": "cc_not_on_file",
            "message": f"No cost center on file for {p.employee_id}",
        }
    cc = db.get("cost_centers", {}).get(cc_code)
    if not cc:
        return {"status": "error", "error_type": "not_found", "message": f"Cost center {cc_code} not found"}
    if cc.get("department_code") != dept:
        return {"status": "error", "error_type": "department_mismatch", "message": f"{cc_code} not for {dept}"}
    return {
        "status": "success",
        "cost_center_code": cc_code,
        "department_code": dept,
        "validated": True,
        "message": f"Cost center {cc_code} validated",
    }


def submit_license_request(params, db, call_index):
    """Merged permanent + temporary license submission (C3)."""
    try:
        p = SubmitLicenseRequestParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, SubmitLicenseRequestParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    record = {
        "request_id": None,
        "employee_id": p.employee_id,
        "catalog_id": p.catalog_id,
        "status": "submitted",
        "created_date": _current_date(db),
    }
    if p.duration_days is None:
        record["request_type"] = "license_request"
        rid = _make_request_id(record, "SW")
        record["request_id"] = rid
        db.setdefault("requests", {})[rid] = record
        return {
            "status": "success",
            "request_id": rid,
            "catalog_id": p.catalog_id,
            "license_type": "permanent",
            "message": f"License request: {rid}",
        }
    exp = (_parse_date(_current_date(db)) + timedelta(days=p.duration_days)).strftime("%Y-%m-%d")
    record.update({"request_type": "temporary_license", "duration_days": p.duration_days, "expiration_date": exp})
    rid = _make_request_id(record, "SW")
    record["request_id"] = rid
    db.setdefault("requests", {})[rid] = record
    return {
        "status": "success",
        "request_id": rid,
        "catalog_id": p.catalog_id,
        "license_type": "temporary",
        "duration_days": p.duration_days,
        "expiration_date": exp,
        "message": f"Temp license: {rid}. Expires {exp}.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FLOW 10: RENEWAL
# ═══════════════════════════════════════════════════════════════════════════════


def get_employee_licenses(params, db, call_index):
    try:
        p = GetEmployeeLicensesParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, GetEmployeeLicensesParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _enf(p.employee_id)
    return {
        "status": "success",
        "employee_id": p.employee_id,
        "licenses": copy.deepcopy(emp.get("software_licenses", [])),
    }


def submit_license_renewal(params, db, call_index):
    """Enforces the renewal window (30 days from expiry or <=14 days expired) in-line (C9)."""
    try:
        p = SubmitLicenseRenewalParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, SubmitLicenseRenewalParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _enf(p.employee_id)
    lic = next(
        (lic for lic in emp.get("software_licenses", []) if lic["license_assignment_id"] == p.license_assignment_id),
        None,
    )
    if not lic:
        return {"status": "error", "error_type": "not_found", "message": f"License {p.license_assignment_id} not found"}
    cur = _current_date(db)
    exp = lic.get("expiration_date", "")
    if cur and exp:
        days = (_parse_date(exp) - _parse_date(cur)).days
        if days > 30:
            return {
                "status": "error",
                "error_type": "not_in_renewal_window",
                "message": f"License expires in {days} days. Renewal is only available within 30 days of expiration. Please call back closer to the expiration date.",
                "expiration_date": exp,
            }
        if days < -14:
            return {
                "status": "error",
                "error_type": "not_in_renewal_window",
                "message": f"License expired {abs(days)} days ago. Renewal is only available up to 14 days past expiration. Submit a new license request instead.",
                "expiration_date": exp,
            }
    new_exp = (max(_parse_date(exp or cur), _parse_date(cur)) + timedelta(days=365)).strftime("%Y-%m-%d")
    lic.update({"expiration_date": new_exp, "status": "active"})
    record = {
        "request_id": None,
        "employee_id": p.employee_id,
        "license_assignment_id": p.license_assignment_id,
        "request_type": "license_renewal",
        "new_expiration_date": new_exp,
        "status": "submitted",
        "created_date": _current_date(db),
    }
    rid = _make_request_id(record, "SW")
    record["request_id"] = rid
    db.setdefault("requests", {})[rid] = record
    return {
        "status": "success",
        "request_id": rid,
        "new_expiration_date": new_exp,
        "message": f"Renewed. Expires {new_exp}. ID: {rid}",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FLOWS 11-14: FACILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def check_desk_availability(params, db, call_index):
    try:
        p = CheckDeskAvailabilityParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, CheckDeskAvailabilityParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    bcode, err = _resolve_facility("building", p.building_code, db)
    if err:
        return err
    desks = db.get("facilities", {}).get("desks", {})
    avail = [
        {"desk_code": c, "zone": d.get("zone"), "near_window": d.get("near_window", False)}
        for c, d in desks.items()
        if d.get("building_code") == bcode and d.get("floor_code") == p.floor_code and d.get("status") == "available"
    ]
    return {"status": "success", "building_code": bcode, "available_desks": avail, "message": f"{len(avail)} desk(s)"}


def submit_desk_assignment(params, db, call_index):
    try:
        p = SubmitDeskAssignmentParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, SubmitDeskAssignmentParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    emp = db.get("employees", {}).get(p.employee_id)
    if emp and emp.get("assigned_desk"):
        return {
            "status": "error",
            "error_type": "already_assigned",
            "message": f"Employee already has desk {emp['assigned_desk']}. Release it first to request a new one.",
        }
    # X9-R2: 90-day cooldown after a prior desk assignment
    last = emp and emp.get("last_desk_assignment_date")
    if last:
        days = (_parse_date(_current_date(db)) - _parse_date(last)).days
        if days < 90:
            return {
                "status": "error",
                "error_type": "too_soon",
                "days_since_last": days,
                "cooldown_days": 90,
                "last_assignment_date": last,
                "message": f"Desk reassignment cooldown: last assignment was {days} days ago; must wait at least 90 days. Eligible again after {(_parse_date(last) + timedelta(days=90)).date().isoformat()}.",
            }
    desk = db.get("facilities", {}).get("desks", {}).get(p.desk_code)
    if not desk:
        return {"status": "error", "error_type": "not_found", "message": f"Desk {p.desk_code} not found"}
    if desk["status"] != "available":
        return {"status": "error", "error_type": "not_available", "message": "Not available"}
    desk.update({"status": "assigned", "assigned_employee_id": p.employee_id})
    if emp is not None:
        emp["assigned_desk"] = p.desk_code
        emp["last_desk_assignment_date"] = _current_date(db)
    return {
        "status": "success",
        "request_id": _make_request_id(
            {
                "employee_id": p.employee_id,
                "desk_code": p.desk_code,
                "request_type": "desk_assignment",
                "created_date": _current_date(db),
            },
            "FAC",
        ),
        "desk_code": p.desk_code,
        "message": f"Assigned {p.desk_code}",
    }


def check_parking_availability(params, db, call_index):
    try:
        p = CheckParkingAvailabilityParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, CheckParkingAvailabilityParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    zcode = None
    if p.zone_code is not None:
        zcode, err = _resolve_facility("zone", p.zone_code, db)
        if err:
            return err
    spaces = db.get("facilities", {}).get("parking", {})
    avail = [
        {
            "parking_space_id": sid,
            "zone_code": s.get("zone_code"),
            "level": s.get("level"),
            "covered": s.get("covered", False),
        }
        for sid, s in spaces.items()
        if (zcode is None or s.get("zone_code") == zcode) and s.get("status") == "available"
    ]
    return {
        "status": "success",
        "zone_code": zcode,
        "available_spaces": avail,
        "message": f"{len(avail)} space(s)" + (" (all zones)" if zcode is None else ""),
    }


def submit_parking_assignment(params, db, call_index):
    try:
        p = SubmitParkingAssignmentParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, SubmitParkingAssignmentParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    emp = db.get("employees", {}).get(p.employee_id)
    if emp and emp.get("assigned_parking"):
        return {
            "status": "error",
            "error_type": "already_assigned",
            "message": f"Employee already has parking {emp['assigned_parking']}. Release it first.",
        }
    # X9-R2: 180-day cooldown after a prior parking assignment
    last = emp and emp.get("last_parking_assignment_date")
    if last:
        days = (_parse_date(_current_date(db)) - _parse_date(last)).days
        if days < 180:
            return {
                "status": "error",
                "error_type": "too_soon",
                "days_since_last": days,
                "cooldown_days": 180,
                "last_assignment_date": last,
                "message": f"Parking reassignment cooldown: last assignment was {days} days ago; must wait at least 180 days. Eligible again after {(_parse_date(last) + timedelta(days=180)).date().isoformat()}.",
            }
    space = db.get("facilities", {}).get("parking", {}).get(p.parking_space_id)
    if not space:
        return {"status": "error", "error_type": "not_found", "message": "Space not found"}
    if space["status"] != "available":
        return {"status": "error", "error_type": "not_available", "message": "Not available"}
    space.update({"status": "assigned", "assigned_employee_id": p.employee_id})
    if emp is not None:
        emp["assigned_parking"] = p.parking_space_id
        emp["last_parking_assignment_date"] = _current_date(db)
    return {
        "status": "success",
        "request_id": _make_request_id(
            {
                "employee_id": p.employee_id,
                "parking_space_id": p.parking_space_id,
                "request_type": "parking_assignment",
                "created_date": _current_date(db),
            },
            "FAC",
        ),
        "parking_space_id": p.parking_space_id,
        "message": f"Assigned {p.parking_space_id}",
    }


def submit_waitlist(params, db, call_index):
    """Place the employee on the waitlist for a desk or parking zone when no spots are currently available."""
    try:
        p = SubmitWaitlistParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, SubmitWaitlistParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _enf(p.employee_id)
    wl = db.setdefault("waitlists", {}).setdefault(p.resource_type.value, [])
    existing = next(
        (w for w in wl if w["employee_id"] == p.employee_id and w["zone_or_building"] == p.zone_or_building), None
    )
    if existing:
        return {
            "status": "error",
            "error_type": "already_waitlisted",
            "message": f"Already on waitlist position {existing['position']}",
        }
    pos = len(wl) + 1
    wid = f"WL-{p.resource_type.value[:3].upper()}-{str(pos).zfill(4)}"
    wl.append(
        {
            "waitlist_id": wid,
            "employee_id": p.employee_id,
            "resource_type": p.resource_type.value,
            "zone_or_building": p.zone_or_building,
            "position": pos,
            "added_date": _current_date(db),
        }
    )
    return {
        "status": "success",
        "waitlist_id": wid,
        "resource_type": p.resource_type.value,
        "position": pos,
        "zone_or_building": p.zone_or_building,
        "message": f"Placed on {p.resource_type.value} waitlist at position {pos}",
    }


def check_ergonomic_assessment(params, db, call_index):
    try:
        p = CheckErgonomicAssessmentParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, CheckErgonomicAssessmentParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _enf(p.employee_id)
    a = emp.get("ergonomic_assessment")
    if not a or a.get("status") != "completed":
        return {
            "status": "error",
            "error_type": "assessment_required",
            "message": "No completed ergonomic assessment. Complete one at occupational health portal.",
        }
    return {
        "status": "success",
        "employee_id": p.employee_id,
        "assessment_date": a.get("completed_date"),
        "message": "Assessment on file",
    }


def submit_equipment_request(params, db, call_index):
    try:
        p = SubmitEquipmentRequestParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, SubmitEquipmentRequestParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    bcode, err = _resolve_facility("building", p.delivery_building, db)
    if err:
        return err
    record = {
        "request_id": None,
        "employee_id": p.employee_id,
        "equipment_type": p.equipment_type,
        "delivery_building": bcode,
        "delivery_floor": p.delivery_floor,
        "status": "submitted",
        "created_date": _current_date(db),
    }
    rid = _make_request_id(record, "FAC")
    record["request_id"] = rid
    db.setdefault("requests", {})[rid] = record
    return {
        "status": "success",
        "request_id": rid,
        "equipment_type": p.equipment_type,
        "delivery_building": bcode,
        "message": f"Equipment request: {rid}",
    }


def check_room_availability(params, db, call_index):
    try:
        p = CheckRoomAvailabilityParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, CheckRoomAvailabilityParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    bcode, err = _resolve_facility("building", p.building_code, db)
    if err:
        return err
    rooms = db.get("facilities", {}).get("conference_rooms", {})
    required = [e.value for e in (p.equipment_required or [])]
    avail = []
    for rc, rm in rooms.items():
        if rm.get("building_code") != bcode:
            continue
        if p.floor_code is not None and rm.get("floor_code") != p.floor_code:
            continue
        if rm.get("capacity", 0) < p.min_capacity:
            continue
        room_equipment = rm.get("equipment", [])
        if required and not all(r in room_equipment for r in required):
            continue
        if any(
            b["date"] == p.date and not (p.end_time <= b.get("start_time", "") or p.start_time >= b.get("end_time", ""))
            for b in rm.get("bookings", [])
        ):
            continue
        avail.append(
            {
                "room_code": rc,
                "floor_code": rm.get("floor_code"),
                "capacity": rm["capacity"],
                "equipment": room_equipment,
            }
        )
    return {"status": "success", "available_rooms": avail, "message": f"{len(avail)} room(s)"}


def submit_room_booking(params, db, call_index):
    try:
        p = SubmitRoomBookingParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, SubmitRoomBookingParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    room = db.get("facilities", {}).get("conference_rooms", {}).get(p.room_code)
    if not room:
        return {"status": "error", "error_type": "not_found", "message": f"Room {p.room_code} not found"}
    if p.attendee_count > room.get("capacity", 0):
        return {
            "status": "error",
            "error_type": "over_capacity",
            "message": f"Attendee count {p.attendee_count} exceeds room capacity {room.get('capacity')}",
        }
    for b in room.get("bookings", []):
        if b["date"] == p.date and not (p.end_time <= b.get("start_time", "") or p.start_time >= b.get("end_time", "")):
            return {
                "status": "error",
                "error_type": "room_conflict",
                "message": f"Booked {p.date} {b['start_time']}-{b['end_time']}",
            }
    booking = {
        "booking_id": None,
        "date": p.date,
        "start_time": p.start_time,
        "end_time": p.end_time,
        "employee_id": p.employee_id,
        "attendee_count": p.attendee_count,
    }
    rid = _make_request_id(booking, "FAC")
    booking["booking_id"] = rid
    room.setdefault("bookings", []).append(booking)
    return {
        "status": "success",
        "request_id": rid,
        "room_code": p.room_code,
        "date": p.date,
        "start_time": p.start_time,
        "end_time": p.end_time,
        "message": f"Booked {p.room_code}. ID: {rid}",
    }


def send_calendar_invite(params, db, call_index):
    """Persists a calendar_events row so task validation can confirm the side effect (C6)."""
    try:
        p = SendCalendarInviteParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, SendCalendarInviteParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    cal_id = f"CAL-{p.request_id[-6:]}"
    db.setdefault("calendar_events", {})[cal_id] = {
        "calendar_event_id": cal_id,
        "request_id": p.request_id,
        "employee_id": p.employee_id,
        "room_code": p.room_code,
        "date": p.date,
        "start_time": p.start_time,
        "end_time": p.end_time,
        "created_date": _current_date(db),
    }
    return {
        "status": "success",
        "calendar_event_id": cal_id,
        "room_code": p.room_code,
        "date": p.date,
        "message": f"Calendar invite sent. Event: {cal_id}",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FLOWS 15-18: ACCOUNTS & ACCESS
# ═══════════════════════════════════════════════════════════════════════════════


def lookup_new_hire(params, db, call_index):
    try:
        p = LookupNewHireParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, LookupNewHireParams)
    if not _ok(db, "manager_auth"):
        return _ar("manager_auth")
    emp = db.get("employees", {}).get(p.new_hire_employee_id)
    if not emp:
        return _enf(p.new_hire_employee_id)
    if emp.get("employment_status") != "pending_start":
        return {
            "status": "error",
            "error_type": "not_new_hire",
            "message": f"Status '{emp['employment_status']}', not 'pending_start'",
        }
    return {
        "status": "success",
        "employee": {
            "employee_id": p.new_hire_employee_id,
            "first_name": emp["first_name"],
            "last_name": emp["last_name"],
            "department_code": emp["department_code"],
            "role_code": emp["role_code"],
            "start_date": emp.get("start_date"),
        },
    }


def check_existing_accounts(params, db, call_index):
    try:
        p = CheckExistingAccountsParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, CheckExistingAccountsParams)
    if not _ok(db, "manager_auth"):
        return _ar("manager_auth")
    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _enf(p.employee_id)
    active = [a for a in emp.get("system_accounts", []) if a.get("status") == "active"]
    if active:
        return {
            "status": "error",
            "error_type": "accounts_already_exist",
            "message": f"{len(active)} active account(s)",
        }
    return {"status": "success", "employee_id": p.employee_id, "existing_accounts": [], "message": "Ready to provision"}


def provision_new_account(params, db, call_index):
    try:
        p = ProvisionNewAccountParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, ProvisionNewAccountParams)
    if not _ok(db, "manager_auth"):
        return _ar("manager_auth")
    if not _ok(db, "otp_auth"):
        return _ar("otp_auth")
    mgr = db.get("employees", {}).get(p.manager_employee_id)
    if not mgr:
        return _enf(p.manager_employee_id)
    if p.new_hire_employee_id not in mgr.get("direct_reports", []):
        return {"status": "error", "error_type": "not_authorized", "message": "Not in direct reports"}
    nh = db.get("employees", {}).get(p.new_hire_employee_id)
    if not nh:
        return _enf(p.new_hire_employee_id)
    # X4: reject provisioning if any requested group is not eligible for the new hire
    for gc in p.access_groups:
        grp = db.get("access_groups", {}).get(gc)
        if grp and not _group_eligible(grp, nh):
            return {
                "status": "error",
                "error_type": "group_not_eligible",
                "group_code": gc,
                "message": f"New hire {p.new_hire_employee_id} (department {nh.get('department_code')}, role {nh.get('role_code')}) is not eligible for {gc}. Use a permission template that matches the role.",
            }
    # C14: email dedup — suffix with a number if the base name collides with another employee's email
    base = f"{nh['first_name'].lower()}.{nh['last_name'].lower()}"
    existing_emails = {e.get("email") for e in db.get("employees", {}).values() if e.get("email")}
    email = f"{base}@company.com"
    suffix = 2
    while email in existing_emails:
        email = f"{base}{suffix}@company.com"
        suffix += 1
    nh["email"] = email
    acct_entry = {
        "case_id": None,
        "status": "active",
        "provisioned_date": _current_date(db),
        "access_groups": list(p.access_groups),
    }
    cid = _make_case_id(acct_entry, "ACCT")
    acct_entry["case_id"] = cid
    nh["system_accounts"] = [acct_entry]
    nh["employment_status"] = "active"
    return {
        "status": "success",
        "case_id": cid,
        "new_hire_employee_id": p.new_hire_employee_id,
        "email": email,
        "access_groups": list(p.access_groups),
        "message": f"Provisioned. Case: {cid}. Email: {email}",
    }


def get_group_memberships(params, db, call_index):
    try:
        p = GetGroupMembershipsParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, GetGroupMembershipsParams)
    if not _ok(db, "otp_auth"):
        return _ar("otp_auth")
    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _enf(p.employee_id)
    return {
        "status": "success",
        "employee_id": p.employee_id,
        "memberships": copy.deepcopy(emp.get("group_memberships", [])),
    }


def get_group_details(params, db, call_index):
    try:
        p = GetGroupDetailsParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, GetGroupDetailsParams)
    if not _ok(db, "otp_auth"):
        return _ar("otp_auth")
    group = db.get("access_groups", {}).get(p.group_code)
    if not group:
        return {"status": "error", "error_type": "not_found", "message": f"Group {p.group_code} not found"}
    return {"status": "success", "group": copy.deepcopy(group)}


def submit_group_membership_change(params, db, call_index):
    try:
        p = SubmitGroupMembershipChangeParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, SubmitGroupMembershipChangeParams)
    if not _ok(db, "otp_auth"):
        return _ar("otp_auth")
    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _enf(p.employee_id)
    group = db.get("access_groups", {}).get(p.group_code)
    if not group:
        return {"status": "error", "error_type": "not_found", "message": f"Group {p.group_code} not found"}
    memberships = emp.get("group_memberships", [])
    codes = [m["group_code"] for m in memberships]
    if p.action == "add":
        if p.group_code in codes:
            return {"status": "error", "error_type": "already_member", "message": f"Already in {p.group_code}"}
        if not _group_eligible(group, emp):
            return {
                "status": "error",
                "error_type": "group_not_eligible",
                "message": f"Employee {p.employee_id} (department {emp.get('department_code')}, role {emp.get('role_code')}) is not eligible for {p.group_code}. Eligible departments: {group.get('eligible_departments') or 'any'}; eligible roles: {group.get('eligible_roles') or 'any'}.",
            }
        appr = group.get("requires_approval", False)
        if appr:
            # Record a pending_approval entry so task validation can see the state change (C2)
            membership = {
                "group_code": p.group_code,
                "group_name": group["name"],
                "status": "pending_approval",
                "case_id": None,
                "request_id": None,
                "requested_date": _current_date(db),
            }
            cid = _make_case_id(membership, "ACCT")
            req_record = {
                "request_id": None,
                "case_id": cid,
                "employee_id": p.employee_id,
                "group_code": p.group_code,
                "action": "add",
                "status": "pending_approval",
                "requires_manager_approval": True,
                "created_date": _current_date(db),
            }
            rid = _make_request_id(req_record, "ACCT")
            membership["case_id"] = cid
            membership["request_id"] = rid
            req_record["request_id"] = rid
            emp.setdefault("group_memberships", []).append(membership)
            db.setdefault("requests", {})[rid] = req_record
            return {
                "status": "success",
                "case_id": cid,
                "request_id": rid,
                "action": "add",
                "group_code": p.group_code,
                "requires_approval": True,
                "message": f"Pending approval. Case: {cid}. Request: {rid}",
            }
        else:
            memberships.append(
                {
                    "group_code": p.group_code,
                    "group_name": group["name"],
                    "status": "active",
                    "added_date": _current_date(db),
                }
            )
            cid = _make_case_id(
                {
                    "employee_id": p.employee_id,
                    "group_code": p.group_code,
                    "action": "add",
                    "auto_approved": True,
                    "created_date": _current_date(db),
                },
                "ACCT",
            )
            return {
                "status": "success",
                "case_id": cid,
                "action": "add",
                "group_code": p.group_code,
                "requires_approval": False,
                "message": f"Added. Case: {cid}",
            }
    else:
        if p.group_code not in codes:
            return {"status": "error", "error_type": "not_member", "message": f"Not in {p.group_code}"}
        emp["group_memberships"] = [m for m in memberships if m["group_code"] != p.group_code]
        cid = _make_case_id(
            {
                "employee_id": p.employee_id,
                "group_code": p.group_code,
                "action": "remove",
                "created_date": _current_date(db),
            },
            "ACCT",
        )
        return {
            "status": "success",
            "case_id": cid,
            "action": "remove",
            "group_code": p.group_code,
            "message": f"Removed. Case: {cid}",
        }


def check_role_change_authorized(params, db, call_index):
    """HR-gated eligibility check for Flow 17 (G2). Requires pending_role_approval on the employee record."""
    try:
        p = CheckRoleChangeAuthorizedParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, CheckRoleChangeAuthorizedParams)
    if not _ok(db, "otp_auth"):
        return _ar("otp_auth")
    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _enf(p.employee_id)
    pr = emp.get("pending_role_approval")
    if not pr or not pr.get("approved_by_hr"):
        return {
            "status": "error",
            "error_type": "not_authorized",
            "message": "HR has not approved a role change for this employee. Ask the caller to contact HR first.",
        }
    if pr.get("new_role_code") != p.new_role_code:
        return {
            "status": "error",
            "error_type": "role_mismatch",
            "message": f"HR-approved role is '{pr.get('new_role_code')}', not '{p.new_role_code}'",
        }
    return {
        "status": "success",
        "employee_id": p.employee_id,
        "new_role_code": p.new_role_code,
        "hr_case_id": pr.get("hr_case_id"),
        "effective_date": pr.get("effective_date"),
        "message": "Role change authorized by HR",
    }


def get_permission_templates(params, db, call_index):
    try:
        p = GetPermissionTemplatesParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, GetPermissionTemplatesParams)
    if not _ok(db, "otp_auth"):
        return _ar("otp_auth")
    tmpls = {
        t: copy.deepcopy(v) for t, v in db.get("permission_templates", {}).items() if v.get("role_code") == p.role_code
    }
    if not tmpls:
        return {"status": "error", "error_type": "not_found", "message": f"No templates for {p.role_code}"}
    return {"status": "success", "role_code": p.role_code, "templates": tmpls}


def submit_permission_change(params, db, call_index):
    try:
        p = SubmitPermissionChangeParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, SubmitPermissionChangeParams)
    if not _ok(db, "otp_auth"):
        return _ar("otp_auth")
    if _parse_date(p.effective_date) < _parse_date(_current_date(db)):
        return {
            "status": "error",
            "error_type": "invalid_effective_date",
            "message": f"effective_date {p.effective_date} is in the past",
        }
    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _enf(p.employee_id)
    # Validate effective_date matches the HR-approved role change record (X23)
    pr = emp.get("pending_role_approval")
    if pr and pr.get("effective_date") and p.effective_date != pr["effective_date"]:
        return {
            "status": "error",
            "error_type": "effective_date_mismatch",
            "message": f"effective_date {p.effective_date} does not match the HR-approved effective date {pr['effective_date']}",
        }
    tmpl = db.get("permission_templates", {}).get(p.permission_template_id)
    if not tmpl:
        return {
            "status": "error",
            "error_type": "not_found",
            "message": f"Template {p.permission_template_id} not found",
        }
    if tmpl["role_code"] != p.new_role_code:
        return {
            "status": "error",
            "error_type": "template_role_mismatch",
            "message": f"Template for '{tmpl['role_code']}', not '{p.new_role_code}'",
        }
    role_change = {
        "case_id": None,
        "new_role_code": p.new_role_code,
        "permission_template_id": p.permission_template_id,
        "effective_date": p.effective_date,
        "status": "pending",
    }
    cid = _make_case_id(role_change, "ACCT")
    role_change["case_id"] = cid
    emp["pending_role_change"] = role_change
    return {
        "status": "success",
        "case_id": cid,
        "employee_id": p.employee_id,
        "effective_date": p.effective_date,
        "message": f"Permission change submitted. Case: {cid}",
    }


def schedule_access_review(params, db, call_index):
    try:
        p = ScheduleAccessReviewParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, ScheduleAccessReviewParams)
    if not _ok(db, "otp_auth"):
        return _ar("otp_auth")
    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _enf(p.employee_id)
    prc = emp.get("pending_role_change")
    if not prc:
        return {
            "status": "error",
            "error_type": "no_pending_role_change",
            "message": "No pending role change to schedule a review for",
        }
    effective = prc.get("effective_date")
    if effective:
        expected = _parse_date(effective) + timedelta(days=90)
        actual = _parse_date(p.review_date)
        if abs((actual - expected).days) > 3:
            return {
                "status": "error",
                "error_type": "invalid_review_date",
                "message": f"review_date must be within 3 days of effective_date + 90 (expected near {expected.strftime('%Y-%m-%d')})",
            }
    rvw = f"ARVW-{p.case_id[-6:]}"
    prc.update({"access_review_id": rvw, "review_date": p.review_date})
    return {
        "status": "success",
        "review_id": rvw,
        "review_date": p.review_date,
        "message": f"90-day review on {p.review_date}. ID: {rvw}",
    }


def get_offboarding_record(params, db, call_index):
    try:
        p = GetOffboardingRecordParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, GetOffboardingRecordParams)
    if not _ok(db, "manager_auth"):
        return _ar("manager_auth")
    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _enf(p.employee_id)
    ob = emp.get("offboarding_record")
    if not ob:
        return {
            "status": "error",
            "error_type": "no_offboarding_record",
            "message": "No off-boarding record. HR must initiate.",
        }
    return {"status": "success", "employee_id": p.employee_id, "offboarding": copy.deepcopy(ob)}


def submit_access_removal(params, db, call_index):
    try:
        p = SubmitAccessRemovalParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, SubmitAccessRemovalParams)
    if not _ok(db, "manager_auth"):
        return _ar("manager_auth")
    if not _ok(db, "otp_auth"):
        return _ar("otp_auth")
    if _parse_date(p.last_working_day) < _parse_date(_current_date(db)):
        return {
            "status": "error",
            "error_type": "invalid_last_working_day",
            "message": f"last_working_day {p.last_working_day} is in the past",
        }
    mgr = db.get("employees", {}).get(p.manager_employee_id)
    if not mgr:
        return _enf(p.manager_employee_id)
    if p.departing_employee_id not in mgr.get("direct_reports", []):
        return {"status": "error", "error_type": "not_authorized", "message": "Not in direct reports"}
    dep = db.get("employees", {}).get(p.departing_employee_id)
    if not dep:
        return _enf(p.departing_employee_id)
    ob = dep.get("offboarding_record")
    if not ob:
        return {"status": "error", "error_type": "no_offboarding_record", "message": "No off-boarding record"}
    dep.update({"employment_status": "terminated", "system_accounts": [], "group_memberships": []})
    if p.removal_scope == "staged":
        dep["email_preserved_until"] = (_parse_date(p.last_working_day) + timedelta(days=30)).strftime("%Y-%m-%d")
    # Apply non-ID fields first, then hash the full offboarding record so the case_id
    # is determined by the same content the migration sees on the migrated DB.
    ob["access_removed"] = True
    ob["removal_scope"] = p.removal_scope
    ob["removal_case_id"] = None
    cid = _make_case_id(ob, "ACCT")
    ob["removal_case_id"] = cid
    return {
        "status": "success",
        "case_id": cid,
        "departing_employee_id": p.departing_employee_id,
        "removal_scope": p.removal_scope,
        "message": f"Access removed. Case: {cid}",
    }


def initiate_asset_recovery(params, db, call_index):
    try:
        p = InitiateAssetRecoveryParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, InitiateAssetRecoveryParams)
    if not _ok(db, "manager_auth"):
        return _ar("manager_auth")
    assets = [
        {"asset_tag": a["asset_tag"], "asset_type": a["asset_type"], "model": a["model"]}
        for a in db.get("assets", {}).values()
        if a.get("assigned_employee_id") == p.departing_employee_id
    ]
    recv = f"RECV-{p.case_id[-6:]}"
    db.setdefault("asset_recoveries", {})[recv] = {
        "recovery_id": recv,
        "case_id": p.case_id,
        "departing_employee_id": p.departing_employee_id,
        "recovery_method": p.recovery_method.value,
        "assets_to_recover": [a["asset_tag"] for a in assets],
        "created_date": _current_date(db),
    }
    return {
        "status": "success",
        "recovery_id": recv,
        "recovery_method": p.recovery_method,
        "assets_to_recover": assets,
        "message": f"Recovery {recv}: {len(assets)} device(s) via {p.recovery_method}",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FLOW 19: SECURITY INCIDENT / STOLEN DEVICE
# ═══════════════════════════════════════════════════════════════════════════════


def report_security_incident(params, db, call_index):
    """Opens a security case for a lost/stolen/compromised device."""
    try:
        p = ReportSecurityIncidentParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, ReportSecurityIncidentParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    asset = db.get("assets", {}).get(p.asset_tag)
    if not asset:
        return {"status": "error", "error_type": "not_found", "message": f"Asset {p.asset_tag} not found"}
    if asset.get("assigned_employee_id") != p.employee_id:
        return {
            "status": "error",
            "error_type": "not_assigned_to_caller",
            "message": f"Asset {p.asset_tag} is not assigned to {p.employee_id}",
        }
    record = {
        "security_case_id": None,
        "employee_id": p.employee_id,
        "asset_tag": p.asset_tag,
        "incident_type": p.incident_type.value,
        "status": "open",
        "opened_date": _current_date(db),
    }
    sec_id = _make_security_case_id(record)
    record["security_case_id"] = sec_id
    db.setdefault("security_cases", {})[sec_id] = record
    if p.incident_type.value in ("lost", "stolen"):
        asset["lifecycle_status"] = "lost_or_stolen"
    return {
        "status": "success",
        "security_case_id": sec_id,
        "asset_tag": p.asset_tag,
        "incident_type": p.incident_type.value,
        "message": f"Security case {sec_id} opened",
    }


def initiate_remote_wipe(params, db, call_index):
    """Triggers a remote wipe on the device and attaches to the security case."""
    try:
        p = InitiateRemoteWipeParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, InitiateRemoteWipeParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    case = db.get("security_cases", {}).get(p.security_case_id)
    if not case:
        return {
            "status": "error",
            "error_type": "not_found",
            "message": f"Security case {p.security_case_id} not found",
        }
    asset = db.get("assets", {}).get(p.asset_tag)
    if not asset:
        return {"status": "error", "error_type": "not_found", "message": f"Asset {p.asset_tag} not found"}
    if case.get("asset_tag") != p.asset_tag:
        return {
            "status": "error",
            "error_type": "case_asset_mismatch",
            "message": "asset_tag does not match the security case",
        }
    wipe_id = f"WIPE-{p.asset_tag[-6:]}"
    asset["remote_wipe"] = {"wipe_id": wipe_id, "status": "dispatched", "dispatched_date": _current_date(db)}
    asset["status"] = "wiped"
    case["remote_wipe_id"] = wipe_id
    return {
        "status": "success",
        "wipe_id": wipe_id,
        "asset_tag": p.asset_tag,
        "message": f"Remote wipe dispatched: {wipe_id}",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FLOW 20: MFA RESET (refusal + escalation)
# ═══════════════════════════════════════════════════════════════════════════════


def submit_mfa_reset(params, db, call_index):
    """Policy: MFA reset / phone-of-record change requires in-person verification. Always returns in_person_required."""
    try:
        p = SubmitMfaResetParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, SubmitMfaResetParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _enf(p.employee_id)
    record = {
        "security_case_id": None,
        "employee_id": p.employee_id,
        "incident_type": "mfa_reset_request",
        "requested_new_phone_last_four": p.new_phone_last_four,
        "status": "requires_in_person",
        "opened_date": _current_date(db),
    }
    sec_id = _make_security_case_id(record)
    record["security_case_id"] = sec_id
    db.setdefault("security_cases", {})[sec_id] = record
    return {
        "status": "error",
        "error_type": "in_person_required",
        "security_case_id": sec_id,
        "message": "Phone-of-record changes cannot be made over the phone. Please visit the IT security office in person with a government-issued ID. A security case has been opened to track this request.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FLOW 21: SOFTWARE REQUEST STATUS / ESCALATION
# ═══════════════════════════════════════════════════════════════════════════════


def get_request_status(params, db, call_index):
    """Returns status + SLA breach flag for a prior request."""
    try:
        p = GetRequestStatusParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, GetRequestStatusParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    req = db.get("requests", {}).get(p.request_id)
    if not req:
        return {"status": "error", "error_type": "not_found", "message": f"Request {p.request_id} not found"}
    sla_deadline = req.get("approval_sla_deadline")
    sla_breached = False
    if sla_deadline:
        try:
            deadline_date = sla_deadline.split(" ")[0]
            sla_breached = _parse_date(_current_date(db)) > _parse_date(deadline_date)
        except Exception:
            sla_breached = False
    return {
        "status": "success",
        "request_id": p.request_id,
        "request_status": req.get("status"),
        "approval_routed_to": req.get("approval_routed_to"),
        "approval_sla_deadline": sla_deadline,
        "sla_breached": sla_breached,
        "created_date": req.get("created_date"),
        "message": f"Request {p.request_id} is {req.get('status')}" + (" (SLA breached)" if sla_breached else ""),
    }


def escalate_approval(params, db, call_index):
    """Escalates a request to a skip-level approver when the original SLA has been breached."""
    try:
        p = EscalateApprovalParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, EscalateApprovalParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    req = db.get("requests", {}).get(p.request_id)
    if not req:
        return {"status": "error", "error_type": "not_found", "message": f"Request {p.request_id} not found"}
    sla_deadline = req.get("approval_sla_deadline")
    if not sla_deadline:
        return {
            "status": "error",
            "error_type": "no_pending_approval",
            "message": "Request has no pending approval routing to escalate",
        }
    try:
        deadline_date = sla_deadline.split(" ")[0]
        if _parse_date(_current_date(db)) <= _parse_date(deadline_date):
            return {
                "status": "error",
                "error_type": "sla_not_breached",
                "message": "Original approval SLA has not yet been breached",
            }
    except Exception:
        pass
    approver = db.get("employees", {}).get(p.escalate_to_employee_id)
    if not approver:
        return _enf(p.escalate_to_employee_id)
    new_deadline = (_parse_date(_current_date(db)) + timedelta(hours=48)).strftime("%Y-%m-%d %H:%M")
    req.update(
        {"escalated_to": p.escalate_to_employee_id, "escalation_sla_deadline": new_deadline, "status": "escalated"}
    )
    return {
        "status": "success",
        "request_id": p.request_id,
        "escalated_to": p.escalate_to_employee_id,
        "approver_name": f"{approver['first_name']} {approver['last_name']}",
        "escalation_deadline": new_deadline,
        "message": f"Escalated to {approver['first_name']} {approver['last_name']}. 48h window.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════


def transfer_to_agent(params, db, call_index):
    try:
        p = TransferToAgentParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, TransferToAgentParams)
    return {
        "status": "success",
        "transfer_id": f"TRF-{p.employee_id}-{str(call_index).zfill(3)}",
        "employee_id": p.employee_id,
        "transfer_reason": p.transfer_reason,
        "estimated_wait": "2-3 minutes",
        "message": "Transferring",
    }


def mark_resolved(params, db, call_index):
    """X10-R2: terminal step for resolved-without-ticket flows (Flow 1a, 3a, 4a). Writes an interactions row so task-completion validation can confirm the call closed cleanly without an incident ticket."""
    try:
        p = MarkResolvedParams.model_validate(params)
    except ValidationError as e:
        return validation_error_response(e, MarkResolvedParams)
    if not _ok(db, "employee_auth"):
        return _ar()
    if p.employee_id not in db.get("employees", {}):
        return _enf(p.employee_id)
    interactions = db.setdefault("interactions", {})
    iid = f"INT-{str(len(interactions) + 1).zfill(6)}"
    interactions[iid] = {
        "interaction_id": iid,
        "employee_id": p.employee_id,
        "flow_id": p.flow_id.value,
        "resolution_type": p.resolution_type.value,
        "logged_date": _current_date(db),
    }
    return {
        "status": "success",
        "interaction_id": iid,
        "flow_id": p.flow_id.value,
        "resolution_type": p.resolution_type.value,
        "message": f"Interaction logged: {iid}",
    }
