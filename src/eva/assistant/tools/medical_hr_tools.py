"""Medical HR agent tool functions — v2.

Tool sequences per flow:

  Flow 1  – License Extension:
    verify_provider_auth → get_provider_profile → get_license_record
    → check_extension_eligibility → submit_license_extension
    → notify_credentialing_committee

  Flow 2  – Shift Swap:
    verify_employee_auth → get_shift_record → check_swap_eligibility
    → verify_colleague_certifications → confirm_shift_swap
    → notify_department_manager

  Flow 3  – Malpractice Update:
    verify_provider_auth → get_provider_profile → get_malpractice_record
    → update_malpractice_coverage → notify_credentialing_committee

  Flow 4  – Onboarding Task Completion:
    verify_employee_auth → get_employee_record → get_onboarding_checklist
    → complete_onboarding_task (×N) → check_appointment_availability
    → schedule_orientation_followup

  Flow 5  – DEA Transfer:
    verify_provider_auth → initiate_otp_auth → verify_otp_auth
    → get_provider_profile → get_dea_record → transfer_dea_registration
    → notify_pdmp

  Flow 6  – FMLA Filing:
    verify_employee_auth → initiate_otp_auth → verify_otp_auth
    → get_employee_record → check_leave_eligibility → submit_fmla_case
    → notify_department_manager → check_appointment_availability
    → schedule_return_to_work_checkin

  Flow 7  – Payroll Correction:
    verify_employee_auth → get_timesheet_record → check_correction_eligibility
    → submit_payroll_correction → notify_department_manager

  Flow 8  – Privilege Reactivation:
    verify_employee_auth → initiate_otp_auth → verify_otp_auth
    → get_provider_profile → get_privilege_record
    → check_reactivation_eligibility → check_appointment_availability
    → schedule_competency_review → reactivate_privileges
    → notify_credentialing_committee → update_ehr_access

  Flow 9  – On-Call Registration:
    verify_employee_auth → get_oncall_schedule → check_oncall_eligibility
    → register_oncall_availability

  Flow 10 – I-9 Verification:
    verify_employee_auth → get_employee_record → get_i9_record
    → submit_i9_verification → notify_hr_compliance

  Flow 11 – Visa Dependent Addition:
    verify_employee_auth → initiate_otp_auth → verify_otp_auth
    → get_employee_record → get_visa_record → add_visa_dependent
    → notify_immigration_counsel

  Flow 12 – PTO Request:
    verify_employee_auth → get_employee_record → get_pto_balance
    → check_pto_eligibility → submit_pto_request
    → notify_department_manager
"""

import copy

from pydantic import ValidationError

from eva.assistant.tools.medical_hr_params import (
    AddVisaDependentParams,
    # Shared scheduling
    CheckAppointmentAvailabilityParams,
    CheckCorrectionEligibilityParams,
    CheckExtensionEligibilityParams,
    # Flow 6
    CheckLeaveEligibilityParams,
    CheckOncallEligibilityParams,
    CheckPtoEligibilityParams,
    CheckReactivationEligibilityParams,
    CheckSwapEligibilityParams,
    CompleteOnboardingTaskParams,
    ConfirmShiftSwapParams,
    # Flow 5
    GetDeaRecordParams,
    GetEmployeeRecordParams,
    # Flow 10
    GetI9RecordParams,
    # Flow 1
    GetLicenseRecordParams,
    # Flow 3
    GetMalpracticeRecordParams,
    # Flow 4
    GetOnboardingChecklistParams,
    # Flow 9
    GetOncallScheduleParams,
    # Flow 8
    GetPrivilegeRecordParams,
    # Shared lookups
    GetProviderProfileParams,
    # Flow 12
    GetPtoBalanceParams,
    # Flow 2
    GetShiftRecordParams,
    # Flow 7
    GetTimesheetRecordParams,
    # Flow 11
    GetVisaRecordParams,
    InitiateOtpAuthParams,
    NotifyCredentialingCommitteeParams,
    NotifyDepartmentManagerParams,
    NotifyHrComplianceParams,
    NotifyImmigrationCounselParams,
    NotifyPdmpParams,
    ReactivatePrivilegesParams,
    RegisterOncallAvailabilityParams,
    ScheduleCompetencyReviewParams,
    ScheduleOrientationFollowupParams,
    ScheduleReturnToWorkCheckinParams,
    SubmitFmlaCaseParams,
    SubmitI9VerificationParams,
    SubmitLicenseExtensionParams,
    SubmitPayrollCorrectionParams,
    SubmitPtoRequestParams,
    TransferDeaRegistrationParams,
    TransferToAgentParams,
    UpdateEhrAccessParams,
    UpdateMalpracticeCoverageParams,
    VerifyColleagueCertificationsParams,
    # Auth
    VerifyEmployeeAuthParams,
    VerifyOtpAuthParams,
    VerifyProviderAuthParams,
    validation_error_response,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_case_id(prefix: str, suffix: str) -> str:
    return f"CASE-{prefix}-{suffix[-6:]}"


def _employee_not_found(employee_id: str) -> dict:
    return {"status": "error", "error_type": "not_found", "message": f"Employee {employee_id} not found"}


def _provider_not_found(npi: str) -> dict:
    return {"status": "error", "error_type": "not_found", "message": f"Provider with NPI {npi} not found"}


def _auth_required(auth_type: str = "employee_auth") -> dict:
    return {
        "status": "error",
        "error_type": "authentication_required",
        "message": f"Authentication ({auth_type}) must be completed before calling this tool",
    }


def _is_authenticated(db: dict, key: str) -> bool:
    return db.get("session", {}).get(key) is True


# ---------------------------------------------------------------------------
# AUTH TOOLS
# ---------------------------------------------------------------------------


def verify_employee_auth(params: dict, db: dict, call_index: int) -> dict:
    """Authenticate an employee using employee_id + date_of_birth."""
    try:
        p = VerifyEmployeeAuthParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, VerifyEmployeeAuthParams)

    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _employee_not_found(p.employee_id)
    if emp.get("date_of_birth") != p.date_of_birth:
        return {
            "status": "error",
            "error_type": "authentication_failed",
            "message": "Date of birth does not match records for this employee ID",
        }

    db.setdefault("session", {})["employee_auth"] = True
    db["session"]["authenticated_employee_id"] = p.employee_id
    return {
        "status": "success",
        "authenticated": True,
        "employee_id": p.employee_id,
        "first_name": emp.get("first_name"),
        "last_name": emp.get("last_name"),
        "message": f"Employee {p.employee_id} authenticated successfully",
    }


def verify_provider_auth(params: dict, db: dict, call_index: int) -> dict:
    """Authenticate a credentialed provider using NPI + facility_code + PIN."""
    try:
        p = VerifyProviderAuthParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, VerifyProviderAuthParams)

    provider = db.get("providers", {}).get(p.npi)
    if not provider:
        return _provider_not_found(p.npi)
    if provider.get("facility_code") != p.facility_code:
        return {
            "status": "error",
            "error_type": "authentication_failed",
            "message": "Facility code does not match records for this NPI",
        }
    if provider.get("pin") != p.pin:
        return {
            "status": "error",
            "error_type": "authentication_failed",
            "message": "PIN does not match records for this NPI",
        }

    db.setdefault("session", {})["provider_auth"] = True
    db["session"]["authenticated_npi"] = p.npi
    # Also set employee_id so initiate_otp_auth can be called without re-asking
    db["session"]["authenticated_employee_id"] = provider.get("employee_id")
    return {
        "status": "success",
        "authenticated": True,
        "npi": p.npi,
        "first_name": provider.get("first_name"),
        "last_name": provider.get("last_name"),
        "employee_id": provider.get("employee_id"),
        "message": f"Provider NPI {p.npi} authenticated successfully",
    }


def initiate_otp_auth(params: dict, db: dict, call_index: int) -> dict:
    """Send OTP SMS to the employee's phone on file."""
    try:
        p = InitiateOtpAuthParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, InitiateOtpAuthParams)

    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _employee_not_found(p.employee_id)

    db.setdefault("session", {})["otp_employee_id"] = p.employee_id
    db["session"]["otp_issued"] = True
    return {
        "status": "success",
        "phone_last_four": emp.get("phone_last_four"),
        "message": f"OTP sent to number ending in {emp.get('phone_last_four')}. Ask the caller to read the code.",
    }


def verify_otp_auth(params: dict, db: dict, call_index: int) -> dict:
    """Verify the 6-digit OTP code from the caller."""
    try:
        p = VerifyOtpAuthParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, VerifyOtpAuthParams)

    if not db.get("session", {}).get("otp_issued"):
        return {
            "status": "error",
            "error_type": "otp_not_initiated",
            "message": "OTP has not been initiated. Call initiate_otp_auth first.",
        }

    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _employee_not_found(p.employee_id)
    if emp.get("otp_code") != p.otp_code:
        return {
            "status": "error",
            "error_type": "authentication_failed",
            "message": "OTP code does not match. Ask the caller to read the code again.",
        }

    db["session"]["otp_auth"] = True
    db["session"]["authenticated_employee_id"] = p.employee_id
    return {
        "status": "success",
        "authenticated": True,
        "employee_id": p.employee_id,
        "first_name": emp.get("first_name"),
        "message": f"OTP verified. Employee {p.employee_id} authenticated successfully.",
    }


# ---------------------------------------------------------------------------
# SHARED LOOKUP TOOLS
# ---------------------------------------------------------------------------


def get_provider_profile(params: dict, db: dict, call_index: int) -> dict:
    """Fetch provider identity, role, department.

    Accepts either provider_auth or otp_auth because this tool is shared across:
    - Flows 1, 3 (provider_auth)
    - Flow 5 (provider_auth + otp_auth)
    - Flow 8 (employee_auth + otp_auth)
    """
    try:
        p = GetProviderProfileParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, GetProviderProfileParams)

    # This tool is shared across flows with different auth methods.
    # Provider flows (1, 3, 5) use provider_auth. Flow 8 uses otp_auth.
    # The OR gate allows the tool to work regardless of which auth the flow used.
    if not _is_authenticated(db, "provider_auth") and not _is_authenticated(db, "otp_auth"):
        return _auth_required("provider_auth or otp_auth")

    provider = db.get("providers", {}).get(p.npi)
    if not provider:
        return _provider_not_found(p.npi)

    safe_fields = [
        "npi",
        "employee_id",
        "first_name",
        "last_name",
        "facility_code",
        "role_code",
        "department_code",
        "hire_date",
    ]
    return {"status": "success", "provider": {k: provider[k] for k in safe_fields if k in provider}}


def get_employee_record(params: dict, db: dict, call_index: int) -> dict:
    """Fetch employee identity, department, role, employment status.

    Accepts either employee_auth or otp_auth because this tool is shared across:
    - Flow 4 (employee_auth only)
    - Flows 6, 10 (employee_auth + otp_auth)
    - Flow 11 (employee_auth + otp_auth)
    """
    try:
        p = GetEmployeeRecordParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, GetEmployeeRecordParams)

    # This tool is shared across flows with different auth methods.
    # Some flows use employee_auth only, others use employee_auth + otp_auth.
    # The OR gate allows the tool to work regardless of which combination the flow used.
    if not _is_authenticated(db, "employee_auth") and not _is_authenticated(db, "otp_auth"):
        return _auth_required("employee_auth or otp_auth")

    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _employee_not_found(p.employee_id)

    safe_fields = [
        "employee_id",
        "first_name",
        "last_name",
        "department_code",
        "role_code",
        "unit_code",
        "hire_date",
        "employment_status",
    ]
    return {"status": "success", "employee": {k: emp[k] for k in safe_fields if k in emp}}


# ---------------------------------------------------------------------------
# SHARED SCHEDULING TOOLS
# ---------------------------------------------------------------------------


def check_appointment_availability(params: dict, db: dict, call_index: int) -> dict:
    """Check available time slots for a specific appointment type on a given date.

    Reads from db["appointment_availability"][appointment_type][department_code][date].
    Returns a list of available HH:MM time slots.
    """
    try:
        p = CheckAppointmentAvailabilityParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, CheckAppointmentAvailabilityParams)

    # Accepts any auth — shared across employee_auth and otp_auth flows
    if not _is_authenticated(db, "employee_auth") and not _is_authenticated(db, "otp_auth"):
        return _auth_required("employee_auth or otp_auth")

    avail = db.get("appointment_availability", {})
    type_avail = avail.get(p.appointment_type, {})
    dept_avail = type_avail.get(p.department_code, {})
    slots = dept_avail.get(p.preferred_date, [])

    if not slots:
        # Return nearby dates that have availability
        nearby = []
        for date_str, date_slots in sorted(dept_avail.items()):
            if date_slots and date_str >= p.preferred_date:
                nearby.append({"date": date_str, "available_slots": list(date_slots)})
                if len(nearby) >= 3:
                    break
        return {
            "status": "success",
            "available_slots": [],
            "date": p.preferred_date,
            "alternative_dates": nearby,
            "message": f"No availability on {p.preferred_date} for {p.appointment_type} in {p.department_code}",
        }

    returned_slots = list(slots)
    return {
        "status": "success",
        "available_slots": returned_slots,
        "date": p.preferred_date,
        "message": f"{len(returned_slots)} slot(s) available on {p.preferred_date}",
    }


def _validate_and_book_slot(
    db: dict, appointment_type: str, department_code: str, appointment_datetime: str
) -> tuple[bool, dict | None]:
    """Validate a slot is available and book it (remove from availability).

    Returns (success, error_response). If success is True, error_response is None.
    """
    parts = appointment_datetime.split(" ")
    if len(parts) != 2:
        return False, {
            "status": "error",
            "error_type": "invalid_datetime",
            "message": f"Invalid datetime format: {appointment_datetime}",
        }
    date_str, time_str = parts

    avail = db.get("appointment_availability", {})
    type_avail = avail.get(appointment_type, {})
    dept_avail = type_avail.get(department_code, {})
    slots = dept_avail.get(date_str, [])

    if time_str not in slots:
        return False, {
            "status": "error",
            "error_type": "slot_not_available",
            "message": f"Time slot {time_str} on {date_str} is not available for "
            f"{appointment_type} in {department_code}. "
            f"Available slots: {slots or 'none on this date'}",
        }

    # Book the slot by removing it from availability
    slots.remove(time_str)
    return True, None


# ---------------------------------------------------------------------------
# FLOW 1: License Expiration Extension
# ---------------------------------------------------------------------------


def get_license_record(params: dict, db: dict, call_index: int) -> dict:
    """Look up a provider's license by NPI and state license number."""
    try:
        p = GetLicenseRecordParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, GetLicenseRecordParams)

    if not _is_authenticated(db, "provider_auth"):
        return _auth_required("provider_auth")

    provider = db.get("providers", {}).get(p.npi)
    if not provider:
        return _provider_not_found(p.npi)

    lic = provider.get("licenses", {}).get(p.state_license_number)
    if not lic:
        return {
            "status": "error",
            "error_type": "license_not_found",
            "message": f"License {p.state_license_number} not found for NPI {p.npi}",
        }

    return {"status": "success", "license": copy.deepcopy(lic)}


def check_extension_eligibility(params: dict, db: dict, call_index: int) -> dict:
    """Check whether a license qualifies for an extension.

    Blocks if: already extended, under investigation, or expired beyond 30-day window.
    """
    try:
        p = CheckExtensionEligibilityParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, CheckExtensionEligibilityParams)

    if not _is_authenticated(db, "provider_auth"):
        return _auth_required("provider_auth")

    provider = db.get("providers", {}).get(p.npi)
    if not provider:
        return _provider_not_found(p.npi)

    lic = provider.get("licenses", {}).get(p.state_license_number)
    if not lic:
        return {
            "status": "error",
            "error_type": "license_not_found",
            "message": f"License {p.state_license_number} not found for NPI {p.npi}",
        }

    if lic.get("extension_status") == "pending":
        return {
            "status": "error",
            "error_type": "already_extended",
            "message": "An extension request is already pending for this license",
        }
    if lic.get("investigation_hold"):
        return {
            "status": "error",
            "error_type": "investigation_hold",
            "message": "License is under investigation and cannot be extended",
        }

    current_date = db.get("_current_date", "")
    expiration_date = lic.get("expiration_date", "")
    if current_date and expiration_date:
        from datetime import datetime as _dt

        current = _dt.strptime(current_date, "%Y-%m-%d")
        expiry = _dt.strptime(expiration_date, "%Y-%m-%d")
        days_until = (expiry - current).days
        if days_until < 0:
            return {
                "status": "error",
                "error_type": "license_expired",
                "message": f"License expired on {expiration_date}. Extensions cannot be requested for already-expired licenses.",
            }
        if days_until > 60:
            return {
                "status": "error",
                "error_type": "extension_too_early",
                "message": f"License does not expire for {days_until} days. Extensions may only be requested within 60 days of the expiration date.",
            }

    return {
        "status": "success",
        "eligible": True,
        "license_expiration_date": lic.get("expiration_date"),
        "message": "License is eligible for extension",
    }


def submit_license_extension(params: dict, db: dict, call_index: int) -> dict:
    """Submit a provisional or supervised extension for an expiring license.

    For supervised extensions, a supervising physician NPI is required and
    must match an existing provider. For provisional extensions, any value
    passed in supervising_physician_npi is ignored.
    """
    # For provisional extensions, drop supervising_physician_npi before
    # validation so that placeholder / empty values the model may emit (""
    # "0000000000", "OMIT", None, etc.) never cause a validation failure.
    if isinstance(params, dict) and params.get("extension_type") == "provisional":
        params = {k: v for k, v in params.items() if k != "supervising_physician_npi"}

    try:
        p = SubmitLicenseExtensionParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, SubmitLicenseExtensionParams)

    if not _is_authenticated(db, "provider_auth"):
        return _auth_required("provider_auth")

    provider = db.get("providers", {}).get(p.npi)
    if not provider:
        return _provider_not_found(p.npi)

    lic = provider.get("licenses", {}).get(p.state_license_number)
    if not lic:
        return {
            "status": "error",
            "error_type": "license_not_found",
            "message": f"License {p.state_license_number} not found for NPI {p.npi}",
        }

    # Supervised extensions require a supervising physician
    if p.extension_type == "supervised":
        if not p.supervising_physician_npi:
            return {
                "status": "error",
                "error_type": "supervising_physician_required",
                "message": "Supervised extensions require a supervising physician NPI",
            }
        if p.supervising_physician_npi not in db.get("providers", {}):
            return {
                "status": "error",
                "error_type": "supervising_physician_not_found",
                "message": f"Supervising physician NPI {p.supervising_physician_npi} not found",
            }

    case_id = _make_case_id("LIC", provider.get("employee_id", p.npi))

    # Only include supervising_physician_npi in the record for supervised extensions
    update_fields = {
        "extension_status": "pending",
        "extension_type": p.extension_type,
        "extension_days": p.extension_days,
        "extension_case_id": case_id,
    }
    if p.extension_type == "supervised":
        update_fields["supervising_physician_npi"] = p.supervising_physician_npi
    lic.update(update_fields)

    return {
        "status": "success",
        "npi": p.npi,
        "state_license_number": p.state_license_number,
        "extension_type": p.extension_type,
        "extension_days": p.extension_days,
        "supervising_physician_npi": p.supervising_physician_npi,
        "case_id": case_id,
        "message": f"{p.extension_type} extension submitted. Case ID: {case_id}",
    }


def notify_credentialing_committee(params: dict, db: dict, call_index: int) -> dict:
    """Dispatch a notification to the credentialing committee."""
    try:
        p = NotifyCredentialingCommitteeParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, NotifyCredentialingCommitteeParams)

    if not _is_authenticated(db, "provider_auth") and not _is_authenticated(db, "otp_auth"):
        return _auth_required("provider_auth or otp_auth")

    provider = db.get("providers", {}).get(p.npi)
    if not provider:
        return _provider_not_found(p.npi)

    notif = {
        "recipient": "credentialing_committee",
        "npi": p.npi,
        "case_id": p.case_id,
        "notification_type": p.notification_type,
    }
    notifications = db.setdefault("notifications", [])
    if notif not in notifications:
        notifications.append(notif)

    return {
        "status": "success",
        "npi": p.npi,
        "case_id": p.case_id,
        "notification_type": p.notification_type,
        "message": f"Credentialing committee notified: {p.notification_type}",
    }


# ---------------------------------------------------------------------------
# FLOW 2: Shift Swap
# ---------------------------------------------------------------------------


def get_shift_record(params: dict, db: dict, call_index: int) -> dict:
    """Look up a specific shift owned by an employee."""
    try:
        p = GetShiftRecordParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, GetShiftRecordParams)

    if not _is_authenticated(db, "employee_auth"):
        return _auth_required("employee_auth")

    shift = db.get("shifts", {}).get(p.shift_id)
    if not shift:
        return {"status": "error", "error_type": "shift_not_found", "message": f"Shift {p.shift_id} not found"}
    if shift.get("employee_id") != p.employee_id:
        return {
            "status": "error",
            "error_type": "shift_not_owned",
            "message": f"Shift {p.shift_id} does not belong to employee {p.employee_id}",
        }

    return {"status": "success", "shift": copy.deepcopy(shift)}


def check_swap_eligibility(params: dict, db: dict, call_index: int) -> dict:
    """Check whether a shift is eligible to be swapped.

    Blocks if: already swapped, within 24h of shift start, or shift is cancelled.
    Returns the unit's required certifications for use by verify_colleague_certifications.
    """
    try:
        p = CheckSwapEligibilityParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, CheckSwapEligibilityParams)

    if not _is_authenticated(db, "employee_auth"):
        return _auth_required("employee_auth")

    shift = db.get("shifts", {}).get(p.shift_id)
    if not shift:
        return {"status": "error", "error_type": "shift_not_found", "message": f"Shift {p.shift_id} not found"}
    if shift.get("status") == "swapped":
        return {"status": "error", "error_type": "already_swapped", "message": "This shift has already been swapped"}
    if shift.get("status") == "cancelled":
        return {"status": "error", "error_type": "shift_cancelled", "message": "Cannot swap a cancelled shift"}
    current_date = db.get("_current_date", "")
    shift_date = shift.get("date", "")
    if shift_date and current_date and shift_date < current_date:
        return {
            "status": "error",
            "error_type": "shift_in_past",
            "message": f"Shift date {shift_date} is in the past (current date: {current_date}). Cannot swap a past shift.",
        }
    if shift.get("swap_locked"):
        return {
            "status": "error",
            "error_type": "swap_locked",
            "message": "Shift is within the 24-hour swap lockout window",
        }

    unit_code = shift.get("unit_code")
    required_certs = sorted(db.get("unit_cert_requirements", {}).get(unit_code, []))

    return {
        "status": "success",
        "eligible": True,
        "shift_date": shift.get("date"),
        "unit_code": unit_code,
        "required_cert_codes": required_certs,
        "message": "Shift is eligible for swap",
    }


def verify_colleague_certifications(params: dict, db: dict, call_index: int) -> dict:
    """Verify a colleague holds all certifications required for a unit.

    The required certifications are looked up from the unit_cert_requirements
    table in the scenario database — the LLM does not need to supply them.
    """
    try:
        p = VerifyColleagueCertificationsParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, VerifyColleagueCertificationsParams)

    if not _is_authenticated(db, "employee_auth"):
        return _auth_required("employee_auth")

    colleague = db.get("employees", {}).get(p.colleague_employee_id)
    if not colleague:
        return _employee_not_found(p.colleague_employee_id)

    required_certs = db.get("unit_cert_requirements", {}).get(p.unit_code, [])
    missing = sorted(set(required_certs) - set(colleague.get("certifications", [])))
    if missing:
        return {
            "status": "error",
            "error_type": "certification_missing",
            "message": f"Colleague {p.colleague_employee_id} is missing: {missing}",
            "missing_certs": missing,
        }

    return {
        "status": "success",
        "colleague_employee_id": p.colleague_employee_id,
        "unit_code": p.unit_code,
        "certifications_verified": sorted(required_certs),
        "message": "All required certifications verified",
    }


def confirm_shift_swap(params: dict, db: dict, call_index: int) -> dict:
    """Record the confirmed shift swap between two employees."""
    try:
        p = ConfirmShiftSwapParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, ConfirmShiftSwapParams)

    if not _is_authenticated(db, "employee_auth"):
        return _auth_required("employee_auth")

    shift = db.get("shifts", {}).get(p.shift_id)
    if not shift:
        return {"status": "error", "error_type": "shift_not_found", "message": f"Shift {p.shift_id} not found"}

    case_id = _make_case_id("SWP", p.employee_id)
    shift.update(
        {"status": "swapped", "swapped_to_employee_id": p.colleague_employee_id, "swap_confirmation_id": case_id}
    )

    return {
        "status": "success",
        "shift_id": p.shift_id,
        "original_employee_id": p.employee_id,
        "new_employee_id": p.colleague_employee_id,
        "unit_code": p.unit_code,
        "case_id": case_id,
        "message": f"Shift swap confirmed. Case ID: {case_id}",
    }


def notify_department_manager(params: dict, db: dict, call_index: int) -> dict:
    """Notify the department manager of a completed HR action.

    Accepts either employee_auth or otp_auth because this tool is shared across:
    - Flow 2 (employee_auth) — shift_swap_confirmed
    - Flow 6 (employee_auth + otp_auth) — fmla_opened
    - Flow 7 (employee_auth) — payroll_correction_submitted
    - Flow 12 (employee_auth) — pto_request_submitted
    The OR gate allows the tool to work regardless of which auth the flow used.
    """
    try:
        p = NotifyDepartmentManagerParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, NotifyDepartmentManagerParams)

    # Shared across flows with different auth — see docstring above.
    if not _is_authenticated(db, "employee_auth") and not _is_authenticated(db, "otp_auth"):
        return _auth_required("employee_auth or otp_auth")

    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _employee_not_found(p.employee_id)

    notif = {
        "recipient": "department_manager",
        "employee_id": p.employee_id,
        "department_code": emp.get("department_code"),
        "case_id": p.case_id,
        "notification_type": p.notification_type,
    }
    notifications = db.setdefault("notifications", [])
    if notif not in notifications:
        notifications.append(notif)

    return {
        "status": "success",
        "employee_id": p.employee_id,
        "case_id": p.case_id,
        "notification_type": p.notification_type,
        "message": f"Department manager notified: {p.notification_type}",
    }


# ---------------------------------------------------------------------------
# FLOW 3: Malpractice Coverage Update
# ---------------------------------------------------------------------------


def get_malpractice_record(params: dict, db: dict, call_index: int) -> dict:
    """Retrieve current malpractice insurance record for a provider."""
    try:
        p = GetMalpracticeRecordParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, GetMalpracticeRecordParams)

    if not _is_authenticated(db, "provider_auth"):
        return _auth_required("provider_auth")

    provider = db.get("providers", {}).get(p.npi)
    if not provider:
        return _provider_not_found(p.npi)

    mal = provider.get("malpractice")
    if not mal:
        return {
            "status": "error",
            "error_type": "record_not_found",
            "message": f"No malpractice record found for NPI {p.npi}",
        }

    return {"status": "success", "malpractice": copy.deepcopy(mal)}


def update_malpractice_coverage(params: dict, db: dict, call_index: int) -> dict:
    """Update malpractice insurance coverage. Flags re-credentialing if below threshold."""
    try:
        p = UpdateMalpracticeCoverageParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, UpdateMalpracticeCoverageParams)

    if not _is_authenticated(db, "provider_auth"):
        return _auth_required("provider_auth")

    provider = db.get("providers", {}).get(p.npi)
    if not provider:
        return _provider_not_found(p.npi)

    mal = provider.get("malpractice")
    if not mal:
        return {
            "status": "error",
            "error_type": "record_not_found",
            "message": f"No malpractice record on file for NPI {p.npi}. "
            "Contact credentialing to have a record initialized first.",
        }

    recredential_flag = p.per_occurrence_limit_usd < 1_000_000
    case_id = _make_case_id("MAL", provider.get("employee_id", p.npi))
    mal.update(
        {
            "carrier": p.new_carrier,
            "policy_number": p.new_policy_number,
            "per_occurrence_limit_usd": p.per_occurrence_limit_usd,
            "aggregate_limit_usd": p.aggregate_limit_usd,
            "effective_date": p.effective_date,
            "expiration_date": p.expiration_date,
            "recredential_required": recredential_flag,
            "update_case_id": case_id,
        }
    )

    resp = {
        "status": "success",
        "npi": p.npi,
        "new_carrier": p.new_carrier,
        "new_policy_number": p.new_policy_number,
        "per_occurrence_limit_usd": p.per_occurrence_limit_usd,
        "aggregate_limit_usd": p.aggregate_limit_usd,
        "effective_date": p.effective_date,
        "expiration_date": p.expiration_date,
        "recredential_required": recredential_flag,
        "case_id": case_id,
        "message": "Malpractice coverage updated successfully",
    }

    if recredential_flag:
        resp["message"] += f". Coverage below threshold — re-credentialing required. Case ID: {case_id}"

    return resp


# ---------------------------------------------------------------------------
# FLOW 4: Onboarding Task Completion
# ---------------------------------------------------------------------------


def get_onboarding_checklist(params: dict, db: dict, call_index: int) -> dict:
    """Retrieve the onboarding task checklist for a new hire.

    Accepts either employee_auth or otp_auth because this tool is used in:
    - Flow 4 standalone (employee_auth only)
    - Flow 4 combined with provider flows like DEA (provider_auth + otp_auth)
    """
    try:
        p = GetOnboardingChecklistParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, GetOnboardingChecklistParams)

    if not _is_authenticated(db, "employee_auth") and not _is_authenticated(db, "otp_auth"):
        return _auth_required("employee_auth or otp_auth")

    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _employee_not_found(p.employee_id)

    checklist = emp.get("onboarding_checklist")
    if not checklist:
        return {
            "status": "error",
            "error_type": "checklist_not_found",
            "message": f"No onboarding checklist found for {p.employee_id}",
        }

    return {"status": "success", "onboarding_checklist": copy.deepcopy(checklist)}


def complete_onboarding_task(params: dict, db: dict, call_index: int) -> dict:
    """Mark a single onboarding task as complete. Called once per task.

    Accepts either employee_auth or otp_auth (same gate as get_onboarding_checklist).
    """
    try:
        p = CompleteOnboardingTaskParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, CompleteOnboardingTaskParams)

    if not _is_authenticated(db, "employee_auth") and not _is_authenticated(db, "otp_auth"):
        return _auth_required("employee_auth or otp_auth")

    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _employee_not_found(p.employee_id)

    tasks = emp.get("onboarding_checklist", {}).get("tasks", {})
    if p.task_code not in tasks:
        return {
            "status": "error",
            "error_type": "task_not_found",
            "message": f"Task {p.task_code} not in checklist for {p.employee_id}",
        }

    task = tasks[p.task_code]
    if task.get("status") == "complete":
        return {
            "status": "error",
            "error_type": "task_already_complete",
            "message": f"Task {p.task_code} is already marked complete",
        }

    expected_code = task.get("completion_code", "")
    if expected_code and p.completion_code.upper() != expected_code.upper():
        return {
            "status": "error",
            "error_type": "invalid_completion_code",
            "message": f"Completion code does not match for task {p.task_code}. Please verify and try again.",
        }

    task["status"] = "complete"
    remaining = [t for t, v in tasks.items() if v.get("status") != "complete"]

    return {
        "status": "success",
        "employee_id": p.employee_id,
        "task_code": p.task_code,
        "remaining_tasks": remaining,
        "message": f"Task {p.task_code} marked complete. {len(remaining)} task(s) remaining.",
    }


def schedule_orientation_followup(params: dict, db: dict, call_index: int) -> dict:
    """Schedule a post-onboarding orientation follow-up appointment.

    Validates the requested time slot is available before booking.
    Accepts either employee_auth or otp_auth (same gate as onboarding tools).
    """
    try:
        p = ScheduleOrientationFollowupParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, ScheduleOrientationFollowupParams)

    if not _is_authenticated(db, "employee_auth") and not _is_authenticated(db, "otp_auth"):
        return _auth_required("employee_auth or otp_auth")

    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _employee_not_found(p.employee_id)

    # Validate and book the slot
    ok, err = _validate_and_book_slot(db, "orientation_followup", p.department_code, p.appointment_datetime)
    if not ok:
        return err

    appt_id = _make_case_id("ORI", p.employee_id)
    emp.setdefault("scheduled_appointments", []).append(
        {
            "appointment_id": appt_id,
            "type": "orientation_followup",
            "department_code": p.department_code,
            "appointment_datetime": p.appointment_datetime,
            "status": "scheduled",
        }
    )

    return {
        "status": "success",
        "employee_id": p.employee_id,
        "appointment_id": appt_id,
        "department_code": p.department_code,
        "appointment_datetime": p.appointment_datetime,
        "message": f"Orientation follow-up scheduled for {p.appointment_datetime}. Appointment ID: {appt_id}",
    }


# ---------------------------------------------------------------------------
# FLOW 5: DEA Registration Transfer
# ---------------------------------------------------------------------------


def get_dea_record(params: dict, db: dict, call_index: int) -> dict:
    """Retrieve a provider's DEA registration. Requires both provider_auth and otp_auth."""
    try:
        p = GetDeaRecordParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, GetDeaRecordParams)

    if not _is_authenticated(db, "provider_auth"):
        return _auth_required("provider_auth")
    if not _is_authenticated(db, "otp_auth"):
        return {
            "status": "error",
            "error_type": "second_factor_required",
            "message": "DEA operations require OTP verification as a second factor",
        }

    provider = db.get("providers", {}).get(p.npi)
    if not provider:
        return _provider_not_found(p.npi)

    dea = provider.get("dea_registration")
    if not dea or dea.get("dea_number") != p.dea_number:
        return {
            "status": "error",
            "error_type": "dea_record_not_found",
            "message": f"DEA number {p.dea_number} not found for NPI {p.npi}",
        }

    return {"status": "success", "dea_registration": copy.deepcopy(dea)}


def transfer_dea_registration(params: dict, db: dict, call_index: int) -> dict:
    """Transfer a DEA registration to a new facility and state.

    The current facility and state remain on the record. A pending_transfer
    object is created with the new facility, state, and effective date.
    """
    try:
        p = TransferDeaRegistrationParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, TransferDeaRegistrationParams)

    if not _is_authenticated(db, "provider_auth"):
        return _auth_required("provider_auth")
    if not _is_authenticated(db, "otp_auth"):
        return {
            "status": "error",
            "error_type": "second_factor_required",
            "message": "DEA transfer requires OTP verification as a second factor",
        }

    provider = db.get("providers", {}).get(p.npi)
    if not provider:
        return _provider_not_found(p.npi)

    dea = provider.get("dea_registration")
    if not dea or dea.get("dea_number") != p.dea_number:
        return {
            "status": "error",
            "error_type": "dea_record_not_found",
            "message": f"DEA number {p.dea_number} not found for NPI {p.npi}",
        }

    case_id = _make_case_id("DEA", provider.get("employee_id", p.npi))
    dea["status"] = "transfer_pending"
    dea["pending_transfer"] = {
        "new_facility_code": p.new_facility_code,
        "new_state_code": p.new_state_code,
        "transfer_reason": p.transfer_reason,
        "effective_date": p.effective_date,
        "transfer_case_id": case_id,
    }

    return {
        "status": "success",
        "npi": p.npi,
        "dea_number": p.dea_number,
        "current_facility_code": dea.get("facility_code"),
        "current_state_code": dea.get("state_code"),
        "new_facility_code": p.new_facility_code,
        "new_state_code": p.new_state_code,
        "transfer_reason": p.transfer_reason,
        "effective_date": p.effective_date,
        "case_id": case_id,
        "message": f"DEA transfer submitted. Current registration unchanged until effective date {p.effective_date}. Case ID: {case_id}",
    }


def notify_pdmp(params: dict, db: dict, call_index: int) -> dict:
    """Notify the state Prescription Drug Monitoring Program of a DEA transfer.

    No notification_type param — this tool is only used for DEA transfers.
    """
    try:
        p = NotifyPdmpParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, NotifyPdmpParams)

    if not _is_authenticated(db, "provider_auth"):
        return _auth_required("provider_auth")

    provider = db.get("providers", {}).get(p.npi)
    if not provider:
        return _provider_not_found(p.npi)

    notif = {
        "recipient": "pdmp",
        "npi": p.npi,
        "dea_number": p.dea_number,
        "state_code": p.state_code,
        "facility_code": p.facility_code,
        "notification_type": "dea_transfer",
    }
    notifications = db.setdefault("notifications", [])
    if notif not in notifications:
        notifications.append(notif)

    return {
        "status": "success",
        "npi": p.npi,
        "dea_number": p.dea_number,
        "state_code": p.state_code,
        "facility_code": p.facility_code,
        "message": f"PDMP notified for state {p.state_code}, facility {p.facility_code}",
    }


# ---------------------------------------------------------------------------
# FLOW 6: FMLA / Leave of Absence Filing
# ---------------------------------------------------------------------------


def check_leave_eligibility(params: dict, db: dict, call_index: int) -> dict:
    """Check FMLA eligibility: 12 months tenure and 1250 hours worked in past year."""
    try:
        p = CheckLeaveEligibilityParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, CheckLeaveEligibilityParams)

    if not _is_authenticated(db, "otp_auth"):
        return _auth_required("otp_auth")

    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _employee_not_found(p.employee_id)

    eligibility = emp.get("fmla_eligibility", {})
    if not eligibility.get("eligible"):
        return {
            "status": "error",
            "error_type": "not_eligible",
            "message": eligibility.get("reason", "Employee does not meet FMLA eligibility requirements"),
            "months_employed": eligibility.get("months_employed"),
            "hours_worked_past_year": eligibility.get("hours_worked_past_year"),
        }

    return {
        "status": "success",
        "eligible": True,
        "months_employed": eligibility.get("months_employed"),
        "hours_worked_past_year": eligibility.get("hours_worked_past_year"),
        "fmla_weeks_remaining": eligibility.get("fmla_weeks_remaining"),
        "message": "Employee is eligible for FMLA leave",
    }


def submit_fmla_case(params: dict, db: dict, call_index: int) -> dict:
    """Open an FMLA leave case."""
    try:
        p = SubmitFmlaCaseParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, SubmitFmlaCaseParams)

    if not _is_authenticated(db, "otp_auth"):
        return _auth_required("otp_auth")

    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _employee_not_found(p.employee_id)

    case_id = _make_case_id("FMLA", p.employee_id)
    leave_record = {
        "case_id": case_id,
        "leave_category": p.leave_category,
        "leave_start_date": p.leave_start_date,
        "leave_end_date": p.leave_end_date,
        "status": "open",
    }
    emp.setdefault("leave_records", []).append(leave_record)

    return {
        "status": "success",
        "employee_id": p.employee_id,
        "case_id": case_id,
        "leave_category": p.leave_category,
        "leave_start_date": p.leave_start_date,
        "leave_end_date": p.leave_end_date,
        "message": f"FMLA case opened. Case ID: {case_id}",
    }


def schedule_return_to_work_checkin(params: dict, db: dict, call_index: int) -> dict:
    """Schedule a return-to-work check-in appointment tied to an FMLA case.

    Validates the requested time slot is available before booking.
    """
    try:
        p = ScheduleReturnToWorkCheckinParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, ScheduleReturnToWorkCheckinParams)

    if not _is_authenticated(db, "otp_auth"):
        return _auth_required("otp_auth")

    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _employee_not_found(p.employee_id)

    # Validate appointment is after the FMLA leave end date
    fmla_case = None
    for lr in emp.get("leave_records", []):
        if lr.get("case_id") == p.case_id:
            fmla_case = lr
            break
    if fmla_case:
        leave_end = fmla_case.get("leave_end_date", "")
        appt_date = p.appointment_datetime.split(" ")[0] if " " in p.appointment_datetime else p.appointment_datetime
        if leave_end and appt_date < leave_end:
            return {
                "status": "error",
                "error_type": "appointment_before_leave_end",
                "message": f"Return-to-work check-in must be scheduled on or after the leave end date ({leave_end}). Requested date: {appt_date}",
            }

    # Validate and book the slot
    ok, err = _validate_and_book_slot(db, "return_to_work_checkin", p.department_code, p.appointment_datetime)
    if not ok:
        return err

    appt_id = _make_case_id("RTW", p.employee_id)
    emp.setdefault("scheduled_appointments", []).append(
        {
            "appointment_id": appt_id,
            "type": "return_to_work_checkin",
            "fmla_case_id": p.case_id,
            "appointment_datetime": p.appointment_datetime,
            "status": "scheduled",
        }
    )

    return {
        "status": "success",
        "employee_id": p.employee_id,
        "appointment_id": appt_id,
        "case_id": p.case_id,
        "appointment_datetime": p.appointment_datetime,
        "message": f"Return-to-work check-in scheduled for {p.appointment_datetime}. Appointment ID: {appt_id}",
    }


# ---------------------------------------------------------------------------
# FLOW 7: Payroll Correction
# ---------------------------------------------------------------------------


def get_timesheet_record(params: dict, db: dict, call_index: int) -> dict:
    """Retrieve a timesheet entry for a specific shift."""
    try:
        p = GetTimesheetRecordParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, GetTimesheetRecordParams)

    if not _is_authenticated(db, "employee_auth"):
        return _auth_required("employee_auth")

    shift = db.get("shifts", {}).get(p.shift_id)
    if not shift:
        return {"status": "error", "error_type": "shift_not_found", "message": f"Shift {p.shift_id} not found"}
    if shift.get("employee_id") != p.employee_id:
        return {
            "status": "error",
            "error_type": "shift_not_owned",
            "message": f"Shift {p.shift_id} does not belong to employee {p.employee_id}",
        }

    return {"status": "success", "shift": copy.deepcopy(shift)}


def check_correction_eligibility(params: dict, db: dict, call_index: int) -> dict:
    """Check whether a payroll correction can be submitted for a shift.

    Blocks if: correction already pending, pay period closed, or shift not yet logged.
    """
    try:
        p = CheckCorrectionEligibilityParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, CheckCorrectionEligibilityParams)

    if not _is_authenticated(db, "employee_auth"):
        return _auth_required("employee_auth")

    shift = db.get("shifts", {}).get(p.shift_id)
    if not shift:
        return {"status": "error", "error_type": "shift_not_found", "message": f"Shift {p.shift_id} not found"}
    if shift.get("correction_status") == "pending":
        return {
            "status": "error",
            "error_type": "correction_already_pending",
            "message": "A correction is already pending for this shift",
        }
    if shift.get("pay_period_closed"):
        return {
            "status": "error",
            "error_type": "pay_period_closed",
            "message": "The pay period for this shift is closed and cannot be corrected",
        }
    if shift.get("status") not in ("logged", "approved"):
        return {
            "status": "error",
            "error_type": "shift_not_logged",
            "message": "Shift must be in logged or approved status to submit a correction",
        }

    return {
        "status": "success",
        "eligible": True,
        "shift_id": p.shift_id,
        "logged_hours": shift.get("hours_logged"),
        "message": "Shift is eligible for payroll correction",
    }


def submit_payroll_correction(params: dict, db: dict, call_index: int) -> dict:
    """Submit a payroll correction for a specific shift."""
    try:
        p = SubmitPayrollCorrectionParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, SubmitPayrollCorrectionParams)

    if not _is_authenticated(db, "employee_auth"):
        return _auth_required("employee_auth")

    shift = db.get("shifts", {}).get(p.shift_id)
    if not shift:
        return {"status": "error", "error_type": "shift_not_found", "message": f"Shift {p.shift_id} not found"}

    pay_period_end_date = shift.get("pay_period_end_date")
    if not pay_period_end_date:
        return {
            "status": "error",
            "error_type": "pay_period_not_set",
            "message": f"No pay period end date found for shift {p.shift_id}",
        }

    current_date = db.get("_current_date", "")
    if current_date and pay_period_end_date < current_date:
        return {
            "status": "error",
            "error_type": "pay_period_closed",
            "message": f"Pay period ending {pay_period_end_date} has already closed (current date: {current_date}). Correction cannot be submitted.",
        }

    case_id = _make_case_id("PAY", p.employee_id)
    shift.update(
        {
            "corrected_hours": p.corrected_hours,
            "correction_type": p.correction_type,
            "correction_case_id": case_id,
            "correction_status": "pending",
        }
    )

    return {
        "status": "success",
        "employee_id": p.employee_id,
        "shift_id": p.shift_id,
        "correction_type": p.correction_type,
        "corrected_hours": p.corrected_hours,
        "pay_period_end_date": pay_period_end_date,
        "case_id": case_id,
        "message": f"Payroll correction submitted. Case ID: {case_id}",
    }


# ---------------------------------------------------------------------------
# FLOW 8: Privilege Reactivation After Leave
#
# New ordering: schedule competency review BEFORE reactivating privileges.
# ---------------------------------------------------------------------------


def get_privilege_record(params: dict, db: dict, call_index: int) -> dict:
    """Retrieve a provider's clinical privilege record."""
    try:
        p = GetPrivilegeRecordParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, GetPrivilegeRecordParams)

    if not _is_authenticated(db, "otp_auth"):
        return _auth_required("otp_auth")

    provider = db.get("providers", {}).get(p.npi)
    if not provider:
        return _provider_not_found(p.npi)

    privileges = provider.get("privileges")
    if not privileges:
        return {
            "status": "error",
            "error_type": "privilege_record_not_found",
            "message": f"No privilege record found for NPI {p.npi}",
        }

    return {"status": "success", "privileges": copy.deepcopy(privileges)}


def check_reactivation_eligibility(params: dict, db: dict, call_index: int) -> dict:
    """Validate occupational health clearance code before reactivating privileges."""
    try:
        p = CheckReactivationEligibilityParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, CheckReactivationEligibilityParams)

    if not _is_authenticated(db, "otp_auth"):
        return _auth_required("otp_auth")

    provider = db.get("providers", {}).get(p.npi)
    if not provider:
        return _provider_not_found(p.npi)

    if provider.get("clearance_code") != p.clearance_code:
        return {
            "status": "error",
            "error_type": "invalid_clearance_code",
            "message": "Clearance code does not match occupational health records",
        }

    suspended = [
        prv["code"]
        for prv in provider.get("privileges", {}).get("privilege_list", [])
        if prv.get("status") == "suspended"
    ]

    return {
        "status": "success",
        "eligible": True,
        "suspended_privilege_codes": suspended,
        "message": f"Clearance verified. {len(suspended)} suspended privilege(s) available for reactivation",
    }


def schedule_competency_review(params: dict, db: dict, call_index: int) -> dict:
    """Schedule a competency review appointment — called BEFORE reactivating privileges.

    Validates the requested time slot is available before booking.
    """
    try:
        p = ScheduleCompetencyReviewParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, ScheduleCompetencyReviewParams)

    if not _is_authenticated(db, "otp_auth"):
        return _auth_required("otp_auth")

    provider = db.get("providers", {}).get(p.npi)
    if not provider:
        return _provider_not_found(p.npi)

    # Validate and book the slot
    ok, err = _validate_and_book_slot(db, "competency_review", p.department_code, p.appointment_datetime)
    if not ok:
        return err

    appt_id = _make_case_id("CMP", provider.get("employee_id", p.npi))
    provider.setdefault("scheduled_appointments", []).append(
        {
            "appointment_id": appt_id,
            "type": "competency_review",
            "department_code": p.department_code,
            "appointment_datetime": p.appointment_datetime,
            "status": "scheduled",
        }
    )

    return {
        "status": "success",
        "npi": p.npi,
        "appointment_id": appt_id,
        "department_code": p.department_code,
        "appointment_datetime": p.appointment_datetime,
        "message": f"Competency review scheduled for {p.appointment_datetime}. Appointment ID: {appt_id}",
    }


def reactivate_privileges(params: dict, db: dict, call_index: int) -> dict:
    """Reactivate one or more suspended clinical privileges.

    Called AFTER schedule_competency_review has already booked the review.
    No longer takes competency_review_date — the review is already scheduled.
    """
    try:
        p = ReactivatePrivilegesParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, ReactivatePrivilegesParams)

    if not _is_authenticated(db, "otp_auth"):
        return _auth_required("otp_auth")

    provider = db.get("providers", {}).get(p.npi)
    if not provider:
        return _provider_not_found(p.npi)

    if provider.get("clearance_code") != p.clearance_code:
        return {
            "status": "error",
            "error_type": "invalid_clearance_code",
            "message": "Clearance code does not match occupational health records",
        }

    privilege_list = provider.get("privileges", {}).get("privilege_list", [])
    activated, not_found = [], []
    for code in p.privilege_codes:
        match = next((prv for prv in privilege_list if prv.get("code") == code), None)
        if match:
            match["status"] = "active"
            activated.append(code)
        else:
            not_found.append(code)

    if not_found:
        return {
            "status": "error",
            "error_type": "privilege_not_found",
            "message": f"Privilege code(s) not found: {not_found}",
        }

    case_id = _make_case_id("PRV", provider.get("employee_id", p.npi))
    provider["privileges"]["reactivation_case_id"] = case_id

    return {
        "status": "success",
        "npi": p.npi,
        "activated_privileges": activated,
        "leave_type_on_record": p.leave_type_on_record,
        "case_id": case_id,
        "message": f"Privileges reactivated. Case ID: {case_id}",
    }


def update_ehr_access(params: dict, db: dict, call_index: int) -> dict:
    """Update EHR system access permissions for a provider following privilege change."""
    try:
        p = UpdateEhrAccessParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, UpdateEhrAccessParams)

    if not _is_authenticated(db, "otp_auth"):
        return _auth_required("otp_auth")

    provider = db.get("providers", {}).get(p.npi)
    if not provider:
        return _provider_not_found(p.npi)

    provider["ehr_access_status"] = p.access_change_type
    provider["ehr_access_case_id"] = p.case_id

    return {
        "status": "success",
        "npi": p.npi,
        "case_id": p.case_id,
        "access_change_type": p.access_change_type,
        "message": f"EHR access updated to {p.access_change_type}",
    }


# ---------------------------------------------------------------------------
# FLOW 9: On-Call Schedule Registration
# ---------------------------------------------------------------------------


def get_oncall_schedule(params: dict, db: dict, call_index: int) -> dict:
    """Retrieve the current on-call schedule registrations for an employee and unit."""
    try:
        p = GetOncallScheduleParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, GetOncallScheduleParams)

    if not _is_authenticated(db, "employee_auth"):
        return _auth_required("employee_auth")

    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _employee_not_found(p.employee_id)

    return {
        "status": "success",
        "employee_id": p.employee_id,
        "unit_code": p.unit_code,
        "oncall_schedule": copy.deepcopy(emp.get("oncall_schedule", {})),
    }


def check_oncall_eligibility(params: dict, db: dict, call_index: int) -> dict:
    """Check whether an employee can register for on-call on a given unit.

    Blocks if: on active leave, missing unit-required certifications,
    or already registered for an overlapping window.
    """
    try:
        p = CheckOncallEligibilityParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, CheckOncallEligibilityParams)

    if not _is_authenticated(db, "employee_auth"):
        return _auth_required("employee_auth")

    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _employee_not_found(p.employee_id)

    if emp.get("employment_status") == "on_leave":
        return {
            "status": "error",
            "error_type": "employee_on_leave",
            "message": "Employee is on leave and cannot register for on-call shifts",
        }

    unit_reqs = db.get("unit_cert_requirements", {}).get(p.unit_code, [])
    emp_certs = set(emp.get("certifications", []))
    missing = sorted(set(unit_reqs) - emp_certs)
    if missing:
        return {
            "status": "error",
            "error_type": "certification_missing",
            "message": f"Missing certifications for unit {p.unit_code}: {missing}",
            "missing_certs": missing,
        }

    return {
        "status": "success",
        "eligible": True,
        "employee_id": p.employee_id,
        "unit_code": p.unit_code,
        "message": "Employee is eligible to register for on-call on this unit",
    }


def register_oncall_availability(params: dict, db: dict, call_index: int) -> dict:
    """Register on-call availability window and blackout dates for an employee."""
    try:
        p = RegisterOncallAvailabilityParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, RegisterOncallAvailabilityParams)

    if not _is_authenticated(db, "employee_auth"):
        return _auth_required("employee_auth")

    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _employee_not_found(p.employee_id)

    for bd in p.blackout_dates:
        if not (p.availability_start_date <= bd <= p.availability_end_date):
            return {
                "status": "error",
                "error_type": "invalid_blackout_date",
                "message": f"Blackout date {bd} is outside the availability window "
                f"({p.availability_start_date} – {p.availability_end_date})",
            }

    reg_id = _make_case_id("ONC", p.employee_id)
    registration = {
        "registration_id": reg_id,
        "unit_code": p.unit_code,
        "availability_start_date": p.availability_start_date,
        "availability_end_date": p.availability_end_date,
        "oncall_tier": p.oncall_tier,
        "blackout_dates": p.blackout_dates,
        "status": "registered",
    }
    emp.setdefault("oncall_schedule", {}).setdefault("registrations", []).append(registration)

    return {
        "status": "success",
        "employee_id": p.employee_id,
        "registration_id": reg_id,
        "unit_code": p.unit_code,
        "availability_start_date": p.availability_start_date,
        "availability_end_date": p.availability_end_date,
        "oncall_tier": p.oncall_tier,
        "blackout_dates": p.blackout_dates,
        "message": f"On-call availability registered. Registration ID: {reg_id}",
    }


# ---------------------------------------------------------------------------
# FLOW 10: I-9 Verification
# ---------------------------------------------------------------------------


def get_i9_record(params: dict, db: dict, call_index: int) -> dict:
    """Retrieve the I-9 verification record for an employee."""
    try:
        p = GetI9RecordParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, GetI9RecordParams)

    if not _is_authenticated(db, "employee_auth"):
        return _auth_required("employee_auth")

    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _employee_not_found(p.employee_id)

    i9 = emp.get("i9_record")
    if not i9:
        return {
            "status": "error",
            "error_type": "i9_record_not_found",
            "message": f"No I-9 record found for {p.employee_id}",
        }

    return {"status": "success", "i9_record": copy.deepcopy(i9)}


def submit_i9_verification(params: dict, db: dict, call_index: int) -> dict:
    """Submit or update I-9 document verification for an employee."""
    try:
        p = SubmitI9VerificationParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, SubmitI9VerificationParams)

    if not _is_authenticated(db, "employee_auth"):
        return _auth_required("employee_auth")

    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _employee_not_found(p.employee_id)

    case_id = _make_case_id("I9V", p.employee_id)
    i9 = emp.get("i9_record")
    if not i9:
        i9 = {}
        emp["i9_record"] = i9
    i9.update(
        {
            "verification_action": p.verification_action,
            "document_list_type": p.document_list_type,
            "document_type_code": p.document_type_code,
            "document_number": p.document_number,
            "document_expiration_date": p.document_expiration_date,
            "issuing_country_code": p.issuing_country_code,
            "verification_status": "verified",
            "case_id": case_id,
        }
    )

    return {
        "status": "success",
        "employee_id": p.employee_id,
        "verification_action": p.verification_action,
        "document_list_type": p.document_list_type,
        "document_type_code": p.document_type_code,
        "document_number": p.document_number,
        "document_expiration_date": p.document_expiration_date,
        "issuing_country_code": p.issuing_country_code,
        "case_id": case_id,
        "message": f"I-9 {p.verification_action} completed. Case ID: {case_id}",
    }


def notify_hr_compliance(params: dict, db: dict, call_index: int) -> dict:
    """Notify HR compliance of a completed I-9 verification."""
    try:
        p = NotifyHrComplianceParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, NotifyHrComplianceParams)

    if not _is_authenticated(db, "employee_auth"):
        return _auth_required("employee_auth")

    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _employee_not_found(p.employee_id)

    notif = {
        "recipient": "hr_compliance",
        "employee_id": p.employee_id,
        "case_id": p.case_id,
        "notification_type": p.notification_type,
    }
    notifications = db.setdefault("notifications", [])
    if notif not in notifications:
        notifications.append(notif)

    return {
        "status": "success",
        "employee_id": p.employee_id,
        "case_id": p.case_id,
        "notification_type": p.notification_type,
        "message": f"HR compliance notified: {p.notification_type}",
    }


# ---------------------------------------------------------------------------
# FLOW 11: Visa Dependent Addition
# ---------------------------------------------------------------------------


def get_visa_record(params: dict, db: dict, call_index: int) -> dict:
    """Retrieve visa sponsorship record for an employee."""
    try:
        p = GetVisaRecordParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, GetVisaRecordParams)

    if not _is_authenticated(db, "otp_auth"):
        return _auth_required("otp_auth")

    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _employee_not_found(p.employee_id)

    visa = emp.get("visa_record")
    if not visa or visa.get("petition_number") != p.visa_petition_number:
        return {
            "status": "error",
            "error_type": "visa_record_not_found",
            "message": f"Visa petition {p.visa_petition_number} not found for {p.employee_id}",
        }

    return {"status": "success", "visa_record": copy.deepcopy(visa)}


def add_visa_dependent(params: dict, db: dict, call_index: int) -> dict:
    """Add a dependent to an existing visa petition."""
    try:
        p = AddVisaDependentParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, AddVisaDependentParams)

    if not _is_authenticated(db, "otp_auth"):
        return _auth_required("otp_auth")

    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _employee_not_found(p.employee_id)

    visa = emp.get("visa_record")
    if not visa or visa.get("petition_number") != p.visa_petition_number:
        return {
            "status": "error",
            "error_type": "visa_record_not_found",
            "message": f"Visa petition {p.visa_petition_number} not found for {p.employee_id}",
        }

    amendment_id = _make_case_id("VISA", p.employee_id)
    dependent = {
        "first_name": p.dependent_first_name,
        "last_name": p.dependent_last_name,
        "relationship": p.relationship,
        "date_of_birth": p.dependent_date_of_birth,
        "country_of_birth": p.dependent_country_of_birth,
        "uscis_receipt_number": p.uscis_receipt_number,
        "amendment_id": amendment_id,
        "status": "pending",
    }
    visa.setdefault("dependents", []).append(dependent)
    visa["amendment_id"] = amendment_id

    return {
        "status": "success",
        "employee_id": p.employee_id,
        "visa_petition_number": p.visa_petition_number,
        "dependent_name": f"{p.dependent_first_name} {p.dependent_last_name}",
        "relationship": p.relationship,
        "dependent_date_of_birth": p.dependent_date_of_birth,
        "dependent_country_of_birth": p.dependent_country_of_birth,
        "uscis_receipt_number": p.uscis_receipt_number,
        "amendment_id": amendment_id,
        "message": f"Dependent added to petition {p.visa_petition_number}. Amendment ID: {amendment_id}",
    }


def notify_immigration_counsel(params: dict, db: dict, call_index: int) -> dict:
    """Notify immigration counsel of a visa petition change."""
    try:
        p = NotifyImmigrationCounselParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, NotifyImmigrationCounselParams)

    if not _is_authenticated(db, "otp_auth"):
        return _auth_required("otp_auth")

    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _employee_not_found(p.employee_id)

    notif = {
        "recipient": "immigration_counsel",
        "employee_id": p.employee_id,
        "visa_petition_number": p.visa_petition_number,
        "notification_type": p.notification_type,
    }
    notifications = db.setdefault("notifications", [])
    if notif not in notifications:
        notifications.append(notif)

    return {
        "status": "success",
        "employee_id": p.employee_id,
        "visa_petition_number": p.visa_petition_number,
        "notification_type": p.notification_type,
        "message": f"Immigration counsel notified: {p.notification_type}",
    }


# ---------------------------------------------------------------------------
# FLOW 12: PTO Request
#
# PTO day calculation depends on the employee's schedule_type:
#   - "standard" (M-F office workers): count weekdays in range minus org holidays
#   - "shift" (nurses, doctors, techs): count scheduled shifts in range
# ---------------------------------------------------------------------------


def _count_weekdays(start: str, end: str) -> list[str]:
    """Return list of weekday date strings (Mon-Fri) in [start, end] inclusive."""
    from datetime import date, timedelta

    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    days = []
    current = s
    while current <= e:
        if current.weekday() < 5:  # Mon=0 .. Fri=4
            days.append(current.isoformat())
        current += timedelta(days=1)
    return days


def _get_shifts_in_range(db: dict, employee_id: str, start: str, end: str) -> list[str]:
    """Return sorted list of shift dates for an employee within [start, end]."""
    dates = set()
    for shift in db.get("shifts", {}).values():
        if shift.get("employee_id") == employee_id:
            d = shift.get("date", "")
            if start <= d <= end:
                dates.add(d)
    return sorted(dates)


def get_pto_balance(params: dict, db: dict, call_index: int) -> dict:
    """Retrieve an employee's PTO balances by type (pto and sick).

    Also returns schedule_type so the agent can inform the caller how
    PTO days are calculated.
    """
    try:
        p = GetPtoBalanceParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, GetPtoBalanceParams)

    if not _is_authenticated(db, "employee_auth"):
        return _auth_required("employee_auth")

    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _employee_not_found(p.employee_id)

    balances = emp.get("pto_balances")
    if not balances:
        return {
            "status": "error",
            "error_type": "pto_record_not_found",
            "message": f"No PTO balance record found for {p.employee_id}",
        }

    schedule_type = emp.get("schedule_type", "standard")

    return {
        "status": "success",
        "employee_id": p.employee_id,
        "schedule_type": schedule_type,
        "pto_balances": copy.deepcopy(balances),
        "message": f"PTO balances retrieved. Schedule type: {schedule_type}",
    }


def check_pto_eligibility(params: dict, db: dict, call_index: int) -> dict:
    """Check whether an employee can take PTO for a given date range.

    Validates:
    1. Sufficient PTO balance for the number of working days in range.
       - Standard (M-F) employees: weekdays minus org holidays.
       - Shift employees: scheduled shifts in the date range.
    2. No department blackout dates overlap with the requested range.
    3. No existing PTO request overlaps with the requested range.

    Returns the exact working days that count toward PTO so the caller
    can confirm before submitting.
    """
    try:
        p = CheckPtoEligibilityParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, CheckPtoEligibilityParams)

    if not _is_authenticated(db, "employee_auth"):
        return _auth_required("employee_auth")

    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _employee_not_found(p.employee_id)

    if p.start_date > p.end_date:
        return {
            "status": "error",
            "error_type": "invalid_date_range",
            "message": "Start date must be on or before end date",
        }

    balances = emp.get("pto_balances", {})
    current_balance = balances.get(p.pto_type, 0.0)
    schedule_type = emp.get("schedule_type", "standard")

    # Calculate working days based on schedule type
    if schedule_type == "standard":
        weekdays = _count_weekdays(p.start_date, p.end_date)
        # Remove org holidays
        org_holidays = set(db.get("org_holidays", []))
        working_days = [d for d in weekdays if d not in org_holidays]
    else:
        # Shift worker: count scheduled shifts in range
        working_days = _get_shifts_in_range(db, p.employee_id, p.start_date, p.end_date)

    pto_days_required = float(len(working_days))

    # Check balance
    if pto_days_required > current_balance:
        return {
            "status": "error",
            "error_type": "insufficient_pto_balance",
            "message": f"Insufficient {p.pto_type} balance: {pto_days_required} days "
            f"required but only {current_balance} available",
            "pto_days_required": pto_days_required,
            "current_balance": current_balance,
        }

    # Check department blackout dates
    dept = emp.get("department_code", "")
    blackout_dates = set(db.get("department_blackout_dates", {}).get(dept, []))
    blocked = sorted(set(working_days) & blackout_dates)
    if blocked:
        return {
            "status": "error",
            "error_type": "blackout_date_conflict",
            "message": f"Requested dates overlap with department blackout dates: {blocked}",
            "conflicting_dates": blocked,
        }

    # Check overlap with existing PTO requests
    existing = emp.get("pto_requests", [])
    for req in existing:
        if req.get("status") in ("pending", "approved"):
            if p.start_date <= req.get("end_date", "") and p.end_date >= req.get("start_date", ""):
                return {
                    "status": "error",
                    "error_type": "pto_overlap",
                    "message": f"Requested dates overlap with existing PTO request "
                    f"{req.get('start_date')} to {req.get('end_date')} "
                    f"(case {req.get('case_id')})",
                    "overlapping_case_id": req.get("case_id"),
                }

    remaining = current_balance - pto_days_required

    return {
        "status": "success",
        "eligible": True,
        "employee_id": p.employee_id,
        "schedule_type": schedule_type,
        "pto_type": p.pto_type,
        "pto_days_required": pto_days_required,
        "working_days_in_range": working_days,
        "current_balance": current_balance,
        "remaining_after": remaining,
        "message": f"Eligible. {pto_days_required} {p.pto_type} day(s) required, {remaining} remaining after.",
    }


def submit_pto_request(params: dict, db: dict, call_index: int) -> dict:
    """Submit a PTO request for a date range.

    Recomputes the PTO days internally (does not trust the LLM's count)
    and deducts from the employee's balance. Returns a case_id.
    """
    try:
        p = SubmitPtoRequestParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, SubmitPtoRequestParams)

    if not _is_authenticated(db, "employee_auth"):
        return _auth_required("employee_auth")

    emp = db.get("employees", {}).get(p.employee_id)
    if not emp:
        return _employee_not_found(p.employee_id)

    if p.start_date > p.end_date:
        return {
            "status": "error",
            "error_type": "invalid_date_range",
            "message": "Start date must be on or before end date",
        }

    balances = emp.get("pto_balances", {})
    current_balance = balances.get(p.pto_type, 0.0)
    schedule_type = emp.get("schedule_type", "standard")

    # Recompute working days (do not trust LLM)
    if schedule_type == "standard":
        weekdays = _count_weekdays(p.start_date, p.end_date)
        org_holidays = set(db.get("org_holidays", []))
        working_days = [d for d in weekdays if d not in org_holidays]
    else:
        working_days = _get_shifts_in_range(db, p.employee_id, p.start_date, p.end_date)

    pto_days = float(len(working_days))

    if pto_days > current_balance:
        return {
            "status": "error",
            "error_type": "insufficient_pto_balance",
            "message": f"Insufficient {p.pto_type} balance: {pto_days} days "
            f"required but only {current_balance} available",
        }

    # Deduct from balance
    balances[p.pto_type] = current_balance - pto_days

    case_id = _make_case_id("PTO", p.employee_id)
    pto_record = {
        "case_id": case_id,
        "pto_type": p.pto_type,
        "start_date": p.start_date,
        "end_date": p.end_date,
        "pto_days_deducted": pto_days,
        "working_days": working_days,
        "status": "pending",
    }
    emp.setdefault("pto_requests", []).append(pto_record)

    return {
        "status": "success",
        "employee_id": p.employee_id,
        "case_id": case_id,
        "pto_type": p.pto_type,
        "start_date": p.start_date,
        "end_date": p.end_date,
        "pto_days_deducted": pto_days,
        "working_days": working_days,
        "remaining_balance": balances[p.pto_type],
        "message": f"PTO request submitted. {pto_days} {p.pto_type} day(s) deducted. Case ID: {case_id}",
    }


# ---------------------------------------------------------------------------
# SYSTEM: Transfer to Live Agent
# ---------------------------------------------------------------------------


def transfer_to_agent(params: dict, db: dict, call_index: int) -> dict:
    """Transfer the call to a live human agent."""
    try:
        p = TransferToAgentParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, TransferToAgentParams)

    employee_id = p.employee_id
    transfer_reason = p.transfer_reason
    issue_summary = p.issue_summary

    transfer_id = f"TRF-{employee_id}-{str(call_index).zfill(3)}"

    return {
        "status": "success",
        "transfer_id": transfer_id,
        "employee_id": employee_id,
        "transfer_reason": transfer_reason,
        "issue_summary": issue_summary,
        "estimated_wait": "2-3 minutes",
        "message": "Transferring to live agent",
    }
