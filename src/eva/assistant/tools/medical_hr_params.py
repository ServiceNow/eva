"""Pydantic parameter models and enums for medical HR tool functions.

Each tool function has a corresponding *Params model that validates and types
its incoming ``params`` dict. Call ``Model.model_validate(params)`` at the top
of the tool function and catch ``ValidationError`` to produce a standard
``{"status": "error", ...}`` response for bad LLM-supplied inputs.

Enums use ``StrEnum`` so values serialise to plain strings in JSON and compare
equal to their string counterparts.

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

from enum import StrEnum
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

# ---------------------------------------------------------------------------
# Shared ID / format annotated types
# ---------------------------------------------------------------------------

EmployeeIdStr = Annotated[
    str,
    Field(pattern=r"^EMP\d{6}$", description="EMP followed by 6 digits", examples=["EMP048271"]),
]

NpiStr = Annotated[
    str,
    Field(pattern=r"^\d{10}$", description="10-digit NPI number", examples=["1487392045"]),
]

FacilityCodeStr = Annotated[
    str,
    Field(
        pattern=r"^[A-Z]{2,4}-\d{2}[A-Z]$",
        description="Facility code: 2-4 uppercase letters, dash, 2 digits, 1 uppercase letter",
        examples=["MGH-04B"],
    ),
]

PinStr = Annotated[
    str,
    Field(pattern=r"^\d{4}$", description="4-digit PIN", examples=["7291"]),
]

OtpStr = Annotated[
    str,
    Field(pattern=r"^\d{6}$", description="6-digit OTP code", examples=["483920"]),
]

DateStr = Annotated[
    str,
    Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD", examples=["2026-05-01"]),
]

AppointmentDatetimeStr = Annotated[
    str,
    Field(
        pattern=r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$",
        description="Appointment date and time in YYYY-MM-DD HH:MM format",
        examples=["2026-05-01 09:00", "2026-06-15 14:00"],
    ),
]

DeaNumberStr = Annotated[
    str,
    Field(
        pattern=r"^[A-Z]{2}\d{7}$",
        description="DEA number: 2 uppercase letters followed by 7 digits",
        examples=["BK4729183"],
    ),
]

StateLicenseStr = Annotated[
    str,
    Field(
        pattern=r"^[A-Z]{2}-[A-Z]{2,4}-\d{6,8}$",
        description="State license: state code dash license type dash number",
        examples=["MA-RN-004821"],
    ),
]

PolicyNumberStr = Annotated[
    str,
    Field(
        pattern=r"^POL-\d{4}-[A-Z0-9]{6}$",
        description="Malpractice policy number in format POL-YYYY-XXXXXX",
        examples=["POL-2024-AX7731"],
    ),
]

ShiftIdStr = Annotated[
    str,
    Field(
        pattern=r"^SHF-\d{8}-\d{4}$",
        description="Shift ID in format SHF-YYYYMMDD-HHMM",
        examples=["SHF-20260501-0700"],
    ),
]

CaseIdStr = Annotated[
    str,
    Field(
        pattern=r"^CASE-[A-Z0-9]{2,6}-\d{6}$",
        description="HR case ID in format CASE-PREFIX-6digits",
        examples=["CASE-FMLA-048271", "CASE-LIC-048271", "CASE-I9V-072948"],
    ),
]

PrivilegeCodeStr = Annotated[
    str,
    Field(
        pattern=r"^PRV-[A-Z]{2,6}-\d{3}$",
        description="Privilege code in format PRV-CATEGORY-3digits",
        examples=["PRV-CARD-001", "PRV-ICU-003"],
    ),
]

UnitCodeStr = Annotated[
    str,
    Field(
        pattern=r"^\d{1,2}[A-Z]-[A-Z]{2,6}$",
        description="Unit code: floor+wing dash unit type",
        examples=["4B-ICU", "2A-MED", "5A-SURG"],
    ),
]

CertCodeStr = Annotated[
    str,
    Field(
        pattern=r"^(ACLS|BLS|PALS|TNCC|NRP|CCRN|CEN|NIHSS)$",
        description="Certification code: one of ACLS, BLS, PALS, TNCC, NRP, CCRN, CEN, NIHSS",
        examples=["ACLS", "BLS"],
    ),
]

VisaPetitionStr = Annotated[
    str,
    Field(
        pattern=r"^[A-Z]{3}\d{10}$",
        description="Visa petition number: 3 uppercase letters followed by 10 digits",
        examples=["WAC2512045678"],
    ),
]

UsciReceiptStr = Annotated[
    str,
    Field(
        pattern=r"^[A-Z]{3}\d{10}$",
        description="USCIS receipt number: 3 uppercase letters followed by 10 digits. Different from the visa petition number.",
        examples=["IOE0912345678"],
    ),
]

OnboardingTaskCodeStr = Annotated[
    str,
    Field(
        pattern=r"^(BLS_UPLOAD|I9_VERIFY|BADGE_PICKUP|OCC_HEALTH|HIPAA_TRAIN|DRUG_SCREEN|TB_TEST|ORIENTATION)$",
        description="Onboarding task code: one of BLS_UPLOAD, I9_VERIFY, BADGE_PICKUP, OCC_HEALTH, HIPAA_TRAIN, DRUG_SCREEN, TB_TEST, ORIENTATION",
        examples=["BLS_UPLOAD", "HIPAA_TRAIN"],
    ),
]

I9DocumentTypeStr = Annotated[
    str,
    Field(
        pattern=r"^(LIST_A|LIST_B|LIST_C)$",
        description="I-9 document list type: LIST_A (single document proving identity + work auth), LIST_B (identity only), or LIST_C (work authorization only)",
        examples=["LIST_A"],
    ),
]

I9DocumentCodeStr = Annotated[
    str,
    Field(
        pattern=r"^(US_PASSPORT|PERM_RESIDENT_CARD|EAD|DRIVERS_LICENSE|STATE_ID|SSN_CARD|BIRTH_CERT)$",
        description="I-9 document type code: US_PASSPORT, PERM_RESIDENT_CARD, EAD, DRIVERS_LICENSE, STATE_ID, SSN_CARD, or BIRTH_CERT",
        examples=["US_PASSPORT", "DRIVERS_LICENSE"],
    ),
]

CountryCodeStr = Annotated[
    str,
    Field(pattern=r"^[A-Z]{2}$", description="ISO 3166-1 alpha-2 country code", examples=["US", "IN", "MX"]),
]

StateCodeStr = Annotated[
    str,
    Field(
        pattern=r"^(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC|PR|GU|VI|AS|MP)$",
        description="2-letter US state or territory abbreviation",
        examples=["MA", "CA", "NY"],
    ),
]

DepartmentCodeStr = Annotated[
    str,
    Field(
        pattern=r"^(CARDIOLOGY|EMERGENCY|ONCOLOGY|SURGERY|NEUROLOGY|PEDIATRICS|RADIOLOGY|PATHOLOGY|ORTHOPEDICS|OBSTETRICS)$",
        description="Department code: CARDIOLOGY, EMERGENCY, ONCOLOGY, SURGERY, NEUROLOGY, PEDIATRICS, RADIOLOGY, PATHOLOGY, ORTHOPEDICS, or OBSTETRICS",
        examples=["CARDIOLOGY", "EMERGENCY"],
    ),
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ExtensionType(StrEnum):
    provisional = "provisional"
    supervised = "supervised"


class LeaveCategory(StrEnum):
    """FMLA leave categories — each is legally distinct:
    - employee_medical_condition: the employee themselves has a serious health condition
    - family_member_serious_illness: caring for a spouse, child, or parent with a serious health condition
    - bonding: birth, adoption, or foster placement of a child (within 12 months of the event)
    - military_exigency: qualifying exigency arising from a family member's active military duty
    """

    employee_medical_condition = "employee_medical_condition"
    family_member_serious_illness = "family_member_serious_illness"
    bonding = "bonding"
    military_exigency = "military_exigency"


class PayrollCorrectionType(StrEnum):
    on_call_hours = "on_call_hours"
    overtime_hours = "overtime_hours"
    differential_missed = "differential_missed"


class OnCallTier(StrEnum):
    primary = "primary"
    backup = "backup"


class PtoType(StrEnum):
    """PTO balance types:
    - pto: general paid time off (covers vacation and personal days)
    - sick: sick leave (separate accrual and legal protections)
    """

    pto = "pto"
    sick = "sick"


class LeaveTypeOnRecord(StrEnum):
    """Administrative leave category the provider was on — mutually exclusive:
    - medical_leave: employer-approved medical leave (not FMLA-protected)
    - personal_leave: approved personal or unpaid leave (not medical, not FMLA)
    - fmla_leave: FMLA-protected leave (federally protected, with specific eligibility)
    """

    medical_leave = "medical_leave"
    personal_leave = "personal_leave"
    fmla_leave = "fmla_leave"


class DependentRelationship(StrEnum):
    spouse = "spouse"
    child = "child"
    domestic_partner = "domestic_partner"


class MalpracticeCarrier(StrEnum):
    proassurance = "proassurance"
    coverys = "coverys"
    the_doctors_company = "the_doctors_company"
    cna = "cna"
    zurich = "zurich"
    mag_mutual = "mag_mutual"


class TransferReason(StrEnum):
    """DEA transfer reasons — each is distinct:
    - facility_relocation: same role, moving to a different facility/state
    - role_change: new clinical role at a different facility
    - additional_practice_site: adding a second practice location while keeping the original
    """

    facility_relocation = "facility_relocation"
    role_change = "role_change"
    additional_practice_site = "additional_practice_site"


class I9VerificationAction(StrEnum):
    initial_verification = "initial_verification"
    reverification = "reverification"


class AppointmentType(StrEnum):
    """Types of schedulable appointments — each maps to a specific flow."""

    orientation_followup = "orientation_followup"
    return_to_work_checkin = "return_to_work_checkin"
    competency_review = "competency_review"


class CredentialingNotificationType(StrEnum):
    license_extension_submitted = "license_extension_submitted"
    malpractice_updated = "malpractice_updated"
    privilege_reactivation = "privilege_reactivation"


class ManagerNotificationType(StrEnum):
    shift_swap_confirmed = "shift_swap_confirmed"
    fmla_opened = "fmla_opened"
    payroll_correction_submitted = "payroll_correction_submitted"
    pto_request_submitted = "pto_request_submitted"


class HrComplianceNotificationType(StrEnum):
    """Maps directly from I9VerificationAction:
    - initial_verification → i9_verified
    - reverification → i9_reverified
    """

    i9_verified = "i9_verified"
    i9_reverified = "i9_reverified"


class EhrAccessChangeType(StrEnum):
    """EHR access levels — caller chooses based on their needs:
    - reactivate_full: full clinical access restored (all modules, all patient records)
    - reactivate_restricted: limited access (read-only or specific modules only)
    - suspend: remove access (used for departures, not typically caller-initiated)
    """

    reactivate_full = "reactivate_full"
    reactivate_restricted = "reactivate_restricted"
    suspend = "suspend"


class ImmigrationNotificationType(StrEnum):
    dependent_added = "dependent_added"


# ---------------------------------------------------------------------------
# Auth Params
# ---------------------------------------------------------------------------


class VerifyEmployeeAuthParams(BaseModel):
    """Standard Employee Auth — employee_id + date_of_birth."""

    employee_id: EmployeeIdStr
    date_of_birth: DateStr


class VerifyProviderAuthParams(BaseModel):
    """Credentialed Provider Auth — NPI + facility_code + PIN."""

    npi: NpiStr
    facility_code: FacilityCodeStr
    pin: PinStr


class InitiateOtpAuthParams(BaseModel):
    """OTP step 1 — send OTP SMS to phone on file."""

    employee_id: EmployeeIdStr


class VerifyOtpAuthParams(BaseModel):
    """OTP step 2 — verify the 6-digit code."""

    employee_id: EmployeeIdStr
    otp_code: OtpStr


# ---------------------------------------------------------------------------
# Shared lookup tools (used across multiple flows)
# ---------------------------------------------------------------------------


class GetProviderProfileParams(BaseModel):
    npi: NpiStr


class GetEmployeeRecordParams(BaseModel):
    employee_id: EmployeeIdStr


# ---------------------------------------------------------------------------
# Shared scheduling tools
# ---------------------------------------------------------------------------


class CheckAppointmentAvailabilityParams(BaseModel):
    """Check available time slots for a specific appointment type on a given date."""

    department_code: DepartmentCodeStr
    appointment_type: AppointmentType
    preferred_date: DateStr


# ---------------------------------------------------------------------------
# Flow 1: License Expiration Extension
# ---------------------------------------------------------------------------


class GetLicenseRecordParams(BaseModel):
    npi: NpiStr
    state_license_number: StateLicenseStr


class CheckExtensionEligibilityParams(BaseModel):
    npi: NpiStr
    state_license_number: StateLicenseStr


class SubmitLicenseExtensionParams(BaseModel):
    npi: NpiStr
    state_license_number: StateLicenseStr
    extension_type: ExtensionType
    extension_days: Literal[30, 60, 90] = Field(description="Extension duration in days: exactly 30, 60, or 90")
    supervising_physician_npi: Optional[NpiStr] = Field(
        default=None,
        description="NPI of the supervising physician. Required for supervised extensions, must be omitted for provisional.",
    )


class NotifyCredentialingCommitteeParams(BaseModel):
    npi: NpiStr
    case_id: CaseIdStr
    notification_type: CredentialingNotificationType


# ---------------------------------------------------------------------------
# Flow 2: Shift Swap
# ---------------------------------------------------------------------------


class GetShiftRecordParams(BaseModel):
    employee_id: EmployeeIdStr
    shift_id: ShiftIdStr


class CheckSwapEligibilityParams(BaseModel):
    employee_id: EmployeeIdStr
    shift_id: ShiftIdStr


class VerifyColleagueCertificationsParams(BaseModel):
    colleague_employee_id: EmployeeIdStr
    unit_code: UnitCodeStr


class ConfirmShiftSwapParams(BaseModel):
    employee_id: EmployeeIdStr
    shift_id: ShiftIdStr
    colleague_employee_id: EmployeeIdStr
    unit_code: UnitCodeStr


class NotifyDepartmentManagerParams(BaseModel):
    employee_id: EmployeeIdStr
    case_id: CaseIdStr
    notification_type: ManagerNotificationType


# ---------------------------------------------------------------------------
# Flow 3: Malpractice Coverage Update
# ---------------------------------------------------------------------------


class GetMalpracticeRecordParams(BaseModel):
    npi: NpiStr


class UpdateMalpracticeCoverageParams(BaseModel):
    npi: NpiStr
    new_carrier: MalpracticeCarrier
    new_policy_number: PolicyNumberStr
    per_occurrence_limit_usd: int = Field(
        description="Per-occurrence coverage limit in USD as an integer",
        examples=[1000000, 2000000],
    )
    aggregate_limit_usd: int = Field(
        description="Aggregate coverage limit in USD as an integer",
        examples=[3000000, 5000000],
    )
    effective_date: DateStr
    expiration_date: DateStr


# ---------------------------------------------------------------------------
# Flow 4: Onboarding Task Completion
# ---------------------------------------------------------------------------


class GetOnboardingChecklistParams(BaseModel):
    employee_id: EmployeeIdStr


class CompleteOnboardingTaskParams(BaseModel):
    employee_id: EmployeeIdStr
    task_code: OnboardingTaskCodeStr


class ScheduleOrientationFollowupParams(BaseModel):
    employee_id: EmployeeIdStr
    department_code: DepartmentCodeStr
    appointment_datetime: AppointmentDatetimeStr


# ---------------------------------------------------------------------------
# Flow 5: DEA Registration Transfer
# ---------------------------------------------------------------------------


class GetDeaRecordParams(BaseModel):
    npi: NpiStr
    dea_number: DeaNumberStr


class TransferDeaRegistrationParams(BaseModel):
    npi: NpiStr
    dea_number: DeaNumberStr
    new_facility_code: FacilityCodeStr
    new_state_code: StateCodeStr
    transfer_reason: TransferReason
    effective_date: DateStr


class NotifyPdmpParams(BaseModel):
    """Notify the state PDMP of a DEA registration transfer.
    No notification_type param — this tool is only used for transfers.
    """

    npi: NpiStr
    dea_number: DeaNumberStr
    state_code: StateCodeStr
    facility_code: FacilityCodeStr


# ---------------------------------------------------------------------------
# Flow 6: FMLA / Leave of Absence Filing
# ---------------------------------------------------------------------------


class CheckLeaveEligibilityParams(BaseModel):
    employee_id: EmployeeIdStr


class SubmitFmlaCaseParams(BaseModel):
    employee_id: EmployeeIdStr
    leave_category: LeaveCategory
    leave_start_date: DateStr
    leave_end_date: DateStr
    covering_employee_id: EmployeeIdStr


class ScheduleReturnToWorkCheckinParams(BaseModel):
    employee_id: EmployeeIdStr
    case_id: CaseIdStr
    department_code: DepartmentCodeStr
    appointment_datetime: AppointmentDatetimeStr


# ---------------------------------------------------------------------------
# Flow 7: Payroll Correction
# ---------------------------------------------------------------------------


class GetTimesheetRecordParams(BaseModel):
    employee_id: EmployeeIdStr
    shift_id: ShiftIdStr


class CheckCorrectionEligibilityParams(BaseModel):
    employee_id: EmployeeIdStr
    shift_id: ShiftIdStr


class SubmitPayrollCorrectionParams(BaseModel):
    employee_id: EmployeeIdStr
    shift_id: ShiftIdStr
    correction_type: PayrollCorrectionType
    corrected_hours: float = Field(
        gt=0,
        le=24,
        description="Correct total hours for the shift (not the delta). Example: 12.0 if 12 hours were worked.",
        examples=[8.0, 12.0, 12.5],
    )
    pay_period_end_date: DateStr


# ---------------------------------------------------------------------------
# Flow 8: Privilege Reactivation After Leave
# ---------------------------------------------------------------------------


class GetPrivilegeRecordParams(BaseModel):
    npi: NpiStr


class CheckReactivationEligibilityParams(BaseModel):
    npi: NpiStr
    clearance_code: str = Field(
        pattern=r"^CLR-[A-Z]{2,4}-\d{6}$",
        description="Occupational health clearance code in format CLR-LETTERS-6digits",
        examples=["CLR-OCC-048271"],
    )


class ScheduleCompetencyReviewParams(BaseModel):
    """Schedule competency review BEFORE reactivating privileges."""

    npi: NpiStr
    department_code: DepartmentCodeStr
    appointment_datetime: AppointmentDatetimeStr


class ReactivatePrivilegesParams(BaseModel):
    """Reactivate suspended privileges. Called AFTER scheduling the competency review."""

    npi: NpiStr
    privilege_codes: list[PrivilegeCodeStr]
    clearance_code: str = Field(
        pattern=r"^CLR-[A-Z]{2,4}-\d{6}$",
        description="Occupational health clearance code — same value used in check_reactivation_eligibility",
        examples=["CLR-OCC-048271"],
    )
    leave_type_on_record: LeaveTypeOnRecord


class UpdateEhrAccessParams(BaseModel):
    npi: NpiStr
    case_id: CaseIdStr
    access_change_type: EhrAccessChangeType


# ---------------------------------------------------------------------------
# Flow 9: On-Call Schedule Registration
# ---------------------------------------------------------------------------


class GetOncallScheduleParams(BaseModel):
    employee_id: EmployeeIdStr
    unit_code: UnitCodeStr


class CheckOncallEligibilityParams(BaseModel):
    employee_id: EmployeeIdStr
    unit_code: UnitCodeStr


class RegisterOncallAvailabilityParams(BaseModel):
    employee_id: EmployeeIdStr
    unit_code: UnitCodeStr
    availability_start_date: DateStr
    availability_end_date: DateStr
    oncall_tier: OnCallTier
    blackout_dates: list[DateStr]


# ---------------------------------------------------------------------------
# Flow 10: I-9 Verification
# ---------------------------------------------------------------------------


class GetI9RecordParams(BaseModel):
    employee_id: EmployeeIdStr


class SubmitI9VerificationParams(BaseModel):
    employee_id: EmployeeIdStr
    verification_action: I9VerificationAction
    document_list_type: I9DocumentTypeStr
    document_type_code: I9DocumentCodeStr
    document_number: str = Field(
        pattern=r"^[A-Z0-9]{6,12}$",
        description="Document ID number: 6-12 uppercase alphanumeric characters",
        examples=["A12345678", "D4829301"],
    )
    document_expiration_date: DateStr
    issuing_country_code: CountryCodeStr


class NotifyHrComplianceParams(BaseModel):
    employee_id: EmployeeIdStr
    case_id: CaseIdStr
    notification_type: HrComplianceNotificationType


# ---------------------------------------------------------------------------
# Flow 11: Visa Dependent Addition
# ---------------------------------------------------------------------------


class GetVisaRecordParams(BaseModel):
    employee_id: EmployeeIdStr
    visa_petition_number: VisaPetitionStr


class AddVisaDependentParams(BaseModel):
    employee_id: EmployeeIdStr
    visa_petition_number: VisaPetitionStr
    dependent_first_name: str = Field(
        pattern=r"^[A-Za-z\-]{2,30}$",
        description="Dependent first name, letters and hyphens only",
        examples=["Priya", "Jean-Luc"],
    )
    dependent_last_name: str = Field(
        pattern=r"^[A-Za-z\-]{2,30}$",
        description="Dependent last name, letters and hyphens only",
        examples=["Sharma", "Dupont"],
    )
    relationship: DependentRelationship
    dependent_date_of_birth: DateStr
    dependent_country_of_birth: CountryCodeStr
    uscis_receipt_number: UsciReceiptStr


class NotifyImmigrationCounselParams(BaseModel):
    employee_id: EmployeeIdStr
    visa_petition_number: VisaPetitionStr
    notification_type: ImmigrationNotificationType


# ---------------------------------------------------------------------------
# Flow 12: PTO Request
# ---------------------------------------------------------------------------


class GetPtoBalanceParams(BaseModel):
    employee_id: EmployeeIdStr


class CheckPtoEligibilityParams(BaseModel):
    employee_id: EmployeeIdStr
    pto_type: PtoType
    start_date: DateStr
    end_date: DateStr


class SubmitPtoRequestParams(BaseModel):
    employee_id: EmployeeIdStr
    pto_type: PtoType
    start_date: DateStr
    end_date: DateStr


# ---------------------------------------------------------------------------
# FIELD_ERROR_TYPES
# ---------------------------------------------------------------------------

FIELD_ERROR_TYPES: dict[str, tuple[str, str]] = {
    # Auth
    "employee_id": ("invalid_employee_id_format", "employee_id"),
    "npi": ("invalid_npi_format", "npi"),
    "facility_code": ("invalid_facility_code_format", "facility_code"),
    "pin": ("invalid_pin_format", "pin"),
    "otp_code": ("invalid_otp_format", "otp_code"),
    "date_of_birth": ("invalid_date_format", "date_of_birth"),
    # License
    "state_license_number": ("invalid_license_number_format", "state_license_number"),
    "extension_type": ("invalid_extension_type", "extension_type"),
    "extension_days": ("invalid_extension_days", "extension_days"),
    "supervising_physician_npi": ("invalid_npi_format", "supervising_physician_npi"),
    # Shift
    "shift_id": ("invalid_shift_id_format", "shift_id"),
    "unit_code": ("invalid_unit_code_format", "unit_code"),
    "colleague_employee_id": ("invalid_employee_id_format", "colleague_employee_id"),
    # Malpractice
    "new_carrier": ("invalid_carrier", "new_carrier"),
    "new_policy_number": ("invalid_policy_number_format", "new_policy_number"),
    "per_occurrence_limit_usd": ("invalid_coverage_limit", "per_occurrence_limit_usd"),
    "aggregate_limit_usd": ("invalid_coverage_limit", "aggregate_limit_usd"),
    "effective_date": ("invalid_date_format", "effective_date"),
    "expiration_date": ("invalid_date_format", "expiration_date"),
    # Onboarding
    "task_code": ("invalid_task_code", "task_code"),
    # DEA
    "dea_number": ("invalid_dea_number_format", "dea_number"),
    "new_facility_code": ("invalid_facility_code_format", "new_facility_code"),
    "new_state_code": ("invalid_state_code_format", "new_state_code"),
    "state_code": ("invalid_state_code_format", "state_code"),
    "transfer_reason": ("invalid_transfer_reason", "transfer_reason"),
    # FMLA
    "leave_category": ("invalid_leave_category", "leave_category"),
    "leave_start_date": ("invalid_date_format", "leave_start_date"),
    "leave_end_date": ("invalid_date_format", "leave_end_date"),
    "covering_employee_id": ("invalid_employee_id_format", "covering_employee_id"),
    # Payroll
    "correction_type": ("invalid_correction_type", "correction_type"),
    "corrected_hours": ("invalid_hours", "corrected_hours"),
    "pay_period_end_date": ("invalid_date_format", "pay_period_end_date"),
    # Privilege
    "privilege_codes": ("invalid_privilege_code", "privilege_codes"),
    "clearance_code": ("invalid_clearance_code_format", "clearance_code"),
    "leave_type_on_record": ("invalid_leave_type_on_record", "leave_type_on_record"),
    "access_change_type": ("invalid_access_change_type", "access_change_type"),
    # Scheduling
    "appointment_type": ("invalid_appointment_type", "appointment_type"),
    "preferred_date": ("invalid_date_format", "preferred_date"),
    "appointment_datetime": ("invalid_appointment_datetime_format", "appointment_datetime"),
    # On-call
    "oncall_tier": ("invalid_oncall_tier", "oncall_tier"),
    "availability_start_date": ("invalid_date_format", "availability_start_date"),
    "availability_end_date": ("invalid_date_format", "availability_end_date"),
    "blackout_dates": ("invalid_date_format", "blackout_dates"),
    # I-9
    "verification_action": ("invalid_verification_action", "verification_action"),
    "document_list_type": ("invalid_document_list_type", "document_list_type"),
    "document_type_code": ("invalid_document_type_code", "document_type_code"),
    "document_number": ("invalid_document_number_format", "document_number"),
    "document_expiration_date": ("invalid_date_format", "document_expiration_date"),
    "issuing_country_code": ("invalid_country_code_format", "issuing_country_code"),
    # Notification types
    "notification_type": ("invalid_notification_type", "notification_type"),
    # Visa
    "visa_petition_number": ("invalid_petition_number_format", "visa_petition_number"),
    "dependent_first_name": ("invalid_name_format", "dependent_first_name"),
    "dependent_last_name": ("invalid_name_format", "dependent_last_name"),
    "relationship": ("invalid_relationship", "relationship"),
    "dependent_date_of_birth": ("invalid_date_format", "dependent_date_of_birth"),
    "dependent_country_of_birth": ("invalid_country_code_format", "dependent_country_of_birth"),
    "uscis_receipt_number": ("invalid_uscis_receipt_format", "uscis_receipt_number"),
    # Department / case
    "department_code": ("invalid_department_code", "department_code"),
    "case_id": ("invalid_case_id_format", "case_id"),
    # PTO
    "pto_type": ("invalid_pto_type", "pto_type"),
    "start_date": ("invalid_date_format", "start_date"),
    "end_date": ("invalid_date_format", "end_date"),
}


def validation_error_response(exc: ValidationError, model: type[BaseModel]) -> dict:
    """Convert a Pydantic ValidationError to a standard tool error response."""
    for error in exc.errors():
        loc = error.get("loc", ())
        if loc:
            field = str(loc[0])
            if field in FIELD_ERROR_TYPES:
                error_type, label = FIELD_ERROR_TYPES[field]
                input_val = error.get("input", "")
                msg = f"Invalid {label} '{input_val}'"
                if (field_info := model.model_fields.get(field)) and field_info.description:
                    msg += f": must be {field_info.description}"
                    if field_info.examples:
                        msg += f" (e.g. {', '.join(str(e) for e in field_info.examples)})"
                elif detail := error.get("msg", ""):
                    msg += f": {detail}"
                return {"status": "error", "error_type": error_type, "message": msg}

    first = exc.errors()[0] if exc.errors() else {}
    loc = first.get("loc", ("parameter",))
    field = str(loc[0]) if loc else "parameter"
    return {
        "status": "error",
        "error_type": "invalid_parameter",
        "message": f"Invalid or missing parameter '{field}': {first.get('msg', str(exc))}",
    }
