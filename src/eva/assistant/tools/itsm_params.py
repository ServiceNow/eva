"""Pydantic parameter models and enums for ITSM tool functions.

Tool sequences per flow (updated):

  Flow 1  – Login Issue:
    verify_employee_auth → get_employee_record → get_troubleshooting_guide
    → attempt_account_unlock | attempt_password_reset
    → (if failed) create_incident_ticket → assign_sla_tier

  Flow 2a – Service Outage (existing):
    verify_employee_auth → get_employee_record → check_existing_outage
    → add_affected_user

  Flow 2b – Service Outage (new):
    verify_employee_auth → get_employee_record → check_existing_outage
    → create_incident_ticket → link_known_error

  Flow 3  – Hardware Malfunction:
    verify_employee_auth → get_employee_assets → get_asset_record
    → create_incident_ticket → schedule_field_dispatch

  Flow 4  – Network/VPN Issue:
    verify_employee_auth → get_employee_record → get_troubleshooting_guide
    → create_incident_ticket → attach_diagnostic_log

  Flow 5  – Laptop Replacement:
    verify_employee_auth → get_employee_assets → check_hardware_entitlement
    → submit_hardware_request → initiate_asset_return

  Flow 6  – Monitor Bundle:
    verify_employee_auth → check_hardware_entitlement
    → verify_cost_center_budget → submit_hardware_request

  Flow 7  – Application Access Request:
    verify_employee_auth → get_application_details
    → submit_access_request → route_approval_workflow

  Flow 8  – License Request:
    verify_employee_auth → get_license_catalog_item
    → validate_cost_center → submit_license_request

  Flow 9  – Temporary License:
    verify_employee_auth → get_license_catalog_item
    → submit_temporary_license

  Flow 10 – License Renewal:
    verify_employee_auth → get_employee_licenses
    → check_renewal_eligibility → submit_license_renewal

  Flow 11 – Desk/Office Space:
    verify_employee_auth → check_space_availability → submit_desk_assignment

  Flow 12 – Parking Space:
    verify_employee_auth → check_parking_availability → submit_parking_assignment

  Flow 13 – Ergonomic Equipment:
    verify_employee_auth → check_ergonomic_assessment → submit_equipment_request

  Flow 14 – Conference Room:
    verify_employee_auth → check_room_availability
    → submit_room_booking → send_calendar_invite

  Flow 15 – Account Provisioning:
    verify_employee_auth → verify_manager_auth → initiate_otp_auth
    → verify_otp_auth → lookup_new_hire → check_existing_accounts
    → provision_new_account

  Flow 16 – Group Membership:
    verify_employee_auth → initiate_otp_auth → verify_otp_auth
    → get_group_memberships → get_group_details
    → submit_group_membership_change

  Flow 17 – Permission Change:
    verify_employee_auth → initiate_otp_auth → verify_otp_auth
    → get_permission_templates → submit_permission_change
    → schedule_access_review

  Flow 18 – Access Removal:
    verify_employee_auth → verify_manager_auth → initiate_otp_auth
    → verify_otp_auth → get_offboarding_record
    → submit_access_removal → initiate_asset_recovery
"""

from enum import StrEnum
from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field, ValidationError

# ---------------------------------------------------------------------------
# Annotated ID types
# ---------------------------------------------------------------------------

EmployeeIdStr = Annotated[str, Field(pattern=r"^EMP\d{6}$", description="EMP followed by 6 digits", examples=["EMP048271"])]
PhoneLastFourStr = Annotated[str, Field(pattern=r"^\d{4}$", description="Last 4 digits of phone number", examples=["7294"])]
OtpStr = Annotated[str, Field(pattern=r"^\d{6}$", description="6-digit OTP code", examples=["483920"])]
ManagerAuthCodeStr = Annotated[str, Field(pattern=r"^[A-Z0-9]{6}$", description="6-char alphanumeric manager auth code", examples=["K4M2P9"])]
TicketNumberStr = Annotated[str, Field(pattern=r"^INC\d{7}$", description="INC followed by 7 digits", examples=["INC0048271"])]
AssetTagStr = Annotated[str, Field(pattern=r"^AST-[A-Z]{3}-\d{6}$", description="AST-XXX-NNNNNN", examples=["AST-LPT-284719"])]
AppCatalogIdStr = Annotated[str, Field(pattern=r"^APP-\d{4}$", description="APP-NNNN", examples=["APP-0042"])]
LicCatalogIdStr = Annotated[str, Field(pattern=r"^LIC-\d{4}$", description="LIC-NNNN", examples=["LIC-0018"])]
LicenseAssignmentIdStr = Annotated[str, Field(pattern=r"^LASGN-\d{6}$", description="LASGN-NNNNNN", examples=["LASGN-048271"])]
BuildingCodeStr = Annotated[str, Field(pattern=r"^BLD\d{1,2}$", description="BLD followed by 1-2 digits", examples=["BLD3"])]
FloorCodeStr = Annotated[str, Field(pattern=r"^FL\d{1,2}$", description="FL followed by 1-2 digits", examples=["FL2"])]
RoomCodeStr = Annotated[str, Field(pattern=r"^BLD\d{1,2}-FL\d{1,2}-RM\d{3}$", description="BLD-FL-RM code", examples=["BLD3-FL2-RM204"])]
DeskCodeStr = Annotated[str, Field(pattern=r"^BLD\d{1,2}-FL\d{1,2}-D\d{3}$", description="BLD-FL-D code", examples=["BLD3-FL2-D107"])]
ParkingZoneStr = Annotated[str, Field(pattern=r"^PZ[A-Z]$", description="PZ + letter", examples=["PZA"])]
ParkingSpaceIdStr = Annotated[str, Field(pattern=r"^PZ[A-Z]-\d{3}$", description="PZX-NNN", examples=["PZA-042"])]
CostCenterCodeStr = Annotated[str, Field(pattern=r"^CC-\d{4}$", description="CC-NNNN", examples=["CC-4021"])]
DepartmentCodeStr = Annotated[str, Field(pattern=r"^(ENG|MKTG|SALES|FIN|HR|OPS|LEGAL|INFRA|SECUR|EXEC|DSGN|DATA)$", description="Department code", examples=["ENG"])]
RoleCodeStr = Annotated[str, Field(pattern=r"^(SWE|PM|DESGN|ANLST|ADMIN|MGENG|MGOPS|MGSLS|MGHR|SECUR|INFRA|DATAN|LEGAL)$", description="Role code", examples=["SWE"])]
GroupCodeStr = Annotated[str, Field(pattern=r"^GRP-[A-Z]{2,6}$", description="GRP-XXXXXX", examples=["GRP-ENGCORE"])]
PermissionTemplateIdStr = Annotated[str, Field(pattern=r"^PTPL-[A-Z]{2,5}-\d{2}$", description="PTPL-XXX-NN", examples=["PTPL-SWE-01"])]
RequestIdStr = Annotated[str, Field(pattern=r"^REQ-[A-Z]{2,5}-\d{6}$", description="REQ-CAT-NNNNNN", examples=["REQ-HW-048271"])]
CaseIdStr = Annotated[str, Field(pattern=r"^CASE-[A-Z]{2,5}-\d{6}$", description="CASE-CAT-NNNNNN", examples=["CASE-ACCT-048271"])]
DateStr = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD", examples=["2026-08-15"])]
TimeStr = Annotated[str, Field(pattern=r"^\d{2}:\d{2}$", description="HH:MM", examples=["09:00"])]
DiagnosticRefCodeStr = Annotated[str, Field(pattern=r"^DIAG-[A-Z0-9]{8}$", description="DIAG-XXXXXXXX", examples=["DIAG-4KM29X7B"])]

ServiceNameStr = Annotated[str, Field(
    pattern=r"^(email_exchange|vpn_gateway|erp_oracle|crm_platform|hr_portal|code_repository|ci_cd_pipeline|file_storage|sso_identity|print_service)$",
    description="Service catalog name", examples=["email_exchange"])]

TargetSystemStr = Annotated[str, Field(
    pattern=r"^(active_directory|sso_identity|email_exchange|vpn_gateway|erp_oracle)$",
    description="Target system: active_directory, sso_identity, email_exchange, vpn_gateway, or erp_oracle",
    examples=["active_directory"])]

AffectedSystemStr = Annotated[str, Field(
    pattern=r"^(active_directory|sso_identity|email_exchange|vpn_gateway|erp_oracle|crm_platform|hr_portal|code_repository|ci_cd_pipeline|file_storage|print_service|vpn|wifi|ethernet|AST-[A-Z]{3}-\d{6})$",
    description="Affected system ID: service name, network type (vpn/wifi/ethernet), or asset tag",
    examples=["email_exchange", "vpn", "AST-LPT-284719"])]

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class IncidentCategory(StrEnum):
    login_issue = "login_issue"
    service_outage = "service_outage"
    hardware_malfunction = "hardware_malfunction"
    network_connectivity = "network_connectivity"

class Urgency(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"

class SLATier(StrEnum):
    tier_1 = "tier_1"
    tier_2 = "tier_2"
    tier_3 = "tier_3"

class DispatchTimeWindow(StrEnum):
    morning = "morning"
    afternoon = "afternoon"
    full_day = "full_day"

class HardwareRequestType(StrEnum):
    laptop_replacement = "laptop_replacement"
    monitor_bundle = "monitor_bundle"

class LaptopReplacementReason(StrEnum):
    end_of_life = "end_of_life"
    performance_degradation = "performance_degradation"
    physical_damage = "physical_damage"

class MonitorSetupReason(StrEnum):
    new_setup = "new_setup"
    replacement = "replacement"

class MonitorSize(StrEnum):
    size_24 = "24_inch"
    size_27 = "27_inch"
    size_32 = "32_inch"

class AccessLevel(StrEnum):
    read_only = "read_only"
    standard = "standard"
    admin = "admin"

class EquipmentType(StrEnum):
    standing_desk_converter = "standing_desk_converter"
    ergonomic_chair = "ergonomic_chair"
    ergonomic_keyboard = "ergonomic_keyboard"
    monitor_arm = "monitor_arm"
    footrest = "footrest"

class GroupMembershipAction(StrEnum):
    add = "add"
    remove = "remove"

class AccessRemovalScope(StrEnum):
    full = "full"
    staged = "staged"

class AssetRecoveryMethod(StrEnum):
    office_pickup = "office_pickup"
    shipping_label = "shipping_label"
    drop_off = "drop_off"

class TroubleshootingCategory(StrEnum):
    login_issue = "login_issue"
    network_connectivity = "network_connectivity"

class TransferReason(StrEnum):
    caller_requested = "caller_requested"
    policy_exception_needed = "policy_exception_needed"
    unable_to_resolve = "unable_to_resolve"
    complaint_escalation = "complaint_escalation"
    technical_issue = "technical_issue"

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class VerifyEmployeeAuthParams(BaseModel):
    employee_id: EmployeeIdStr
    phone_last_four: PhoneLastFourStr

class InitiateOtpAuthParams(BaseModel):
    employee_id: EmployeeIdStr

class VerifyOtpAuthParams(BaseModel):
    employee_id: EmployeeIdStr
    otp_code: OtpStr

class VerifyManagerAuthParams(BaseModel):
    employee_id: EmployeeIdStr
    manager_auth_code: ManagerAuthCodeStr

# ---------------------------------------------------------------------------
# Shared lookups
# ---------------------------------------------------------------------------

class GetEmployeeRecordParams(BaseModel):
    employee_id: EmployeeIdStr

class GetAssetRecordParams(BaseModel):
    asset_tag: AssetTagStr

class GetEmployeeAssetsParams(BaseModel):
    employee_id: EmployeeIdStr

# ---------------------------------------------------------------------------
# Troubleshooting + direct resolution (Flow 1)
# ---------------------------------------------------------------------------

class GetTroubleshootingGuideParams(BaseModel):
    issue_category: TroubleshootingCategory

class AttemptAccountUnlockParams(BaseModel):
    employee_id: EmployeeIdStr
    target_system: TargetSystemStr

class AttemptPasswordResetParams(BaseModel):
    employee_id: EmployeeIdStr
    target_system: TargetSystemStr

# ---------------------------------------------------------------------------
# Incident creation + trailing actions (Flows 1-4)
# ---------------------------------------------------------------------------

class CreateIncidentTicketParams(BaseModel):
    employee_id: EmployeeIdStr
    category: IncidentCategory
    urgency: Urgency
    affected_system: AffectedSystemStr
    troubleshooting_completed: bool

class AssignSLATierParams(BaseModel):
    ticket_number: TicketNumberStr
    sla_tier: SLATier

class CheckExistingOutageParams(BaseModel):
    service_name: ServiceNameStr

class AddAffectedUserParams(BaseModel):
    ticket_number: TicketNumberStr
    employee_id: EmployeeIdStr

class LinkKnownErrorParams(BaseModel):
    ticket_number: TicketNumberStr
    service_name: ServiceNameStr

class ScheduleFieldDispatchParams(BaseModel):
    ticket_number: TicketNumberStr
    employee_id: EmployeeIdStr
    building_code: BuildingCodeStr
    floor_code: FloorCodeStr
    preferred_date: DateStr
    time_window: DispatchTimeWindow

class AttachDiagnosticLogParams(BaseModel):
    ticket_number: TicketNumberStr
    diagnostic_ref_code: DiagnosticRefCodeStr

# ---------------------------------------------------------------------------
# Hardware (Flows 5-6)
# ---------------------------------------------------------------------------

class CheckHardwareEntitlementParams(BaseModel):
    employee_id: EmployeeIdStr
    request_type: HardwareRequestType

class SubmitHardwareRequestParams(BaseModel):
    employee_id: EmployeeIdStr
    request_type: HardwareRequestType
    replacement_reason: str
    current_asset_tag: Optional[AssetTagStr] = None
    monitor_size: Optional[MonitorSize] = None
    delivery_building: BuildingCodeStr
    delivery_floor: FloorCodeStr

class InitiateAssetReturnParams(BaseModel):
    employee_id: EmployeeIdStr
    asset_tag: AssetTagStr
    request_id: RequestIdStr

class VerifyCostCenterBudgetParams(BaseModel):
    department_code: DepartmentCodeStr
    cost_center_code: CostCenterCodeStr

# ---------------------------------------------------------------------------
# Software - access (Flow 7)
# ---------------------------------------------------------------------------

class GetApplicationDetailsParams(BaseModel):
    catalog_id: AppCatalogIdStr

class SubmitAccessRequestParams(BaseModel):
    employee_id: EmployeeIdStr
    catalog_id: AppCatalogIdStr
    access_level: AccessLevel

class RouteApprovalWorkflowParams(BaseModel):
    request_id: RequestIdStr
    employee_id: EmployeeIdStr
    approver_employee_id: EmployeeIdStr

# ---------------------------------------------------------------------------
# Software - license (Flows 8-9)
# ---------------------------------------------------------------------------

class GetLicenseCatalogItemParams(BaseModel):
    catalog_id: LicCatalogIdStr

class ValidateCostCenterParams(BaseModel):
    cost_center_code: CostCenterCodeStr
    department_code: DepartmentCodeStr

class SubmitLicenseRequestParams(BaseModel):
    employee_id: EmployeeIdStr
    catalog_id: LicCatalogIdStr

class SubmitTemporaryLicenseParams(BaseModel):
    employee_id: EmployeeIdStr
    catalog_id: LicCatalogIdStr
    duration_days: Literal[30, 60, 90]

# ---------------------------------------------------------------------------
# Software - renewal (Flow 10)
# ---------------------------------------------------------------------------

class GetEmployeeLicensesParams(BaseModel):
    employee_id: EmployeeIdStr

class CheckRenewalEligibilityParams(BaseModel):
    employee_id: EmployeeIdStr
    license_assignment_id: LicenseAssignmentIdStr

class SubmitLicenseRenewalParams(BaseModel):
    employee_id: EmployeeIdStr
    license_assignment_id: LicenseAssignmentIdStr

# ---------------------------------------------------------------------------
# Facilities (Flows 11-14)
# ---------------------------------------------------------------------------

class CheckSpaceAvailabilityParams(BaseModel):
    building_code: BuildingCodeStr
    floor_code: FloorCodeStr

class SubmitDeskAssignmentParams(BaseModel):
    employee_id: EmployeeIdStr
    desk_code: DeskCodeStr

class CheckParkingAvailabilityParams(BaseModel):
    zone_code: ParkingZoneStr

class SubmitParkingAssignmentParams(BaseModel):
    employee_id: EmployeeIdStr
    parking_space_id: ParkingSpaceIdStr

class CheckErgonomicAssessmentParams(BaseModel):
    employee_id: EmployeeIdStr

class SubmitEquipmentRequestParams(BaseModel):
    employee_id: EmployeeIdStr
    equipment_type: EquipmentType
    delivery_building: BuildingCodeStr
    delivery_floor: FloorCodeStr

class CheckRoomAvailabilityParams(BaseModel):
    building_code: BuildingCodeStr
    floor_code: FloorCodeStr
    date: DateStr
    start_time: TimeStr
    end_time: TimeStr
    min_capacity: int = Field(gt=0, le=50)

class SubmitRoomBookingParams(BaseModel):
    employee_id: EmployeeIdStr
    room_code: RoomCodeStr
    date: DateStr
    start_time: TimeStr
    end_time: TimeStr
    attendee_count: int = Field(gt=0, le=50)

class SendCalendarInviteParams(BaseModel):
    request_id: RequestIdStr
    employee_id: EmployeeIdStr
    room_code: RoomCodeStr
    date: DateStr
    start_time: TimeStr
    end_time: TimeStr

# ---------------------------------------------------------------------------
# Accounts (Flows 15-18)
# ---------------------------------------------------------------------------

class LookupNewHireParams(BaseModel):
    new_hire_employee_id: EmployeeIdStr

class CheckExistingAccountsParams(BaseModel):
    employee_id: EmployeeIdStr

class ProvisionNewAccountParams(BaseModel):
    manager_employee_id: EmployeeIdStr
    new_hire_employee_id: EmployeeIdStr
    department_code: DepartmentCodeStr
    role_code: RoleCodeStr
    start_date: DateStr
    access_groups: list[GroupCodeStr]

class GetGroupMembershipsParams(BaseModel):
    employee_id: EmployeeIdStr

class GetGroupDetailsParams(BaseModel):
    group_code: GroupCodeStr

class SubmitGroupMembershipChangeParams(BaseModel):
    employee_id: EmployeeIdStr
    group_code: GroupCodeStr
    action: GroupMembershipAction

class GetPermissionTemplatesParams(BaseModel):
    role_code: RoleCodeStr

class SubmitPermissionChangeParams(BaseModel):
    employee_id: EmployeeIdStr
    new_role_code: RoleCodeStr
    permission_template_id: PermissionTemplateIdStr
    effective_date: DateStr

class ScheduleAccessReviewParams(BaseModel):
    case_id: CaseIdStr
    employee_id: EmployeeIdStr
    review_date: DateStr

class GetOffboardingRecordParams(BaseModel):
    employee_id: EmployeeIdStr

class SubmitAccessRemovalParams(BaseModel):
    manager_employee_id: EmployeeIdStr
    departing_employee_id: EmployeeIdStr
    last_working_day: DateStr
    removal_scope: AccessRemovalScope

class InitiateAssetRecoveryParams(BaseModel):
    departing_employee_id: EmployeeIdStr
    case_id: CaseIdStr
    recovery_method: AssetRecoveryMethod

# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

class TransferToAgentParams(BaseModel):
    employee_id: EmployeeIdStr
    transfer_reason: TransferReason
    issue_summary: str = Field(min_length=10, max_length=500)

# ---------------------------------------------------------------------------
# FIELD_ERROR_TYPES
# ---------------------------------------------------------------------------

FIELD_ERROR_TYPES: dict[str, tuple[str, str]] = {
    "employee_id": ("invalid_employee_id_format", "employee_id"),
    "phone_last_four": ("invalid_phone_format", "phone_last_four"),
    "otp_code": ("invalid_otp_format", "otp_code"),
    "manager_auth_code": ("invalid_manager_auth_code_format", "manager_auth_code"),
    "ticket_number": ("invalid_ticket_number_format", "ticket_number"),
    "asset_tag": ("invalid_asset_tag_format", "asset_tag"),
    "current_asset_tag": ("invalid_asset_tag_format", "current_asset_tag"),
    "catalog_id": ("invalid_catalog_id_format", "catalog_id"),
    "license_assignment_id": ("invalid_license_assignment_id_format", "license_assignment_id"),
    "building_code": ("invalid_building_code_format", "building_code"),
    "floor_code": ("invalid_floor_code_format", "floor_code"),
    "room_code": ("invalid_room_code_format", "room_code"),
    "desk_code": ("invalid_desk_code_format", "desk_code"),
    "zone_code": ("invalid_parking_zone_format", "zone_code"),
    "parking_space_id": ("invalid_parking_space_id_format", "parking_space_id"),
    "cost_center_code": ("invalid_cost_center_code_format", "cost_center_code"),
    "department_code": ("invalid_department_code", "department_code"),
    "role_code": ("invalid_role_code", "role_code"),
    "new_role_code": ("invalid_role_code", "new_role_code"),
    "group_code": ("invalid_group_code_format", "group_code"),
    "permission_template_id": ("invalid_permission_template_id_format", "permission_template_id"),
    "category": ("invalid_incident_category", "category"),
    "urgency": ("invalid_urgency", "urgency"),
    "issue_category": ("invalid_issue_category", "issue_category"),
    "request_type": ("invalid_request_type", "request_type"),
    "replacement_reason": ("invalid_replacement_reason", "replacement_reason"),
    "monitor_size": ("invalid_monitor_size", "monitor_size"),
    "access_level": ("invalid_access_level", "access_level"),
    "equipment_type": ("invalid_equipment_type", "equipment_type"),
    "action": ("invalid_membership_action", "action"),
    "removal_scope": ("invalid_removal_scope", "removal_scope"),
    "transfer_reason": ("invalid_transfer_reason", "transfer_reason"),
    "service_name": ("invalid_service_name", "service_name"),
    "target_system": ("invalid_target_system", "target_system"),
    "affected_system": ("invalid_affected_system", "affected_system"),
    "duration_days": ("invalid_duration", "duration_days"),
    "sla_tier": ("invalid_sla_tier", "sla_tier"),
    "time_window": ("invalid_time_window", "time_window"),
    "recovery_method": ("invalid_recovery_method", "recovery_method"),
    "diagnostic_ref_code": ("invalid_diagnostic_ref_code_format", "diagnostic_ref_code"),
    "date": ("invalid_date_format", "date"),
    "start_date": ("invalid_date_format", "start_date"),
    "effective_date": ("invalid_date_format", "effective_date"),
    "last_working_day": ("invalid_date_format", "last_working_day"),
    "review_date": ("invalid_date_format", "review_date"),
    "preferred_date": ("invalid_date_format", "preferred_date"),
    "start_time": ("invalid_time_format", "start_time"),
    "end_time": ("invalid_time_format", "end_time"),
    "request_id": ("invalid_request_id_format", "request_id"),
    "case_id": ("invalid_case_id_format", "case_id"),
    "new_hire_employee_id": ("invalid_employee_id_format", "new_hire_employee_id"),
    "manager_employee_id": ("invalid_employee_id_format", "manager_employee_id"),
    "departing_employee_id": ("invalid_employee_id_format", "departing_employee_id"),
    "approver_employee_id": ("invalid_employee_id_format", "approver_employee_id"),
    "delivery_building": ("invalid_building_code_format", "delivery_building"),
    "delivery_floor": ("invalid_floor_code_format", "delivery_floor"),
}

def validation_error_response(exc: ValidationError, model: type[BaseModel]) -> dict:
    for error in exc.errors():
        loc = error.get("loc", ())
        if loc:
            field = str(loc[0])
            if field in FIELD_ERROR_TYPES:
                error_type, label = FIELD_ERROR_TYPES[field]
                input_val = error.get("input", "")
                msg = f"Invalid {label} '{input_val}'"
                if (fi := model.model_fields.get(field)) and fi.description:
                    msg += f": must be {fi.description}"
                    if fi.examples: msg += f" (e.g. {', '.join(str(e) for e in fi.examples)})"
                elif detail := error.get("msg", ""): msg += f": {detail}"
                return {"status": "error", "error_type": error_type, "message": msg}
    first = exc.errors()[0] if exc.errors() else {}
    loc = first.get("loc", ("parameter",))
    return {"status": "error", "error_type": "invalid_parameter", "message": f"Invalid '{loc[0] if loc else 'parameter'}': {first.get('msg', str(exc))}"}