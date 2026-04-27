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
    → verify_cost_center_budget (auto-fetches CC from employee record)
    → submit_hardware_request (justification + laptop_os + laptop_size required)
    → initiate_asset_return

  Flow 6  – Monitor Bundle:
    verify_employee_auth → check_hardware_entitlement
    → verify_cost_center_budget (auto-fetches CC from employee record) → submit_hardware_request

  Flow 7  – Application Access Request:
    verify_employee_auth → get_application_details (by application_name)
    → submit_access_request → route_approval_workflow (only when approval required)

  Flow 8  – License Request (Permanent):
    verify_employee_auth → get_license_catalog_item (by license_name)
    → validate_cost_center (auto-fetches) → submit_license_request (duration_days=None)

  Flow 9  – Temporary License:
    verify_employee_auth → get_license_catalog_item (by license_name)
    → submit_license_request (duration_days ∈ {30,60,90})

  Flow 10 – License Renewal:
    verify_employee_auth → get_employee_licenses
    → submit_license_renewal (tool enforces 30-day / 14-day-expired window)

  Flow 11 – Desk/Office Space:
    verify_employee_auth → check_desk_availability
    → submit_desk_assignment | submit_waitlist (when unavailable)

  Flow 12 – Parking Space:
    verify_employee_auth → check_parking_availability
    → submit_parking_assignment | submit_waitlist (when unavailable)

  Flow 13 – Ergonomic Equipment:
    verify_employee_auth → [check_ergonomic_assessment only for standing desk + chair]
    → submit_equipment_request

  Flow 14 – Conference Room:
    verify_employee_auth → check_room_availability (floor_code optional)
    → submit_room_booking → send_calendar_invite (persists calendar_events row)

  Flow 15 – Account Provisioning:
    verify_manager_auth (phone_last_four + manager_auth_code) → initiate_otp_auth
    → verify_otp_auth → lookup_new_hire → check_existing_accounts
    → provision_new_account

  Flow 16 – Group Membership:
    verify_employee_auth → initiate_otp_auth → verify_otp_auth
    → get_group_memberships → get_group_details
    → submit_group_membership_change
    → route_approval_workflow (only when group requires_approval and action=add;
      uses request_id returned by submit_group_membership_change)

  Flow 17 – Permission Change:
    verify_employee_auth → initiate_otp_auth → verify_otp_auth
    → check_role_change_authorized → get_permission_templates
    → submit_permission_change → schedule_access_review

  Flow 18 – Access Removal:
    verify_manager_auth (phone_last_four + manager_auth_code) → initiate_otp_auth
    → verify_otp_auth → get_offboarding_record
    → submit_access_removal → initiate_asset_recovery

  Flow 19 – Security Incident / Stolen Device:
    verify_employee_auth → get_employee_assets → report_security_incident
    → initiate_remote_wipe → submit_hardware_request (justification=lost_or_stolen)

  Flow 20 – MFA Reset / Lost Device (Refusal + Escalation):
    verify_employee_auth → submit_mfa_reset (returns in_person_required)
    → transfer_to_agent

  Flow 21 – Software Request Status / Escalation:
    verify_employee_auth → get_request_status
    → escalate_approval (only when SLA exceeded)
"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

_EMPTY_SENTINELS = {"", ".", "none", "null", "n/a", "na"}


def _empty_to_none(v):
    """Coerce common empty/placeholder strings to None for Optional fields.

    The agent sometimes emits an empty string instead of omitting an optional
    field; without this, enum/pattern validators reject the value before the
    @model_validator conditional check can produce a helpful error.
    """
    if isinstance(v, str) and v.strip().lower() in _EMPTY_SENTINELS:
        return None
    return v


# ---------------------------------------------------------------------------
# Annotated ID types
# ---------------------------------------------------------------------------

EmployeeIdStr = Annotated[
    str, Field(pattern=r"^EMP\d{6}$", description="EMP followed by 6 digits", examples=["EMP048271"])
]
PhoneLastFourStr = Annotated[
    str, Field(pattern=r"^\d{4}$", description="Last 4 digits of phone number", examples=["7294"])
]
OtpStr = Annotated[str, Field(pattern=r"^\d{6}$", description="6-digit OTP code", examples=["483920"])]
ManagerAuthCodeStr = Annotated[
    str, Field(pattern=r"^[A-Z0-9]{6}$", description="6-char alphanumeric manager auth code", examples=["K4M2P9"])
]
TicketNumberStr = Annotated[
    str,
    Field(
        pattern=r"^INC[a-f0-9]{12}$",
        description="INC followed by 12 hex chars (content-hash)",
        examples=["INCfa5e5b2062da"],
    ),
]
AssetTagStr = Annotated[
    str, Field(pattern=r"^AST-[A-Z]{3}-\d{6}$", description="AST-XXX-NNNNNN", examples=["AST-LPT-284719"])
]
LicenseAssignmentIdStr = Annotated[
    str, Field(pattern=r"^LASGN-\d{6}$", description="LASGN-NNNNNN", examples=["LASGN-048271"])
]
BuildingCodeStr = Annotated[
    str,
    Field(
        min_length=1,
        max_length=60,
        description="Building code (e.g. 'BLD3') OR building name/alias (e.g. 'Headquarters', 'HQ', 'Downtown'). Tools resolve names to canonical code via facilities.buildings aliases; unknown names return `name_not_found`.",
        examples=["BLD3", "Downtown"],
    ),
]
FloorCodeStr = Annotated[str, Field(pattern=r"^FL\d{1,2}$", description="FL followed by 1-2 digits", examples=["FL2"])]
RoomCodeStr = Annotated[
    str,
    Field(
        pattern=r"^BLD\d{1,2}-FL\d{1,2}-RM\d{3}$",
        description="BLD-FL-RM code (returned by check_room_availability; not caller-provided)",
        examples=["BLD3-FL2-RM204"],
    ),
]
DeskCodeStr = Annotated[
    str,
    Field(
        pattern=r"^BLD\d{1,2}-FL\d{1,2}-D\d{3}$",
        description="BLD-FL-D code (returned by check_desk_availability)",
        examples=["BLD3-FL2-D107"],
    ),
]
ParkingZoneStr = Annotated[
    str,
    Field(
        min_length=1,
        max_length=60,
        description="Parking zone code (e.g. 'PZA') OR zone name/alias (e.g. 'Executive Garage'). Tools resolve names to canonical code via facilities.zones aliases.",
        examples=["PZA", "Executive Garage"],
    ),
]
ParkingSpaceIdStr = Annotated[
    str,
    Field(
        pattern=r"^PZ[A-Z]-\d{3}$", description="PZX-NNN (returned by check_parking_availability)", examples=["PZA-042"]
    ),
]
DepartmentCodeStr = Annotated[
    str,
    Field(
        pattern=r"^(ENG|MKTG|SALES|FIN|HR|OPS|LEGAL|INFRA|SECUR|EXEC|DSGN|DATA)$",
        description="Department code",
        examples=["ENG"],
    ),
]
RoleCodeStr = Annotated[
    str,
    Field(
        pattern=r"^(SWE|PM|DESGN|ANLST|ADMIN|MGENG|MGOPS|MGSLS|MGHR|SECUR|INFRA|DATAN|LEGAL)$",
        description="Role code",
        examples=["SWE"],
    ),
]
GroupCodeStr = Annotated[str, Field(pattern=r"^GRP-[A-Z]{2,8}$", description="GRP-XXXXXX", examples=["GRP-ENGCORE"])]
PermissionTemplateIdStr = Annotated[
    str, Field(pattern=r"^PTPL-[A-Z]{2,5}-\d{2}$", description="PTPL-XXX-NN", examples=["PTPL-SWE-01"])
]
RequestIdStr = Annotated[
    str,
    Field(
        pattern=r"^REQ-[A-Z]{2,5}-[a-f0-9]{12}$",
        description="REQ-CAT-XXXXXXXXXXXX (CAT plus 12-char content hash)",
        examples=["REQ-HW-6d94147c44a3"],
    ),
]
CaseIdStr = Annotated[
    str,
    Field(
        pattern=r"^CASE-[A-Z]{2,5}-[a-f0-9]{12}$",
        description="CASE-CAT-XXXXXXXXXXXX",
        examples=["CASE-ACCT-6b54dcee093d"],
    ),
]
SecurityCaseIdStr = Annotated[
    str,
    Field(pattern=r"^SEC-[a-f0-9]{12}$", description="SEC-XXXXXXXXXXXX", examples=["SEC-a3f8e1b2c4d5"]),
]
DateStr = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD", examples=["2026-08-15"])]
TimeStr = Annotated[str, Field(pattern=r"^\d{2}:\d{2}$", description="HH:MM", examples=["09:00"])]
DiagnosticRefCodeStr = Annotated[
    str, Field(pattern=r"^DIAG-[A-Z0-9]{8}$", description="DIAG-XXXXXXXX", examples=["DIAG-4KM29X7B"])
]

# Application/License names are free-form strings; tools validate against catalog names + name_aliases (exact, case-insensitive).
CatalogNameStr = Annotated[
    str,
    Field(
        min_length=1,
        max_length=120,
        description="Catalog item name (e.g. 'Slack Enterprise'); resolved to catalog_id via names + aliases",
        examples=["Slack Enterprise"],
    ),
]

ServiceNameStr = Annotated[
    str,
    Field(
        pattern=r"^(email_exchange|vpn_gateway|erp_oracle|crm_platform|hr_portal|code_repository|ci_cd_pipeline|file_storage|sso_identity|print_service)$",
        description="Service catalog name",
        examples=["email_exchange"],
    ),
]

TargetSystemStr = Annotated[
    str,
    Field(
        pattern=r"^(active_directory|sso_identity|email_exchange|vpn_gateway|erp_oracle)$",
        description="Target system: active_directory, sso_identity, email_exchange, vpn_gateway, or erp_oracle",
        examples=["active_directory"],
    ),
]

AffectedSystemStr = Annotated[
    str,
    Field(
        pattern=r"^(active_directory|sso_identity|email_exchange|vpn_gateway|erp_oracle|crm_platform|hr_portal|code_repository|ci_cd_pipeline|file_storage|print_service|vpn|wifi|ethernet|AST-[A-Z]{3}-\d{6})$",
        description="Affected system ID: service name, network type (vpn/wifi/ethernet), or asset tag",
        examples=["email_exchange", "vpn", "AST-LPT-284719"],
    ),
]

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
    lost_or_stolen = "lost_or_stolen"


class MonitorSetupReason(StrEnum):
    new_setup = "new_setup"
    replacement = "replacement"


class MonitorSize(StrEnum):
    size_24 = "24_inch"
    size_27 = "27_inch"
    size_32 = "32_inch"


class LaptopOS(StrEnum):
    macos = "macos"
    windows = "windows"


class LaptopSize(StrEnum):
    size_13 = "13_inch"
    size_14 = "14_inch"
    size_16 = "16_inch"


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
    shipping_label = "shipping_label"
    drop_off = "drop_off"


class TroubleshootingCategory(StrEnum):
    login_issue = "login_issue"
    network_connectivity = "network_connectivity"
    hardware_malfunction = "hardware_malfunction"


class TransferReason(StrEnum):
    caller_requested = "caller_requested"
    policy_exception_needed = "policy_exception_needed"
    unable_to_resolve = "unable_to_resolve"
    complaint_escalation = "complaint_escalation"
    technical_issue = "technical_issue"


class SecurityIncidentType(StrEnum):
    lost = "lost"
    stolen = "stolen"
    suspected_compromise = "suspected_compromise"


class WaitlistResourceType(StrEnum):
    desk = "desk"
    parking = "parking"


class InteractionResolutionType(StrEnum):
    """How a resolved-without-ticket call concluded. Used by `mark_resolved`.

    resolved_via_troubleshooting — Issue was fixed during the troubleshooting
        guide (login unlock/reset, hardware power cycle, network reconnect).
        This is the value for Flows 1a, 3a, and 4a.
    transferred — Call was transferred to a live agent before resolution
        (e.g., Flow 20 MFA-reset refusal). Not currently called by any
        automated flow, but reserved for future transfer-logging.
    """

    resolved_via_troubleshooting = "resolved_via_troubleshooting"
    transferred = "transferred"


class InteractionFlowId(StrEnum):
    login_issue = "login_issue"
    network_connectivity = "network_connectivity"
    hardware_malfunction = "hardware_malfunction"


class AccountLockReason(StrEnum):
    failed_attempts = "failed_attempts"
    security_investigation = "security_investigation"
    admin_hold = "admin_hold"


class RoomEquipment(StrEnum):
    projector = "projector"
    whiteboard = "whiteboard"
    video_conferencing = "video_conferencing"
    display_screen = "display_screen"
    speakerphone = "speakerphone"


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
    """Performs standard employee verification AND manager authorization in one call."""

    employee_id: EmployeeIdStr
    phone_last_four: PhoneLastFourStr
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
# Hardware (Flows 5-6, 19)
# ---------------------------------------------------------------------------


class CheckHardwareEntitlementParams(BaseModel):
    employee_id: EmployeeIdStr
    request_type: HardwareRequestType


class SubmitHardwareRequestParams(BaseModel):
    employee_id: EmployeeIdStr
    request_type: HardwareRequestType
    justification: str
    current_asset_tag: AssetTagStr | None = Field(
        default=None,
        description=(
            "Asset tag of the existing device. Required for laptop_replacement (unless justification is "
            "lost_or_stolen) and for monitor_bundle when justification is replacement. **Omit this field** "
            "(do not include the key) for monitor_bundle with justification=new_setup, or when otherwise "
            "not applicable — do not pass null, empty string, or 'N/A'."
        ),
    )
    laptop_os: LaptopOS | None = Field(
        default=None,
        description=(
            "Operating system. Required only when request_type is laptop_replacement. "
            "**Omit this field** for monitor_bundle and other request types."
        ),
    )
    laptop_size: LaptopSize | None = Field(
        default=None,
        description=(
            "Laptop screen size. Required only when request_type is laptop_replacement. "
            "**Omit this field** for monitor_bundle and other request types."
        ),
    )
    monitor_size: MonitorSize | None = Field(
        default=None,
        description=(
            "Monitor size. Required only when request_type is monitor_bundle. "
            "**Omit this field** for laptop_replacement and other request types."
        ),
    )
    delivery_building: BuildingCodeStr
    delivery_floor: FloorCodeStr

    @field_validator("current_asset_tag", "laptop_os", "laptop_size", "monitor_size", mode="before")
    @classmethod
    def _optional_empty_to_none(cls, v):
        return _empty_to_none(v)

    @model_validator(mode="after")
    def _validate_by_request_type(self):
        """Validate required fields per request_type; silently drop inapplicable ones.

        The stored request record always writes `None` for inapplicable fields
        (see itsm_tools.submit_hardware_request), and expected_scenario_db
        reflects that. Silently zeroing extraneous fields here makes the tool
        robust to agents that keep cross-type fields populated on retry, while
        still failing hard when a required field is missing.
        """
        if self.request_type == HardwareRequestType.laptop_replacement:
            if self.justification not in {m.value for m in LaptopReplacementReason}:
                raise ValueError(
                    f"justification '{self.justification}' invalid for laptop_replacement; "
                    f"must be one of {[m.value for m in LaptopReplacementReason]}"
                )
            # Silently drop fields that only apply to monitor_bundle.
            self.monitor_size = None
            # Required fields stay strict.
            if self.laptop_os is None:
                raise ValueError(
                    "laptop_os is required when request_type is laptop_replacement. "
                    "Add 'laptop_os' to your tool call with one of: macos, windows."
                )
            if self.laptop_size is None:
                raise ValueError(
                    "laptop_size is required when request_type is laptop_replacement. "
                    "Add 'laptop_size' to your tool call with one of: 13_inch, 14_inch, 16_inch."
                )
            if self.justification != LaptopReplacementReason.lost_or_stolen.value and self.current_asset_tag is None:
                raise ValueError(
                    "current_asset_tag is required when request_type is laptop_replacement "
                    "(except when justification is lost_or_stolen). "
                    "Add 'current_asset_tag' with the format AST-XXX-NNNNNN (e.g. AST-LPT-284719)."
                )
            if self.justification == LaptopReplacementReason.lost_or_stolen.value:
                # lost_or_stolen reports the stolen device; tag is passed through if provided, else None.
                pass
        elif self.request_type == HardwareRequestType.monitor_bundle:
            if self.justification not in {m.value for m in MonitorSetupReason}:
                raise ValueError(
                    f"justification '{self.justification}' invalid for monitor_bundle; "
                    f"must be one of {[m.value for m in MonitorSetupReason]}"
                )
            # Silently drop fields that only apply to laptop_replacement.
            self.laptop_os = None
            self.laptop_size = None
            # Required fields stay strict.
            if self.monitor_size is None:
                raise ValueError(
                    "monitor_size is required when request_type is monitor_bundle. "
                    "Add 'monitor_size' to your tool call with one of: 24_inch, 27_inch, 32_inch."
                )
            if self.justification == MonitorSetupReason.replacement.value and self.current_asset_tag is None:
                raise ValueError(
                    "current_asset_tag is required when request_type is monitor_bundle and justification is replacement. "
                    "Add 'current_asset_tag' with the format AST-XXX-NNNNNN (e.g. AST-MON-104582)."
                )
            if self.justification == MonitorSetupReason.new_setup.value:
                # new_setup has no existing asset; silently drop any tag the agent supplied.
                self.current_asset_tag = None
        return self


class InitiateAssetReturnParams(BaseModel):
    employee_id: EmployeeIdStr
    asset_tag: AssetTagStr
    request_id: RequestIdStr


class VerifyCostCenterBudgetParams(BaseModel):
    """Auto-fetches department_code and cost_center_code from the employee record."""

    employee_id: EmployeeIdStr


# ---------------------------------------------------------------------------
# Software - access (Flow 7)
# ---------------------------------------------------------------------------


class GetApplicationDetailsParams(BaseModel):
    application_name: CatalogNameStr


class SubmitAccessRequestParams(BaseModel):
    employee_id: EmployeeIdStr
    catalog_id: str = Field(
        pattern=r"^APP-\d{4}$", description="APP-NNNN (obtained from get_application_details)", examples=["APP-0042"]
    )
    access_level: AccessLevel


class RouteApprovalWorkflowParams(BaseModel):
    request_id: RequestIdStr
    employee_id: EmployeeIdStr
    approver_employee_id: EmployeeIdStr


# ---------------------------------------------------------------------------
# Software - license (Flows 8-9)
# ---------------------------------------------------------------------------


class GetLicenseCatalogItemParams(BaseModel):
    license_name: CatalogNameStr


class ValidateCostCenterParams(BaseModel):
    """Auto-fetches department_code and cost_center_code from the employee record."""

    employee_id: EmployeeIdStr


class SubmitLicenseRequestParams(BaseModel):
    employee_id: EmployeeIdStr
    catalog_id: str = Field(
        pattern=r"^LIC-\d{4}$", description="LIC-NNNN (obtained from get_license_catalog_item)", examples=["LIC-0018"]
    )
    duration_days: Literal[30, 60, 90] | None = Field(
        default=None,
        description=(
            "Trial duration in days. Use 30, 60, or 90 for trial licenses. "
            "**Omit this field** (do not include the key) for permanent licenses — "
            "do not pass null, 'null', or empty string."
        ),
    )  # None = permanent

    @field_validator("duration_days", mode="before")
    @classmethod
    def _normalize_duration(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip().lower()
            if s in _EMPTY_SENTINELS:
                return None
            try:
                v = int(s)
            except ValueError:
                return v  # let the Literal validator raise with a clear error
        if isinstance(v, (int, float)) and v == 0:
            return None
        return v


# ---------------------------------------------------------------------------
# Software - renewal (Flow 10)
# ---------------------------------------------------------------------------


class GetEmployeeLicensesParams(BaseModel):
    employee_id: EmployeeIdStr


class SubmitLicenseRenewalParams(BaseModel):
    """Enforces the renewal window directly: within 30 days of expiration or <=14 days expired."""

    employee_id: EmployeeIdStr
    license_assignment_id: LicenseAssignmentIdStr


# ---------------------------------------------------------------------------
# Facilities (Flows 11-14)
# ---------------------------------------------------------------------------


class CheckDeskAvailabilityParams(BaseModel):
    building_code: BuildingCodeStr
    floor_code: FloorCodeStr


class SubmitDeskAssignmentParams(BaseModel):
    employee_id: EmployeeIdStr
    desk_code: DeskCodeStr


class CheckParkingAvailabilityParams(BaseModel):
    zone_code: ParkingZoneStr | None = None

    @field_validator("zone_code", mode="before")
    @classmethod
    def _optional_empty_to_none(cls, v):
        return _empty_to_none(v)


class SubmitParkingAssignmentParams(BaseModel):
    employee_id: EmployeeIdStr
    parking_space_id: ParkingSpaceIdStr


class SubmitWaitlistParams(BaseModel):
    employee_id: EmployeeIdStr
    resource_type: WaitlistResourceType
    zone_or_building: str = Field(
        min_length=1,
        max_length=20,
        description="Building code for desks, zone code for parking",
        examples=["BLD3", "PZA"],
    )


class MarkResolvedParams(BaseModel):
    """X10-R2: final step in resolved-without-ticket flows. Writes an interactions row so state validation can confirm the call closed cleanly."""

    employee_id: EmployeeIdStr
    flow_id: InteractionFlowId
    resolution_type: InteractionResolutionType


class CheckErgonomicAssessmentParams(BaseModel):
    employee_id: EmployeeIdStr


class SubmitEquipmentRequestParams(BaseModel):
    employee_id: EmployeeIdStr
    equipment_type: EquipmentType
    delivery_building: BuildingCodeStr
    delivery_floor: FloorCodeStr


class CheckRoomAvailabilityParams(BaseModel):
    building_code: BuildingCodeStr
    floor_code: FloorCodeStr | None = Field(
        default=None,
        description=(
            "Specific floor code (e.g. 'FL2'). **Omit this field** to search all floors — "
            "do not pass an empty string or null."
        ),
    )
    date: DateStr
    start_time: TimeStr
    end_time: TimeStr
    min_capacity: int = Field(gt=0, le=50)
    equipment_required: list[RoomEquipment] | None = None

    @field_validator("floor_code", mode="before")
    @classmethod
    def _optional_floor_empty_to_none(cls, v):
        return _empty_to_none(v)

    @field_validator("equipment_required", mode="before")
    @classmethod
    def _coerce_equipment(cls, v):
        # Accept a bare string ("whiteboard") or empty sentinel from the agent
        # and normalize to a list. Agents sometimes forget JSON array syntax.
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip().lower()
            if s in _EMPTY_SENTINELS:
                return None
            return [v]
        return v


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


class CheckRoleChangeAuthorizedParams(BaseModel):
    employee_id: EmployeeIdStr
    new_role_code: RoleCodeStr


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
# Extended flows (19-21)
# ---------------------------------------------------------------------------


class ReportSecurityIncidentParams(BaseModel):
    employee_id: EmployeeIdStr
    asset_tag: AssetTagStr
    incident_type: SecurityIncidentType


class InitiateRemoteWipeParams(BaseModel):
    asset_tag: AssetTagStr
    security_case_id: SecurityCaseIdStr


class SubmitMfaResetParams(BaseModel):
    employee_id: EmployeeIdStr
    new_phone_last_four: PhoneLastFourStr


class GetRequestStatusParams(BaseModel):
    request_id: RequestIdStr


class EscalateApprovalParams(BaseModel):
    request_id: RequestIdStr
    escalate_to_employee_id: EmployeeIdStr


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
    "new_phone_last_four": ("invalid_phone_format", "new_phone_last_four"),
    "otp_code": ("invalid_otp_format", "otp_code"),
    "manager_auth_code": ("invalid_manager_auth_code_format", "manager_auth_code"),
    "ticket_number": ("invalid_ticket_number_format", "ticket_number"),
    "asset_tag": ("invalid_asset_tag_format", "asset_tag"),
    "current_asset_tag": ("invalid_asset_tag_format", "current_asset_tag"),
    "catalog_id": ("invalid_catalog_id_format", "catalog_id"),
    "application_name": ("invalid_application_name", "application_name"),
    "license_name": ("invalid_license_name", "license_name"),
    "license_assignment_id": ("invalid_license_assignment_id_format", "license_assignment_id"),
    "building_code": ("invalid_building_code_format", "building_code"),
    "floor_code": ("invalid_floor_code_format", "floor_code"),
    "room_code": ("invalid_room_code_format", "room_code"),
    "desk_code": ("invalid_desk_code_format", "desk_code"),
    "zone_code": ("invalid_parking_zone_format", "zone_code"),
    "parking_space_id": ("invalid_parking_space_id_format", "parking_space_id"),
    "department_code": ("invalid_department_code", "department_code"),
    "role_code": ("invalid_role_code", "role_code"),
    "new_role_code": ("invalid_role_code", "new_role_code"),
    "group_code": ("invalid_group_code_format", "group_code"),
    "permission_template_id": ("invalid_permission_template_id_format", "permission_template_id"),
    "category": ("invalid_incident_category", "category"),
    "urgency": ("invalid_urgency", "urgency"),
    "issue_category": ("invalid_issue_category", "issue_category"),
    "request_type": ("invalid_request_type", "request_type"),
    "justification": ("invalid_justification", "justification"),
    "monitor_size": ("invalid_monitor_size", "monitor_size"),
    "laptop_os": ("invalid_laptop_os", "laptop_os"),
    "laptop_size": ("invalid_laptop_size", "laptop_size"),
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
    "incident_type": ("invalid_security_incident_type", "incident_type"),
    "resource_type": ("invalid_resource_type", "resource_type"),
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
    "security_case_id": ("invalid_security_case_id_format", "security_case_id"),
    "new_hire_employee_id": ("invalid_employee_id_format", "new_hire_employee_id"),
    "manager_employee_id": ("invalid_employee_id_format", "manager_employee_id"),
    "departing_employee_id": ("invalid_employee_id_format", "departing_employee_id"),
    "approver_employee_id": ("invalid_employee_id_format", "approver_employee_id"),
    "escalate_to_employee_id": ("invalid_employee_id_format", "escalate_to_employee_id"),
    "delivery_building": ("invalid_building_code_format", "delivery_building"),
    "delivery_floor": ("invalid_floor_code_format", "delivery_floor"),
}


# Per-field hints appended to validation errors, to steer the agent toward the
# correct fix when the raw pydantic message is ambiguous or misleading (e.g.
# Optional enum fields that must be OMITTED, not sent as empty string).
FIELD_ERROR_HINTS: dict[str, str] = {
    "monitor_size": (
        "Required for monitor_bundle (one of 24_inch/27_inch/32_inch). "
        "OMIT this field entirely (do not pass empty string) for laptop_replacement."
    ),
    "laptop_os": ("Required for laptop_replacement (macos or windows). OMIT this field entirely for monitor_bundle."),
    "laptop_size": (
        "Required for laptop_replacement (13_inch, 14_inch, or 16_inch). OMIT this field entirely for monitor_bundle."
    ),
    "current_asset_tag": (
        "Required for laptop_replacement unless justification is lost_or_stolen. "
        "Required for monitor_bundle replacement; OMIT for monitor_bundle new_setup."
    ),
    "duration_days": (
        "OMIT duration_days for a permanent license. Valid values for temporary licenses are exactly 30, 60, or 90."
    ),
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
                    if fi.examples:
                        msg += f" (e.g. {', '.join(str(e) for e in fi.examples)})"
                elif detail := error.get("msg", ""):
                    msg += f": {detail}"
                if hint := FIELD_ERROR_HINTS.get(field):
                    msg += f". {hint}"
                return {"status": "error", "error_type": error_type, "message": msg}
    # Fallback path — typically `@model_validator` errors where loc is empty.
    # Scan the message for any FIELD_ERROR_HINTS key and append the matching hint
    # so cross-field rules (e.g. "omit monitor_size for laptop_replacement") still
    # carry the directive guidance.
    first = exc.errors()[0] if exc.errors() else {}
    loc = first.get("loc", ("parameter",))
    raw_msg = first.get("msg", str(exc))
    base = f"Invalid '{loc[0] if loc else 'parameter'}': {raw_msg}"
    for field, hint in FIELD_ERROR_HINTS.items():
        if field in raw_msg and hint not in base:
            base += f" {hint}"
            break
    return {
        "status": "error",
        "error_type": "invalid_parameter",
        "message": base,
    }
