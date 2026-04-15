"""ITSM agent tool functions — see itsm_params.py for flow sequences."""

import copy
from datetime import datetime as _dt, timedelta
from pydantic import ValidationError
from eva.assistant.tools.itsm_params import (
    VerifyEmployeeAuthParams, InitiateOtpAuthParams, VerifyOtpAuthParams, VerifyManagerAuthParams,
    GetEmployeeRecordParams, GetAssetRecordParams, GetEmployeeAssetsParams,
    GetTroubleshootingGuideParams, AttemptAccountUnlockParams, AttemptPasswordResetParams,
    CreateIncidentTicketParams, AssignSLATierParams,
    CheckExistingOutageParams, AddAffectedUserParams, LinkKnownErrorParams,
    ScheduleFieldDispatchParams, AttachDiagnosticLogParams,
    CheckHardwareEntitlementParams, SubmitHardwareRequestParams, InitiateAssetReturnParams, VerifyCostCenterBudgetParams,
    GetApplicationDetailsParams, SubmitAccessRequestParams, RouteApprovalWorkflowParams,
    GetLicenseCatalogItemParams, ValidateCostCenterParams, SubmitLicenseRequestParams, SubmitTemporaryLicenseParams,
    GetEmployeeLicensesParams, CheckRenewalEligibilityParams, SubmitLicenseRenewalParams,
    CheckSpaceAvailabilityParams, SubmitDeskAssignmentParams,
    CheckParkingAvailabilityParams, SubmitParkingAssignmentParams,
    CheckErgonomicAssessmentParams, SubmitEquipmentRequestParams,
    CheckRoomAvailabilityParams, SubmitRoomBookingParams, SendCalendarInviteParams,
    LookupNewHireParams, CheckExistingAccountsParams, ProvisionNewAccountParams,
    GetGroupMembershipsParams, GetGroupDetailsParams, SubmitGroupMembershipChangeParams,
    GetPermissionTemplatesParams, SubmitPermissionChangeParams, ScheduleAccessReviewParams,
    GetOffboardingRecordParams, SubmitAccessRemovalParams, InitiateAssetRecoveryParams,
    TransferToAgentParams, validation_error_response,
)

def _make_request_id(cat, eid): return f"REQ-{cat}-{eid[-6:]}"
def _make_case_id(cat, eid): return f"CASE-{cat}-{eid[-6:]}"
def _enf(eid): return {"status":"error","error_type":"not_found","message":f"Employee {eid} not found"}
def _ar(t="employee_auth"): return {"status":"error","error_type":"authentication_required","message":f"Authentication ({t}) required"}
def _ok(db, k): return db.get("session",{}).get(k) is True

# ═══════════════════════════════════════════════════════════════════════════════
# AUTH
# ═══════════════════════════════════════════════════════════════════════════════

def verify_employee_auth(params, db, call_index):
    try: p = VerifyEmployeeAuthParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, VerifyEmployeeAuthParams)
    emp = db.get("employees",{}).get(p.employee_id)
    if not emp: return _enf(p.employee_id)
    if emp.get("phone_last_four") != p.phone_last_four:
        return {"status":"error","error_type":"authentication_failed","message":"Phone number does not match"}
    db.setdefault("session",{})["employee_auth"] = True
    db["session"]["authenticated_employee_id"] = p.employee_id
    return {"status":"success","authenticated":True,"employee_id":p.employee_id,"first_name":emp.get("first_name"),"last_name":emp.get("last_name"),"message":f"Employee {p.employee_id} authenticated"}

def initiate_otp_auth(params, db, call_index):
    try: p = InitiateOtpAuthParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, InitiateOtpAuthParams)
    if not _ok(db,"employee_auth"): return _ar()
    emp = db.get("employees",{}).get(p.employee_id)
    if not emp: return _enf(p.employee_id)
    db.setdefault("session",{}).update({"otp_employee_id":p.employee_id,"otp_issued":True})
    return {"status":"success","phone_last_four":emp.get("phone_last_four"),"message":f"OTP sent to ***{emp.get('phone_last_four')}"}

def verify_otp_auth(params, db, call_index):
    try: p = VerifyOtpAuthParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, VerifyOtpAuthParams)
    if not db.get("session",{}).get("otp_issued"): return {"status":"error","error_type":"otp_not_initiated","message":"Call initiate_otp_auth first"}
    emp = db.get("employees",{}).get(p.employee_id)
    if not emp: return _enf(p.employee_id)
    if emp.get("otp_code") != p.otp_code: return {"status":"error","error_type":"authentication_failed","message":"OTP does not match"}
    db["session"]["otp_auth"] = True
    return {"status":"success","authenticated":True,"employee_id":p.employee_id,"message":"OTP verified"}

def verify_manager_auth(params, db, call_index):
    try: p = VerifyManagerAuthParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, VerifyManagerAuthParams)
    if not _ok(db,"employee_auth"): return _ar()
    emp = db.get("employees",{}).get(p.employee_id)
    if not emp: return _enf(p.employee_id)
    if emp.get("manager_auth_code") != p.manager_auth_code: return {"status":"error","error_type":"authentication_failed","message":"Manager auth code does not match"}
    if not emp.get("is_manager"): return {"status":"error","error_type":"not_authorized","message":f"{p.employee_id} is not a manager"}
    db["session"]["manager_auth"] = True
    db["session"]["manager_employee_id"] = p.employee_id
    return {"status":"success","confirmed":True,"employee_id":p.employee_id,"department_code":emp.get("department_code"),"direct_reports":emp.get("direct_reports",[]),"message":"Manager auth confirmed"}

# ═══════════════════════════════════════════════════════════════════════════════
# SHARED LOOKUPS
# ═══════════════════════════════════════════════════════════════════════════════

def get_employee_record(params, db, call_index):
    try: p = GetEmployeeRecordParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, GetEmployeeRecordParams)
    if not _ok(db,"employee_auth"): return _ar()
    emp = db.get("employees",{}).get(p.employee_id)
    if not emp: return _enf(p.employee_id)
    fields = ["employee_id","first_name","last_name","department_code","role_code","hire_date","employment_status","building_code","floor_code","manager_employee_id"]
    return {"status":"success","employee":{k:emp[k] for k in fields if k in emp}}

def get_asset_record(params, db, call_index):
    try: p = GetAssetRecordParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, GetAssetRecordParams)
    if not _ok(db,"employee_auth"): return _ar()
    asset = db.get("assets",{}).get(p.asset_tag)
    if not asset: return {"status":"error","error_type":"not_found","message":f"Asset {p.asset_tag} not found"}
    return {"status":"success","asset":copy.deepcopy(asset)}

def get_employee_assets(params, db, call_index):
    try: p = GetEmployeeAssetsParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, GetEmployeeAssetsParams)
    if not _ok(db,"employee_auth"): return _ar()
    assets = [copy.deepcopy(a) for a in db.get("assets",{}).values() if a.get("assigned_employee_id")==p.employee_id]
    return {"status":"success","employee_id":p.employee_id,"assets":assets,"message":f"{len(assets)} asset(s)"}

# ═══════════════════════════════════════════════════════════════════════════════
# FLOW 1: LOGIN — troubleshooting + direct resolution
# ═══════════════════════════════════════════════════════════════════════════════

def get_troubleshooting_guide(params, db, call_index):
    try: p = GetTroubleshootingGuideParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, GetTroubleshootingGuideParams)
    if not _ok(db,"employee_auth"): return _ar()
    guide = db.get("troubleshooting_guides",{}).get(p.issue_category)
    if not guide: return {"status":"error","error_type":"guide_not_found","message":f"No guide for {p.issue_category}"}
    return {"status":"success","issue_category":p.issue_category,"steps":copy.deepcopy(guide["steps"]),"message":f"{len(guide['steps'])} steps"}

def attempt_account_unlock(params, db, call_index):
    """Attempt to unlock a locked account on a target system."""
    try: p = AttemptAccountUnlockParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, AttemptAccountUnlockParams)
    if not _ok(db,"employee_auth"): return _ar()
    emp = db.get("employees",{}).get(p.employee_id)
    if not emp: return _enf(p.employee_id)
    acct_status = emp.get("account_status",{}).get(p.target_system)
    if not acct_status:
        return {"status":"error","error_type":"no_account","message":f"No account found on {p.target_system} for {p.employee_id}"}
    if acct_status.get("locked") is not True:
        return {"status":"error","error_type":"not_locked","message":f"Account on {p.target_system} is not locked"}
    acct_status["locked"] = False
    return {"status":"success","employee_id":p.employee_id,"target_system":p.target_system,"resolved":True,"message":f"Account unlocked on {p.target_system}"}

def attempt_password_reset(params, db, call_index):
    """Initiate a password reset on a target system."""
    try: p = AttemptPasswordResetParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, AttemptPasswordResetParams)
    if not _ok(db,"employee_auth"): return _ar()
    emp = db.get("employees",{}).get(p.employee_id)
    if not emp: return _enf(p.employee_id)
    acct_status = emp.get("account_status",{}).get(p.target_system)
    if not acct_status:
        return {"status":"error","error_type":"no_account","message":f"No account on {p.target_system} for {p.employee_id}"}
    if acct_status.get("requires_in_person_reset"):
        return {"status":"error","error_type":"in_person_required","message":f"Password reset for {p.target_system} requires in-person verification at the IT security office"}
    acct_status["password_expired"] = False
    acct_status["temp_password_issued"] = True
    return {"status":"success","employee_id":p.employee_id,"target_system":p.target_system,"resolved":True,"message":f"Temporary password sent to phone on file. Caller must change on first login."}

# ═══════════════════════════════════════════════════════════════════════════════
# INCIDENT CREATION + SLA (Flows 1-4)
# ═══════════════════════════════════════════════════════════════════════════════

def create_incident_ticket(params, db, call_index):
    try: p = CreateIncidentTicketParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, CreateIncidentTicketParams)
    if not _ok(db,"employee_auth"): return _ar()
    if p.category in ("login_issue","network_connectivity") and not p.troubleshooting_completed:
        return {"status":"error","error_type":"troubleshooting_required","message":f"Complete troubleshooting before creating {p.category} ticket"}
    ctr = db.get("_ticket_counter",48270)+1; db["_ticket_counter"] = ctr
    tn = f"INC{str(ctr).zfill(7)}"
    db.setdefault("tickets",{})[tn] = {"ticket_number":tn,"employee_id":p.employee_id,"category":p.category,"urgency":p.urgency,"affected_system":p.affected_system,"troubleshooting_completed":p.troubleshooting_completed,"status":"open","sla_tier":None,"created_date":db.get("_current_date","")}
    return {"status":"success","ticket_number":tn,"category":p.category,"urgency":p.urgency,"message":f"Ticket created: {tn}"}

def assign_sla_tier(params, db, call_index):
    try: p = AssignSLATierParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, AssignSLATierParams)
    if not _ok(db,"employee_auth"): return _ar()
    ticket = db.get("tickets",{}).get(p.ticket_number)
    if not ticket: return {"status":"error","error_type":"not_found","message":f"Ticket {p.ticket_number} not found"}
    sla = {"tier_1":{"resp":1,"res":4},"tier_2":{"resp":4,"res":8},"tier_3":{"resp":8,"res":24}}[p.sla_tier]
    ticket.update({"sla_tier":p.sla_tier,"sla_response_hours":sla["resp"],"sla_resolution_hours":sla["res"]})
    return {"status":"success","ticket_number":p.ticket_number,"sla_tier":p.sla_tier,"response_target":f"{sla['resp']}h","resolution_target":f"{sla['res']}h","message":f"SLA {p.sla_tier}: respond {sla['resp']}h, resolve {sla['res']}h"}

# ═══════════════════════════════════════════════════════════════════════════════
# FLOW 2: OUTAGE
# ═══════════════════════════════════════════════════════════════════════════════

def check_existing_outage(params, db, call_index):
    try: p = CheckExistingOutageParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, CheckExistingOutageParams)
    if not _ok(db,"employee_auth"): return _ar()
    outage = db.get("active_outages",{}).get(p.service_name)
    if not outage: return {"status":"success","existing_outage":False,"service_name":p.service_name,"message":"No active outage found."}
    return {"status":"success","existing_outage":True,"ticket_number":outage["ticket_number"],"service_name":p.service_name,"affected_user_count":outage.get("affected_user_count",0),"message":f"Active outage: {outage['ticket_number']}"}

def add_affected_user(params, db, call_index):
    try: p = AddAffectedUserParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, AddAffectedUserParams)
    if not _ok(db,"employee_auth"): return _ar()
    outage = next((o for o in db.get("active_outages",{}).values() if o.get("ticket_number")==p.ticket_number), None)
    if not outage: return {"status":"error","error_type":"ticket_not_found","message":f"No active outage {p.ticket_number}"}
    affected = outage.setdefault("affected_users",[])
    if p.employee_id not in affected: affected.append(p.employee_id)
    outage["affected_user_count"] = len(affected)
    return {"status":"success","ticket_number":p.ticket_number,"employee_id":p.employee_id,"total_affected":len(affected),"message":f"Added to {p.ticket_number}"}

def link_known_error(params, db, call_index):
    try: p = LinkKnownErrorParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, LinkKnownErrorParams)
    if not _ok(db,"employee_auth"): return _ar()
    ticket = db.get("tickets",{}).get(p.ticket_number)
    if not ticket: return {"status":"error","error_type":"not_found","message":f"Ticket {p.ticket_number} not found"}
    ke = db.get("known_errors",{}).get(p.service_name)
    if not ke: return {"status":"success","linked":False,"ticket_number":p.ticket_number,"message":"No known error found."}
    ticket["linked_known_error_id"] = ke["known_error_id"]
    return {"status":"success","linked":True,"ticket_number":p.ticket_number,"known_error_id":ke["known_error_id"],"workaround":ke.get("workaround"),"message":f"Linked {ke['known_error_id']}"}

# ═══════════════════════════════════════════════════════════════════════════════
# FLOW 3: HARDWARE MALFUNCTION — field dispatch
# ═══════════════════════════════════════════════════════════════════════════════

def schedule_field_dispatch(params, db, call_index):
    try: p = ScheduleFieldDispatchParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, ScheduleFieldDispatchParams)
    if not _ok(db,"employee_auth"): return _ar()
    ticket = db.get("tickets",{}).get(p.ticket_number)
    if not ticket: return {"status":"error","error_type":"not_found","message":f"Ticket {p.ticket_number} not found"}
    avail = db.get("field_dispatch_availability",{})
    window = avail.get(p.preferred_date,{}).get(p.time_window)
    if not window or not window.get("available"):
        alts = [{"date":d,"time_window":w} for d,ws in sorted(avail.items()) if d>=p.preferred_date for w,info in ws.items() if info.get("available")][:3]
        return {"status":"error","error_type":"no_availability","message":f"No tech on {p.preferred_date} {p.time_window}","alternative_slots":alts}
    dsp = f"DSP-{p.ticket_number[-7:]}"; window["available"] = False
    ticket.update({"dispatch_id":dsp,"dispatch_date":p.preferred_date,"dispatch_window":p.time_window})
    return {"status":"success","dispatch_id":dsp,"date":p.preferred_date,"time_window":p.time_window,"message":f"Dispatch {dsp} confirmed"}

# ═══════════════════════════════════════════════════════════════════════════════
# FLOW 4: NETWORK — diagnostic log
# ═══════════════════════════════════════════════════════════════════════════════

def attach_diagnostic_log(params, db, call_index):
    try: p = AttachDiagnosticLogParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, AttachDiagnosticLogParams)
    if not _ok(db,"employee_auth"): return _ar()
    ticket = db.get("tickets",{}).get(p.ticket_number)
    if not ticket: return {"status":"error","error_type":"not_found","message":f"Ticket {p.ticket_number} not found"}
    ticket.update({"diagnostic_ref_code":p.diagnostic_ref_code,"diagnostic_attached":True})
    return {"status":"success","ticket_number":p.ticket_number,"diagnostic_ref_code":p.diagnostic_ref_code,"message":f"Diagnostic attached"}

# ═══════════════════════════════════════════════════════════════════════════════
# FLOWS 5-6: HARDWARE REQUESTS
# ═══════════════════════════════════════════════════════════════════════════════

def check_hardware_entitlement(params, db, call_index):
    try: p = CheckHardwareEntitlementParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, CheckHardwareEntitlementParams)
    if not _ok(db,"employee_auth"): return _ar()
    emp = db.get("employees",{}).get(p.employee_id)
    if not emp: return _enf(p.employee_id)
    ent = emp.get("hardware_entitlements",{}).get(p.request_type)
    if not ent: return {"status":"error","error_type":"no_entitlement","message":f"No entitlement for {p.request_type}"}
    if ent.get("pending_request"): return {"status":"error","error_type":"request_already_pending","message":f"Pending: {ent.get('pending_request_id')}"}
    return {"status":"success","eligible":True,"employee_id":p.employee_id,"request_type":p.request_type,"current_asset_tag":ent.get("current_asset_tag"),"device_age_months":ent.get("device_age_months"),"message":"Eligible"}

def submit_hardware_request(params, db, call_index):
    try: p = SubmitHardwareRequestParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, SubmitHardwareRequestParams)
    if not _ok(db,"employee_auth"): return _ar()
    if p.request_type=="laptop_replacement":
        if p.replacement_reason not in {"end_of_life","performance_degradation","physical_damage"}:
            return {"status":"error","error_type":"invalid_replacement_reason","message":f"Invalid: {p.replacement_reason}"}
        if not p.current_asset_tag: return {"status":"error","error_type":"missing_asset_tag","message":"current_asset_tag required"}
    elif p.request_type=="monitor_bundle":
        if p.replacement_reason not in {"new_setup","replacement"}:
            return {"status":"error","error_type":"invalid_replacement_reason","message":f"Invalid: {p.replacement_reason}"}
        if not p.monitor_size: return {"status":"error","error_type":"missing_monitor_size","message":"monitor_size required"}
    rid = _make_request_id("HW",p.employee_id)
    db.setdefault("requests",{})[rid] = {"request_id":rid,"employee_id":p.employee_id,"request_type":p.request_type,"replacement_reason":p.replacement_reason,"current_asset_tag":p.current_asset_tag,"monitor_size":p.monitor_size,"delivery_building":p.delivery_building,"delivery_floor":p.delivery_floor,"status":"submitted","created_date":db.get("_current_date","")}
    ent = db.get("employees",{}).get(p.employee_id,{}).get("hardware_entitlements",{}).get(p.request_type)
    if ent: ent.update({"pending_request":True,"pending_request_id":rid})
    return {"status":"success","request_id":rid,"request_type":p.request_type,"message":f"Submitted: {rid}"}

def initiate_asset_return(params, db, call_index):
    try: p = InitiateAssetReturnParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, InitiateAssetReturnParams)
    if not _ok(db,"employee_auth"): return _ar()
    asset = db.get("assets",{}).get(p.asset_tag)
    if not asset: return {"status":"error","error_type":"not_found","message":f"Asset {p.asset_tag} not found"}
    deadline = (_dt.strptime(db.get("_current_date","2026-08-12"),"%Y-%m-%d")+timedelta(days=14)).strftime("%Y-%m-%d")
    rma = f"RMA-{p.asset_tag[-6:]}"
    asset["return_authorization"] = {"return_auth_id":rma,"request_id":p.request_id,"return_deadline":deadline,"status":"pending_return"}
    return {"status":"success","return_auth_id":rma,"asset_tag":p.asset_tag,"return_deadline":deadline,"message":f"RMA {rma}: return by {deadline}"}

def verify_cost_center_budget(params, db, call_index):
    """Pre-check: verify cost center has budget before submitting hardware request."""
    try: p = VerifyCostCenterBudgetParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, VerifyCostCenterBudgetParams)
    if not _ok(db,"employee_auth"): return _ar()
    cc = db.get("cost_centers",{}).get(p.cost_center_code)
    if not cc: return {"status":"error","error_type":"not_found","message":f"Cost center {p.cost_center_code} not found"}
    if cc.get("department_code") != p.department_code:
        return {"status":"error","error_type":"department_mismatch","message":f"{p.cost_center_code} belongs to {cc['department_code']}, not {p.department_code}"}
    if not cc.get("has_budget",True):
        return {"status":"error","error_type":"insufficient_budget","message":f"Cost center {p.cost_center_code} has insufficient budget"}
    return {"status":"success","cost_center_code":p.cost_center_code,"has_budget":True,"remaining_budget_usd":cc.get("remaining_budget_usd",0),"message":"Budget verified"}

# ═══════════════════════════════════════════════════════════════════════════════
# FLOW 7: APP ACCESS
# ═══════════════════════════════════════════════════════════════════════════════

def get_application_details(params, db, call_index):
    try: p = GetApplicationDetailsParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, GetApplicationDetailsParams)
    if not _ok(db,"employee_auth"): return _ar()
    app = db.get("software_catalog",{}).get("applications",{}).get(p.catalog_id)
    if not app: return {"status":"error","error_type":"not_found","message":f"App {p.catalog_id} not found"}
    return {"status":"success","application":copy.deepcopy(app)}

def submit_access_request(params, db, call_index):
    try: p = SubmitAccessRequestParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, SubmitAccessRequestParams)
    if not _ok(db,"employee_auth"): return _ar()
    app = db.get("software_catalog",{}).get("applications",{}).get(p.catalog_id)
    if not app: return {"status":"error","error_type":"not_found","message":f"App {p.catalog_id} not found"}
    if p.access_level not in app.get("available_access_levels",[]):
        return {"status":"error","error_type":"invalid_access_level","message":f"'{p.access_level}' not available. Options: {app['available_access_levels']}"}
    approval = app.get("requires_manager_approval",False)
    rid = _make_request_id("SW",p.employee_id)
    db.setdefault("requests",{})[rid] = {"request_id":rid,"employee_id":p.employee_id,"catalog_id":p.catalog_id,"application_name":app["name"],"access_level":p.access_level,"status":"pending_approval" if approval else "approved","requires_manager_approval":approval,"created_date":db.get("_current_date","")}
    return {"status":"success","request_id":rid,"application_name":app["name"],"access_level":p.access_level,"requires_approval":approval,"message":f"{rid}" + (" (pending approval)" if approval else " (auto-approved)")}

def route_approval_workflow(params, db, call_index):
    try: p = RouteApprovalWorkflowParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, RouteApprovalWorkflowParams)
    if not _ok(db,"employee_auth"): return _ar()
    req = db.get("requests",{}).get(p.request_id)
    if not req: return {"status":"error","error_type":"not_found","message":f"Request {p.request_id} not found"}
    approver = db.get("employees",{}).get(p.approver_employee_id)
    if not approver: return _enf(p.approver_employee_id)
    deadline = (_dt.strptime(db.get("_current_date","2026-08-12"),"%Y-%m-%d")+timedelta(hours=48)).strftime("%Y-%m-%d %H:%M")
    req.update({"approval_routed_to":p.approver_employee_id,"approval_sla_deadline":deadline})
    return {"status":"success","request_id":p.request_id,"approver_name":f"{approver['first_name']} {approver['last_name']}","approval_deadline":deadline,"message":f"Routed to {approver['first_name']} {approver['last_name']}. 48h window."}

# ═══════════════════════════════════════════════════════════════════════════════
# FLOWS 8-9: LICENSE
# ═══════════════════════════════════════════════════════════════════════════════

def get_license_catalog_item(params, db, call_index):
    try: p = GetLicenseCatalogItemParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, GetLicenseCatalogItemParams)
    if not _ok(db,"employee_auth"): return _ar()
    lic = db.get("software_catalog",{}).get("licenses",{}).get(p.catalog_id)
    if not lic: return {"status":"error","error_type":"not_found","message":f"License {p.catalog_id} not found"}
    return {"status":"success","license":copy.deepcopy(lic)}

def validate_cost_center(params, db, call_index):
    """Pre-check: validate cost center before submitting license request."""
    try: p = ValidateCostCenterParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, ValidateCostCenterParams)
    if not _ok(db,"employee_auth"): return _ar()
    cc = db.get("cost_centers",{}).get(p.cost_center_code)
    if not cc: return {"status":"error","error_type":"not_found","message":f"Cost center {p.cost_center_code} not found"}
    if cc.get("department_code") != p.department_code:
        return {"status":"error","error_type":"department_mismatch","message":f"{p.cost_center_code} not for {p.department_code}"}
    return {"status":"success","cost_center_code":p.cost_center_code,"validated":True,"message":f"Cost center {p.cost_center_code} validated"}

def submit_license_request(params, db, call_index):
    try: p = SubmitLicenseRequestParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, SubmitLicenseRequestParams)
    if not _ok(db,"employee_auth"): return _ar()
    rid = _make_request_id("SW",p.employee_id)
    db.setdefault("requests",{})[rid] = {"request_id":rid,"employee_id":p.employee_id,"catalog_id":p.catalog_id,"request_type":"license_request","status":"submitted","created_date":db.get("_current_date","")}
    return {"status":"success","request_id":rid,"catalog_id":p.catalog_id,"message":f"License request: {rid}"}

def submit_temporary_license(params, db, call_index):
    try: p = SubmitTemporaryLicenseParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, SubmitTemporaryLicenseParams)
    if not _ok(db,"employee_auth"): return _ar()
    exp = (_dt.strptime(db.get("_current_date","2026-08-01"),"%Y-%m-%d")+timedelta(days=p.duration_days)).strftime("%Y-%m-%d")
    rid = _make_request_id("SW",p.employee_id)
    db.setdefault("requests",{})[rid] = {"request_id":rid,"employee_id":p.employee_id,"catalog_id":p.catalog_id,"request_type":"temporary_license","duration_days":p.duration_days,"expiration_date":exp,"status":"submitted","created_date":db.get("_current_date","")}
    return {"status":"success","request_id":rid,"duration_days":p.duration_days,"expiration_date":exp,"message":f"Temp license: {rid}. Expires {exp}."}

# ═══════════════════════════════════════════════════════════════════════════════
# FLOW 10: RENEWAL
# ═══════════════════════════════════════════════════════════════════════════════

def get_employee_licenses(params, db, call_index):
    try: p = GetEmployeeLicensesParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, GetEmployeeLicensesParams)
    if not _ok(db,"employee_auth"): return _ar()
    emp = db.get("employees",{}).get(p.employee_id)
    if not emp: return _enf(p.employee_id)
    return {"status":"success","employee_id":p.employee_id,"licenses":copy.deepcopy(emp.get("software_licenses",[]))}

def check_renewal_eligibility(params, db, call_index):
    try: p = CheckRenewalEligibilityParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, CheckRenewalEligibilityParams)
    if not _ok(db,"employee_auth"): return _ar()
    emp = db.get("employees",{}).get(p.employee_id)
    if not emp: return _enf(p.employee_id)
    lic = next((l for l in emp.get("software_licenses",[]) if l["license_assignment_id"]==p.license_assignment_id), None)
    if not lic: return {"status":"error","error_type":"not_found","message":f"License {p.license_assignment_id} not found"}
    cur, exp = db.get("_current_date",""), lic.get("expiration_date","")
    if cur and exp:
        days = (_dt.strptime(exp,"%Y-%m-%d")-_dt.strptime(cur,"%Y-%m-%d")).days
        if days > 30: return {"status":"error","error_type":"renewal_too_early","message":f"Expires in {days} days. Available within 30."}
        if days < -14: return {"status":"error","error_type":"renewal_expired","message":f"Expired {abs(days)} days ago. Submit new request."}
    return {"status":"success","eligible":True,"license_assignment_id":p.license_assignment_id,"software_name":lic.get("software_name"),"expiration_date":exp}

def submit_license_renewal(params, db, call_index):
    try: p = SubmitLicenseRenewalParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, SubmitLicenseRenewalParams)
    if not _ok(db,"employee_auth"): return _ar()
    emp = db.get("employees",{}).get(p.employee_id)
    if not emp: return _enf(p.employee_id)
    lic = next((l for l in emp.get("software_licenses",[]) if l["license_assignment_id"]==p.license_assignment_id), None)
    if not lic: return {"status":"error","error_type":"not_found","message":"License not found"}
    cur = db.get("_current_date","2026-08-01")
    new_exp = (max(_dt.strptime(lic.get("expiration_date",cur),"%Y-%m-%d"),_dt.strptime(cur,"%Y-%m-%d"))+timedelta(days=365)).strftime("%Y-%m-%d")
    lic.update({"expiration_date":new_exp,"status":"active"})
    rid = _make_request_id("SW",p.employee_id)
    return {"status":"success","request_id":rid,"new_expiration_date":new_exp,"message":f"Renewed. Expires {new_exp}. ID: {rid}"}

# ═══════════════════════════════════════════════════════════════════════════════
# FLOWS 11-14: FACILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def check_space_availability(params, db, call_index):
    try: p = CheckSpaceAvailabilityParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, CheckSpaceAvailabilityParams)
    if not _ok(db,"employee_auth"): return _ar()
    desks = db.get("facilities",{}).get("desks",{})
    avail = [{"desk_code":c,"zone":d.get("zone"),"near_window":d.get("near_window",False)} for c,d in desks.items() if d.get("building_code")==p.building_code and d.get("floor_code")==p.floor_code and d.get("status")=="available"]
    return {"status":"success","available_desks":avail,"message":f"{len(avail)} desk(s)"}

def submit_desk_assignment(params, db, call_index):
    try: p = SubmitDeskAssignmentParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, SubmitDeskAssignmentParams)
    if not _ok(db,"employee_auth"): return _ar()
    desk = db.get("facilities",{}).get("desks",{}).get(p.desk_code)
    if not desk: return {"status":"error","error_type":"not_found","message":f"Desk {p.desk_code} not found"}
    if desk["status"] != "available": return {"status":"error","error_type":"not_available","message":"Not available"}
    desk.update({"status":"assigned","assigned_employee_id":p.employee_id})
    return {"status":"success","request_id":_make_request_id("FAC",p.employee_id),"desk_code":p.desk_code,"message":f"Assigned {p.desk_code}"}

def check_parking_availability(params, db, call_index):
    try: p = CheckParkingAvailabilityParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, CheckParkingAvailabilityParams)
    if not _ok(db,"employee_auth"): return _ar()
    spaces = db.get("facilities",{}).get("parking",{})
    avail = [{"parking_space_id":sid,"level":s.get("level"),"covered":s.get("covered",False)} for sid,s in spaces.items() if s.get("zone_code")==p.zone_code and s.get("status")=="available"]
    return {"status":"success","zone_code":p.zone_code,"available_spaces":avail,"message":f"{len(avail)} space(s)"}

def submit_parking_assignment(params, db, call_index):
    try: p = SubmitParkingAssignmentParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, SubmitParkingAssignmentParams)
    if not _ok(db,"employee_auth"): return _ar()
    space = db.get("facilities",{}).get("parking",{}).get(p.parking_space_id)
    if not space: return {"status":"error","error_type":"not_found","message":"Space not found"}
    if space["status"] != "available": return {"status":"error","error_type":"not_available","message":"Not available"}
    space.update({"status":"assigned","assigned_employee_id":p.employee_id})
    return {"status":"success","request_id":_make_request_id("FAC",p.employee_id),"parking_space_id":p.parking_space_id,"message":f"Assigned {p.parking_space_id}"}

def check_ergonomic_assessment(params, db, call_index):
    try: p = CheckErgonomicAssessmentParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, CheckErgonomicAssessmentParams)
    if not _ok(db,"employee_auth"): return _ar()
    emp = db.get("employees",{}).get(p.employee_id)
    if not emp: return _enf(p.employee_id)
    a = emp.get("ergonomic_assessment")
    if not a or a.get("status") != "completed":
        return {"status":"error","error_type":"assessment_required","message":"No completed ergonomic assessment. Complete one at occupational health portal."}
    return {"status":"success","employee_id":p.employee_id,"assessment_date":a.get("completion_date"),"message":"Assessment on file"}

def submit_equipment_request(params, db, call_index):
    try: p = SubmitEquipmentRequestParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, SubmitEquipmentRequestParams)
    if not _ok(db,"employee_auth"): return _ar()
    rid = _make_request_id("FAC",p.employee_id)
    db.setdefault("requests",{})[rid] = {"request_id":rid,"employee_id":p.employee_id,"equipment_type":p.equipment_type,"delivery_building":p.delivery_building,"delivery_floor":p.delivery_floor,"status":"submitted","created_date":db.get("_current_date","")}
    return {"status":"success","request_id":rid,"equipment_type":p.equipment_type,"message":f"Equipment request: {rid}"}

def check_room_availability(params, db, call_index):
    try: p = CheckRoomAvailabilityParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, CheckRoomAvailabilityParams)
    if not _ok(db,"employee_auth"): return _ar()
    rooms = db.get("facilities",{}).get("conference_rooms",{})
    avail = []
    for rc, rm in rooms.items():
        if rm.get("building_code")==p.building_code and rm.get("floor_code")==p.floor_code and rm.get("capacity",0)>=p.min_capacity:
            if not any(b["date"]==p.date and not(p.end_time<=b.get("start_time","") or p.start_time>=b.get("end_time","")) for b in rm.get("bookings",[])):
                avail.append({"room_code":rc,"capacity":rm["capacity"],"equipment":rm.get("equipment",[])})
    return {"status":"success","available_rooms":avail,"message":f"{len(avail)} room(s)"}

def submit_room_booking(params, db, call_index):
    try: p = SubmitRoomBookingParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, SubmitRoomBookingParams)
    if not _ok(db,"employee_auth"): return _ar()
    room = db.get("facilities",{}).get("conference_rooms",{}).get(p.room_code)
    if not room: return {"status":"error","error_type":"not_found","message":f"Room {p.room_code} not found"}
    for b in room.get("bookings",[]):
        if b["date"]==p.date and not(p.end_time<=b.get("start_time","") or p.start_time>=b.get("end_time","")):
            return {"status":"error","error_type":"room_conflict","message":f"Booked {p.date} {b['start_time']}-{b['end_time']}"}
    rid = _make_request_id("FAC",p.employee_id)
    room.setdefault("bookings",[]).append({"booking_id":rid,"date":p.date,"start_time":p.start_time,"end_time":p.end_time,"employee_id":p.employee_id,"attendee_count":p.attendee_count})
    return {"status":"success","request_id":rid,"room_code":p.room_code,"date":p.date,"start_time":p.start_time,"end_time":p.end_time,"message":f"Booked {p.room_code}. ID: {rid}"}

def send_calendar_invite(params, db, call_index):
    try: p = SendCalendarInviteParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, SendCalendarInviteParams)
    if not _ok(db,"employee_auth"): return _ar()
    cal_id = f"CAL-{p.request_id[-6:]}"
    return {"status":"success","calendar_event_id":cal_id,"room_code":p.room_code,"date":p.date,"message":f"Calendar invite sent. Event: {cal_id}"}

# ═══════════════════════════════════════════════════════════════════════════════
# FLOWS 15-18: ACCOUNTS & ACCESS
# ═══════════════════════════════════════════════════════════════════════════════

def lookup_new_hire(params, db, call_index):
    try: p = LookupNewHireParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, LookupNewHireParams)
    if not _ok(db,"manager_auth"): return _ar("manager_auth")
    emp = db.get("employees",{}).get(p.new_hire_employee_id)
    if not emp: return _enf(p.new_hire_employee_id)
    if emp.get("employment_status") != "pending_start":
        return {"status":"error","error_type":"not_new_hire","message":f"Status '{emp['employment_status']}', not 'pending_start'"}
    return {"status":"success","employee":{"employee_id":p.new_hire_employee_id,"first_name":emp["first_name"],"last_name":emp["last_name"],"department_code":emp["department_code"],"role_code":emp["role_code"],"start_date":emp.get("start_date")}}

def check_existing_accounts(params, db, call_index):
    try: p = CheckExistingAccountsParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, CheckExistingAccountsParams)
    if not _ok(db,"manager_auth"): return _ar("manager_auth")
    emp = db.get("employees",{}).get(p.employee_id)
    if not emp: return _enf(p.employee_id)
    active = [a for a in emp.get("system_accounts",[]) if a.get("status")=="active"]
    if active: return {"status":"error","error_type":"accounts_already_exist","message":f"{len(active)} active account(s)"}
    return {"status":"success","employee_id":p.employee_id,"existing_accounts":[],"message":"Ready to provision"}

def provision_new_account(params, db, call_index):
    try: p = ProvisionNewAccountParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, ProvisionNewAccountParams)
    if not _ok(db,"manager_auth"): return _ar("manager_auth")
    if not _ok(db,"otp_auth"): return _ar("otp_auth")
    mgr = db.get("employees",{}).get(p.manager_employee_id)
    if not mgr: return _enf(p.manager_employee_id)
    if p.new_hire_employee_id not in mgr.get("direct_reports",[]):
        return {"status":"error","error_type":"not_authorized","message":"Not in direct reports"}
    nh = db.get("employees",{}).get(p.new_hire_employee_id)
    if not nh: return _enf(p.new_hire_employee_id)
    cid = _make_case_id("ACCT",p.new_hire_employee_id)
    nh["system_accounts"] = [{"case_id":cid,"status":"active","provisioned_date":db.get("_current_date",""),"access_groups":list(p.access_groups)}]
    nh["employment_status"] = "active"
    email = f"{nh['first_name'].lower()}.{nh['last_name'].lower()}@company.com"
    return {"status":"success","case_id":cid,"new_hire_employee_id":p.new_hire_employee_id,"email":email,"access_groups":list(p.access_groups),"message":f"Provisioned. Case: {cid}"}

def get_group_memberships(params, db, call_index):
    try: p = GetGroupMembershipsParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, GetGroupMembershipsParams)
    if not _ok(db,"otp_auth"): return _ar("otp_auth")
    emp = db.get("employees",{}).get(p.employee_id)
    if not emp: return _enf(p.employee_id)
    return {"status":"success","employee_id":p.employee_id,"memberships":copy.deepcopy(emp.get("group_memberships",[]))}

def get_group_details(params, db, call_index):
    try: p = GetGroupDetailsParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, GetGroupDetailsParams)
    if not _ok(db,"otp_auth"): return _ar("otp_auth")
    group = db.get("access_groups",{}).get(p.group_code)
    if not group: return {"status":"error","error_type":"not_found","message":f"Group {p.group_code} not found"}
    return {"status":"success","group":copy.deepcopy(group)}

def submit_group_membership_change(params, db, call_index):
    try: p = SubmitGroupMembershipChangeParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, SubmitGroupMembershipChangeParams)
    if not _ok(db,"otp_auth"): return _ar("otp_auth")
    emp = db.get("employees",{}).get(p.employee_id)
    if not emp: return _enf(p.employee_id)
    group = db.get("access_groups",{}).get(p.group_code)
    if not group: return {"status":"error","error_type":"not_found","message":f"Group {p.group_code} not found"}
    memberships = emp.get("group_memberships",[])
    codes = [m["group_code"] for m in memberships]
    cid = _make_case_id("ACCT",p.employee_id)
    if p.action == "add":
        if p.group_code in codes: return {"status":"error","error_type":"already_member","message":f"Already in {p.group_code}"}
        appr = group.get("requires_approval",False)
        if not appr: memberships.append({"group_code":p.group_code,"group_name":group["name"],"status":"active","added_date":db.get("_current_date","")})
        return {"status":"success","case_id":cid,"action":"add","group_code":p.group_code,"requires_approval":appr,"message":f"{'Pending approval' if appr else 'Added'}. Case: {cid}"}
    else:
        if p.group_code not in codes: return {"status":"error","error_type":"not_member","message":f"Not in {p.group_code}"}
        emp["group_memberships"] = [m for m in memberships if m["group_code"]!=p.group_code]
        return {"status":"success","case_id":cid,"action":"remove","group_code":p.group_code,"message":f"Removed. Case: {cid}"}

def get_permission_templates(params, db, call_index):
    try: p = GetPermissionTemplatesParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, GetPermissionTemplatesParams)
    if not _ok(db,"otp_auth"): return _ar("otp_auth")
    tmpls = {t:copy.deepcopy(v) for t,v in db.get("permission_templates",{}).items() if v.get("role_code")==p.role_code}
    if not tmpls: return {"status":"error","error_type":"not_found","message":f"No templates for {p.role_code}"}
    return {"status":"success","role_code":p.role_code,"templates":tmpls}

def submit_permission_change(params, db, call_index):
    try: p = SubmitPermissionChangeParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, SubmitPermissionChangeParams)
    if not _ok(db,"otp_auth"): return _ar("otp_auth")
    tmpl = db.get("permission_templates",{}).get(p.permission_template_id)
    if not tmpl: return {"status":"error","error_type":"not_found","message":f"Template {p.permission_template_id} not found"}
    if tmpl["role_code"] != p.new_role_code:
        return {"status":"error","error_type":"template_role_mismatch","message":f"Template for '{tmpl['role_code']}', not '{p.new_role_code}'"}
    cid = _make_case_id("ACCT",p.employee_id)
    emp = db.get("employees",{}).get(p.employee_id)
    if emp: emp["pending_role_change"] = {"case_id":cid,"new_role_code":p.new_role_code,"permission_template_id":p.permission_template_id,"effective_date":p.effective_date,"status":"pending"}
    return {"status":"success","case_id":cid,"employee_id":p.employee_id,"effective_date":p.effective_date,"message":f"Permission change submitted. Case: {cid}"}

def schedule_access_review(params, db, call_index):
    try: p = ScheduleAccessReviewParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, ScheduleAccessReviewParams)
    if not _ok(db,"otp_auth"): return _ar("otp_auth")
    rvw = f"ARVW-{p.case_id[-6:]}"
    emp = db.get("employees",{}).get(p.employee_id)
    if emp and emp.get("pending_role_change"): emp["pending_role_change"].update({"access_review_id":rvw,"review_date":p.review_date})
    return {"status":"success","review_id":rvw,"review_date":p.review_date,"message":f"90-day review on {p.review_date}. ID: {rvw}"}

def get_offboarding_record(params, db, call_index):
    try: p = GetOffboardingRecordParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, GetOffboardingRecordParams)
    if not _ok(db,"manager_auth"): return _ar("manager_auth")
    emp = db.get("employees",{}).get(p.employee_id)
    if not emp: return _enf(p.employee_id)
    ob = emp.get("offboarding_record")
    if not ob: return {"status":"error","error_type":"no_offboarding_record","message":"No off-boarding record. HR must initiate."}
    return {"status":"success","employee_id":p.employee_id,"offboarding":copy.deepcopy(ob)}

def submit_access_removal(params, db, call_index):
    try: p = SubmitAccessRemovalParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, SubmitAccessRemovalParams)
    if not _ok(db,"manager_auth"): return _ar("manager_auth")
    if not _ok(db,"otp_auth"): return _ar("otp_auth")
    mgr = db.get("employees",{}).get(p.manager_employee_id)
    if not mgr: return _enf(p.manager_employee_id)
    if p.departing_employee_id not in mgr.get("direct_reports",[]):
        return {"status":"error","error_type":"not_authorized","message":"Not in direct reports"}
    dep = db.get("employees",{}).get(p.departing_employee_id)
    if not dep: return _enf(p.departing_employee_id)
    ob = dep.get("offboarding_record")
    if not ob: return {"status":"error","error_type":"no_offboarding_record","message":"No off-boarding record"}
    cid = _make_case_id("ACCT",p.departing_employee_id)
    dep.update({"employment_status":"terminated","system_accounts":[],"group_memberships":[]})
    if p.removal_scope=="staged": dep["email_preserved_until"]=(_dt.strptime(p.last_working_day,"%Y-%m-%d")+timedelta(days=30)).strftime("%Y-%m-%d")
    ob.update({"access_removed":True,"removal_case_id":cid,"removal_scope":p.removal_scope})
    return {"status":"success","case_id":cid,"departing_employee_id":p.departing_employee_id,"removal_scope":p.removal_scope,"message":f"Access removed. Case: {cid}"}

def initiate_asset_recovery(params, db, call_index):
    try: p = InitiateAssetRecoveryParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, InitiateAssetRecoveryParams)
    if not _ok(db,"manager_auth"): return _ar("manager_auth")
    assets = [{"asset_tag":a["asset_tag"],"asset_type":a["asset_type"],"model":a["model"]} for a in db.get("assets",{}).values() if a.get("assigned_employee_id")==p.departing_employee_id]
    recv = f"RECV-{p.case_id[-6:]}"
    return {"status":"success","recovery_id":recv,"recovery_method":p.recovery_method,"assets_to_recover":assets,"message":f"Recovery {recv}: {len(assets)} device(s) via {p.recovery_method}"}

def transfer_to_agent(params, db, call_index):
    try: p = TransferToAgentParams.model_validate(params)
    except ValidationError as e: return validation_error_response(e, TransferToAgentParams)
    return {"status":"success","transfer_id":f"TRF-{p.employee_id}-{str(call_index).zfill(3)}","employee_id":p.employee_id,"transfer_reason":p.transfer_reason,"estimated_wait":"2-3 minutes","message":"Transferring"}