"""生成合成工单库（self-built，非上游 4.2K 工单）。

本脚本只读本地 `data/normalized/{assets,sensors}.csv` 的目录字段（site / asset_id /
asset_class / sensor 名），按确定性模板派生出 26 条工单，写到
`data/wo/synthetic_workorders.json`。**不生成任何上游工单记录**：本机没有工单数据源，
这里的工单只用于把 WO 工具面（检索 / 统计 / 归类 / SLA / 停机 / 备件 / 故障模式映射）
接到本地目录上，数值与字段结构均为自建。

用法（在 /data/repro 下）：
    .venv/bin/python data/wo/gen_workorders.py
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NORM = ROOT / "data" / "normalized"
OUT = ROOT / "data" / "wo" / "synthetic_workorders.json"

#: 每条模板：(type, status, priority, defect_category, symptom, sensor_key,
#:             parts, downtime_hours, failure_mode, description)
TEMPLATES = [
    (
        "repair",
        "closed",
        "high",
        "thermal",
        "compressor discharge temperature above setpoint",
        "Supply Temperature",
        ["compressor suction valve", "thermistor probe"],
        12.5,
        "Compressor Overheating",
        "Compressor ran hot after high load; inspected thermistor and suction line.",
    ),
    (
        "inspection",
        "resolved",
        "medium",
        "mechanical",
        "motor current unbalanced across phases",
        "Motor_Current_A",
        ["motor terminal block", "insulation tester"],
        0.0,
        "Compressor Motor Failure",
        "Routine insulation and winding resistance check; no winding fault found.",
    ),
    (
        "preventive",
        "closed",
        "low",
        "hydraulic",
        "condenser water flow below design",
        "Condenser Water Flow",
        ["condenser tube brush", "flow meter calibration kit"],
        6.0,
        "Condenser Water Side Fouling",
        "Annual condenser tube cleaning and flow verification.",
    ),
    (
        "repair",
        "open",
        "critical",
        "thermal",
        "evaporator temperature drift on load",
        "Liquid Refrigerant Evaporator Temperature",
        ["evaporator pressure transducer", "refrigerant charge gauge"],
        None,
        "Evaporator Water Side Fouling",
        "Evaporator side shows slow thermal drift; awaiting spare transducer.",
    ),
    (
        "emergency",
        "in_progress",
        "critical",
        "seal",
        "low oil level alarm triggered",
        "Low_Oil_Level_Alarm",
        ["oil seal kit", "oil filter", "gasket set"],
        4.0,
        "Compressor Motor Failure",
        "Oil seal weeping; oil level alarm active, seal replacement in progress.",
    ),
    (
        "repair",
        "closed",
        "medium",
        "hydraulic",
        "filter pressure drop rising",
        "Cyclone_Filter_Drop_Pressure_bar",
        ["filter element", "drain valve seal"],
        3.0,
        "Liquid Line Restriction",
        "Filter element clogged; replaced element and flushed return line.",
    ),
    (
        "inspection",
        "open",
        "medium",
        "control",
        "run status signal intermittent",
        "Run Status",
        ["limit switch", "control relay"],
        None,
        "Chiller Trip",
        "Intermittent run-status contact; wiring harness under inspection.",
    ),
    (
        "preventive",
        "resolved",
        "low",
        "mechanical",
        "vibration pulse counter count low",
        "Airflow_Pulse_Counter",
        ["pulse sensor"],
        0.0,
        None,
        "Routine airflow pulse counter verification; count within tolerance.",
    ),
    (
        "repair",
        "closed",
        "high",
        "hydraulic",
        "reservoir pressure below threshold",
        "Reservoir_Pressure_bar",
        ["pressure relief valve", "reservoir breather"],
        5.5,
        "Evaporator Water Flow Reduction",
        "Reservoir pressure low after relief valve weep; valve replaced.",
    ),
    (
        "repair",
        "open",
        "high",
        "electrical",
        "motor power draw elevated",
        "P1_MotorPower_W_100Hz",
        ["power contactor", "current transformer"],
        None,
        "Compressor Motor Failure",
        "Motor power elevated under load; contactor contacts being checked.",
    ),
    (
        "inspection",
        "resolved",
        "low",
        "control",
        "pneumatic panel pressure stable",
        "Pneumatic_Panel_Pressure_bar",
        [],
        0.0,
        None,
        "Pneumatic panel pressure check; reading stable across shift.",
    ),
    (
        "repair",
        "closed",
        "medium",
        "hydraulic",
        "flow sensor FS1 reading low",
        "FS1_VolumeFlow_l_per_min_10Hz",
        ["flow sensor O-ring", "sensor cable"],
        2.5,
        "Evaporator Water Flow Reduction",
        "FS1 O-ring leak corrected; flow reading restored.",
    ),
    (
        "repair",
        "in_progress",
        "high",
        "mechanical",
        "filter drop pressure high on pump side",
        "Tower_Discharge_Switch_Status",
        ["discharge switch", "check valve"],
        1.5,
        "Condenser Water Flow Reduction",
        "Discharge switch chatter traced to check valve; replacement pending.",
    ),
    (
        "preventive",
        "closed",
        "low",
        "thermal",
        "oil temperature within range",
        "Oil_Temperature_C",
        ["oil sample bottle"],
        0.0,
        None,
        "Scheduled oil sample and temperature check; sample clean.",
    ),
    (
        "repair",
        "open",
        "medium",
        "control",
        "low pressure switch status open",
        "Low_Pressure_Switch_Status",
        ["pressure switch", "switch bracket"],
        None,
        "Refrigerant Undercharge",
        "Low-pressure switch open; suspected undercharge pending leak test.",
    ),
    (
        "inspection",
        "resolved",
        "low",
        "mechanical",
        "outlet valve status logged",
        "Compressor_Outlet_Valve_Status",
        ["valve stem packing"],
        0.0,
        None,
        "Outlet valve packing inspected; no leakage recorded.",
    ),
    (
        "repair",
        "closed",
        "high",
        "hydraulic",
        "pump discharge pressure oscillating",
        "PS1_Pressure_bar_100Hz",
        ["pressure gauge", "pump impeller seal"],
        4.0,
        "Liquid Line Restriction",
        "Pressure oscillation traced to restricted suction strainer; strainer cleaned.",
    ),
    (
        "repair",
        "in_progress",
        "medium",
        "thermal",
        "bearing temperature rising",
        "TS1_Temperature_C_1Hz",
        ["bearing temperature sensor"],
        1.0,
        None,
        "Bearing temperature trend under watch; sensor calibration in progress.",
    ),
    (
        "inspection",
        "open",
        "low",
        "control",
        "load start signal timing check",
        "Load_Start_Signal",
        ["start timer relay"],
        None,
        None,
        "Load-start signal timing verification scheduled.",
    ),
    (
        "repair",
        "closed",
        "high",
        "seal",
        "pump seal leak observed",
        "PS2_Pressure_bar_100Hz",
        ["mechanical seal", "gland packing"],
        3.5,
        None,
        "Pump mechanical seal replaced; no leak on post-run check.",
    ),
    (
        "preventive",
        "resolved",
        "low",
        "mechanical",
        "pump motor current nominal",
        "Motor_Current_A",
        [],
        0.0,
        None,
        "Scheduled motor current survey; current within nameplate.",
    ),
    (
        "repair",
        "open",
        "critical",
        "hydraulic",
        "filter drop pressure above alarm",
        "Cyclone_Filter_Drop_Pressure_bar",
        ["filter element", "pressure switch"],
        None,
        "Liquid Line Restriction",
        "Filter head pressure above alarm; element replacement blocked on spare availability.",
    ),
    (
        "repair",
        "closed",
        "medium",
        "thermal",
        "chiller efficiency dropped",
        "Chiller Efficiency",
        ["water treatment test kit"],
        7.0,
        "Condenser Water Side Fouling",
        "Efficiency drop linked to condenser fouling; tubes chemically cleaned.",
    ),
    (
        "inspection",
        "resolved",
        "low",
        "control",
        "setpoint vs return temperature review",
        "Setpoint Temperature",
        [],
        0.0,
        None,
        "Setpoint/return temperature alignment review; offsets acceptable.",
    ),
    (
        "repair",
        "in_progress",
        "high",
        "electrical",
        "power input spike on start",
        "Power Input",
        ["start capacitor", "contactor"],
        2.0,
        "Chiller Trip",
        "Start-up power spike and trip; capacitor and contactor under test.",
    ),
    (
        "preventive",
        "closed",
        "low",
        "hydraulic",
        "condenser flow annual check",
        "Condenser Water Flow",
        ["flow meter", "flange gasket"],
        1.0,
        "Condenser Water Flow Reduction",
        "Annual flow verification; meter recalibrated, flow within design.",
    ),
]


def _load_csv(name: str) -> list[dict[str, str]]:
    with (NORM / name).open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    assets = _load_csv("assets.csv")
    sensors = _load_csv("sensors.csv")
    sensor_by_asset: dict[str, list[str]] = {}
    for row in sensors:
        sensor_by_asset.setdefault(row["asset_id"], []).append(row["sensor"])

    base = datetime(2020, 6, 1, 8, 0, 0)
    workorders: list[dict[str, object]] = []
    for idx, tpl in enumerate(TEMPLATES):
        asset = assets[idx % len(assets)]
        aid = asset["asset_id"]
        sensor_name = tpl[5]
        if sensor_name not in sensor_by_asset.get(aid, []):
            # 派生兜底：取该资产目录里的第一个传感器，保证 related_sensor 真实存在
            sensor_name = sensor_by_asset[aid][0]
        reported = base + timedelta(hours=idx * 17)
        downtime = tpl[7]
        closed_at = None
        sla_due = reported + timedelta(hours=48)
        if tpl[1] in ("closed", "resolved"):
            closed_at = reported + timedelta(hours=(downtime or 0) + 6)
        workorders.append(
            {
                "wo_id": f"WO-2020-{idx + 1:03d}",
                "site": asset["site"],
                "asset_id": aid,
                "asset_class": asset["asset_class"],
                "type": tpl[0],
                "status": tpl[1],
                "priority": tpl[2],
                "defect_category": tpl[3],
                "symptom": tpl[4],
                "related_sensor": sensor_name,
                "parts": tpl[6],
                "reported_at": reported.isoformat(),
                "closed_at": closed_at.isoformat() if closed_at else None,
                "downtime_hours": downtime,
                "sla_due": sla_due.isoformat(),
                "failure_mode": tpl[8],
                "description": tpl[9],
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "count": len(workorders),
        "workorders": workorders,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[wo] wrote {len(workorders)} synthetic workorders -> {OUT}")


if __name__ == "__main__":
    main()
