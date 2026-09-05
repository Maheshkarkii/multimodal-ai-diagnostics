"""
Generate realistic technical engineering manuals and SOPs in PDF and Markdown formats.
Creates standard reference documents for pump motors, gearboxes, bearings, and hydraulic systems.
"""

from pathlib import Path

import pypdf


def generate_pdf_manual(pdf_path: Path, title: str, pages_content: list) -> None:
    """Generate a clean searchable text PDF manual using pypdf writer."""
    writer = pypdf.PdfWriter()

    for _page_num, _text_content in enumerate(pages_content, start=1):
        # Create blank canvas page in points (letter size: 612 x 792)
        writer.add_blank_page(width=612, height=792)
        # Note: Since pypdf is primarily a reader/manipulator without built-in text rasterizer canvas,
        # we attach extracted text streams / metadata or create pure formatted documents.
        # Alternatively, we create standard text/markdown files and generate PDFs with proper streams.

    # To guarantee standard robust PDF text parsing tests, we use reportlab or raw PDF generation syntax:
    pass


def create_raw_pdf_with_text(pdf_path: Path, title: str, pages: list) -> None:
    """
    Generate valid searchable multi-page PDF files using basic PDF 1.4 stream syntax
    without external rendering dependencies.
    """
    page_obj_ids = []

    # Object 1: Catalog
    # Object 2: Outlines
    # Object 3: Pages
    # For each page: Page object + Content stream object + Font object

    font_obj_id = 4
    font_stream = f"{font_obj_id} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"

    content_objects = []
    page_objects = []

    current_obj_id = 5

    for _page_idx, (page_title, text_body) in enumerate(pages, start=1):
        # Format text into PDF content stream
        stream_lines = [
            "BT",
            "/F1 14 Tf",
            "50 740 Td",
            f"({page_title}) Tj",
            "/F1 10 Tf",
            "0 -25 Td",
        ]

        for line in text_body.split("\n"):
            # Escape parentheses
            clean_l = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream_lines.append(f"({clean_l}) Tj")
            stream_lines.append("0 -14 Td")

        stream_lines.append("ET")
        stream_data = "\n".join(stream_lines).encode("latin1", "replace")

        content_id = current_obj_id
        page_id = current_obj_id + 1
        page_obj_ids.append(page_id)
        current_obj_id += 2

        c_obj = f"{content_id} 0 obj\n<< /Length {len(stream_data)} >>\nstream\n{stream_data.decode('latin1')}\nendstream\nendobj\n"
        p_obj = (
            f"{page_id} 0 obj\n"
            f"<< /Type /Page /Parent 3 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_obj_id} 0 R >> >> >>\nendobj\n"
        )

        content_objects.append(c_obj)
        page_objects.append(p_obj)

    # Assemble main structure
    kids_str = " ".join([f"{pid} 0 R" for pid in page_obj_ids])
    pages_root = f"3 0 obj\n<< /Type /Pages /Kids [{kids_str}] /Count {len(pages)} >>\nendobj\n"
    catalog = "1 0 obj\n<< /Type /Catalog /Pages 3 0 R >>\nendobj\n"
    outlines = "2 0 obj\n<< /Type /Outlines /Count 0 >>\nendobj\n"

    all_objs = [catalog, outlines, pages_root, font_stream] + content_objects + page_objects

    # Calculate xref
    body = "%PDF-1.4\n"
    offsets = [0]
    for obj in all_objs:
        offsets.append(len(body.encode("latin1")))
        body += obj

    xref_offset = len(body.encode("latin1"))
    body += f"xref\n0 {len(offsets)}\n0000000000 65535 f \n"
    for off in offsets[1:]:
        body += f"{off:010d} 00000 n \n"

    body += f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(body.encode("latin1"))


def generate_all_sample_manuals(output_dir: Path) -> None:
    """Generate complete suite of industrial technical documentation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Industrial Motor Maintenance Manual (PDF)
    motor_pdf_pages = [
        (
            "SECTION 1: GENERAL SPECIFICATIONS & OPERATING LIMITS",
            "1.1 Introduction\n"
            "This technical maintenance manual provides operating limits, inspection procedures, and\n"
            "corrective repair workflows for Model M-4500 Three-Phase Industrial Induction Motors.\n\n"
            "1.2 Vibration Thresholds Table\n"
            "ISO 10816-3 Industrial Vibration Severity Standard:\n"
            "Zone A (Normal): RMS vibration velocity below 2.3 mm/s (0.09 in/s). Smooth operation.\n"
            "Zone B (Acceptable): RMS velocity between 2.3 mm/s and 4.5 mm/s. Continuous operation allowed.\n"
            "Zone C (Warning): RMS velocity between 4.5 mm/s and 7.1 mm/s. Immediate maintenance required.\n"
            "Zone D (Critical Danger): RMS velocity exceeding 7.1 mm/s. Immediate emergency shutdown.",
        ),
        (
            "SECTION 2: BEARING INSPECTION & LUBRICATION PROCEDURES",
            "2.1 Bearing Degradation Symptoms\n"
            "Common symptoms of rolling element bearing wear include high-frequency acoustic squealing,\n"
            "inner race spalling harmonics at BPFI (Ball Pass Frequency Inner race), outer race defects at BPFO,\n"
            "and localized bearing housing temperature exceeding 75 degrees Celsius.\n\n"
            "2.2 Recommended Inspection Steps for Bearing Problems\n"
            "Step 1: Perform acoustic ultrasound listening check for periodic impacts or clicking.\n"
            "Step 2: Measure overall RMS vibration velocity and acceleration enveloping peak-to-peak.\n"
            "Step 3: Measure thermal infrared gradient across drive-end (DE) and non-drive-end (NDE) bearings.\n"
            "Step 4: Inspect grease sample for metallic particle discoloration or oxidation degradation.\n"
            "Step 5: Check shaft axial and radial play with dial indicator (< 0.05 mm allowable).",
        ),
        (
            "SECTION 3: ROTOR UNBALANCE & SHAFT MISALIGNMENT",
            "3.1 Rotor Dynamic Unbalance\n"
            "Rotor unbalance generates a dominant 1X running speed radial vibration frequency component.\n"
            "When radial vibration at 1X rotational speed exceeds 5.0 mm/s RMS, perform dynamic two-plane balancing.\n\n"
            "3.2 Angular and Parallel Misalignment\n"
            "Shaft misalignment typically exhibits strong 2X and 3X shaft speed harmonic vibration peaks\n"
            "accompanied by high axial vibration amplitudes. Check flexible coupling elastomeric inserts\n"
            "and verify laser alignment tolerances within 0.05 mm offset and 0.04 mm/100mm angularity.",
        ),
    ]
    create_raw_pdf_with_text(
        output_dir / "motor_m4500_maintenance_manual.pdf", "MOTOR M-4500 SERVICE MANUAL", motor_pdf_pages
    )

    # 2. Hydraulic Centrifugal Pump Guide (Markdown)
    pump_guide_content = """# CENTRIFUGAL PUMP CP-800 SERVICE & TROUBLESHOOTING GUIDE

--- PAGE 1 ---
## SECTION 1: HYDRAULIC CAVITATION PHENOMENON & DIAGNOSIS

1.1 Cavitation Symptoms
Hydraulic cavitation occurs when suction pressure drops below liquid vapor pressure, causing vapor bubbles to form and violently collapse against impeller vanes.
Key diagnostic indicators:
- Audible gravel-like popping or continuous rushing hiss in pump casing
- High-frequency broadband acoustic noise between 5 kHz and 15 kHz
- Erratic discharge pressure fluctuations (> 15% delta P)
- Severe erosion pitting on impeller inlet leading edges

1.2 Recommended Corrective Actions for Cavitation
1. Verify Net Positive Suction Head Available (NPSHa) exceeds NPSHr by at least 1.5 meters.
2. Inspect suction line strainers and intake foot valves for partial blockages.
3. Check suction pipe diameter to ensure flow velocity remains below 2.0 m/s.
4. Ensure suction isolation valve is 100% fully open and not throttled.

--- PAGE 2 ---
## SECTION 2: MECHANICAL SEAL LEAKAGE & INSPECTION

2.1 Mechanical Seal Troubleshooting
Primary causes of mechanical seal failure include dry running, abrasive particle accumulation, face distortion from thermal shock, and excessive shaft deflection.
Allowable static leakage rate: < 5 drops per minute during standard operation.
If leakage exceeds 10 ml/hour, isolate pump, depressurize casing, and replace primary carbon/silicon-carbide seal rings.

2.2 Impeller Wear Tolerances
Standard impeller clearance: 0.30 mm to 0.45 mm.
Maximum allowable operational wear clearance before rebuild: 0.80 mm.
"""
    with open(output_dir / "centrifugal_pump_cp800_troubleshooting.md", "w", encoding="utf-8") as f:
        f.write(pump_guide_content)

    # 3. Industrial Gearbox Repair Manual (Text)
    gearbox_guide_content = """INDUSTRIAL GEARBOX GB-200 TECHNICAL REPAIR GUIDE

--- PAGE 1 ---
SECTION 1: GEAR MESHING FAULTS AND ACOUSTIC NOISE

1.1 Gear Tooth Wear and Pitting
Gear tooth flank pitting generates gear mesh frequency (GMF = Number of Teeth x Shaft RPM) sideband modulation.
Acoustic signatures characterized by harmonic modulation spaced at the rotating shaft speed indicate chipped, cracked, or worn gear teeth.
Recommended Oil Analysis:
- Iron (Fe) wear debris > 100 ppm requires lubricant flush and filter replacement.
- Copper/Bronze (Cu) > 50 ppm indicates thrust washer or bronze cage degradation.

--- PAGE 2 ---
SECTION 2: BACKLASH AND BEARING PRELOAD SPECIFICATIONS

2.1 Backlash Tolerances
Normal backlash for GB-200 helical gear sets: 0.15 mm to 0.25 mm.
Excessive backlash (> 0.45 mm) produces rattling impact sounds during speed or torque transitions.

2.2 Thermal Operating Limits
Maximum continuous gearbox sump oil temperature: 85 degrees Celsius.
Thermal warning threshold: 80 degrees Celsius.
"""
    with open(output_dir / "industrial_gearbox_gb200_repair.txt", "w", encoding="utf-8") as f:
        f.write(gearbox_guide_content)

    # 4. Scanned / Degraded Document Example (For negative and edge-case testing)
    scanned_pdf_pages = [
        ("SCANNED COVER PAGE", ""),  # Blank text page to test scanned/empty page detection
        ("PAGE 2", "Sparse warranty notice."),
    ]
    create_raw_pdf_with_text(output_dir / "sparse_unreadable_sample.pdf", "SPARSE SAMPLE", scanned_pdf_pages)

    print(f"Generated sample technical manuals in '{output_dir}'.")


if __name__ == "__main__":
    generate_all_sample_manuals(Path("data/rag/documents"))
