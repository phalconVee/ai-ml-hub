"""Generate sample_docs/sample_lease.pdf: a short, invented residential lease.

Run:  uv run python sample_docs/make_sample.py
The parties, address and terms are made up. The tests call `build_sample_pdf()`
to get the same PDF as bytes. fpdf2 is a dev dependency only.
"""

from __future__ import annotations

from pathlib import Path

# (heading, body) pairs. "\f" in the list forces a page break, so the sample
# also exercises clauses on page 2 and a clause continuing across pages.
CLAUSES: list = [
    ("RESIDENTIAL LEASE AGREEMENT",
     "This Lease is made between Harbor Lane Properties LLC (\"Landlord\") and "
     "Jordan Avery (\"Tenant\") for the premises at 14 Alder Court, Unit 3 (the \"Premises\")."),
    ("1. Term", "The Lease begins on 1 March 2026 and ends on 28 February 2027 (the \"Term\")."),
    ("2. Rent", "Tenant shall pay rent of USD 2,150 per month, due on the first day of each "
     "month, by bank transfer to the account Landlord designates in writing."),
    ("3. Late Payment", "If rent is not received within five (5) days after it is due, Tenant "
     "shall pay a late fee of ten percent (10%) of the overdue amount. Late fees are not a waiver "
     "of Landlord's other remedies."),
    ("4. Security Deposit", "Tenant shall pay a security deposit of USD 4,300 before moving in. "
     "Landlord shall return the deposit, less lawful deductions for unpaid rent and damage beyond "
     "normal wear and tear, within thirty (30) days after Tenant vacates."),
    "\f",
    ("5. Early Termination", "Tenant may terminate this Lease before the end of the Term by giving "
     "Landlord at least sixty (60) days' written notice and paying an early termination fee equal to "
     "two (2) months' rent. Landlord may not terminate early except under Section 8."),
    ("6. Maintenance and Repairs", "Landlord shall keep the structure, roof, plumbing, heating and "
     "electrical systems in good repair. Tenant shall promptly notify Landlord in writing of any needed "
     "repair. Tenant is responsible for damage caused by Tenant or Tenant's guests."),
    ("7. Entry by Landlord", "Landlord may enter the Premises to inspect or make repairs after giving "
     "Tenant at least twenty-four (24) hours' notice, except in an emergency."),
    ("8. Default", "If Tenant fails to pay rent within fifteen (15) days after written notice of "
     "non-payment, Landlord may terminate this Lease as permitted by applicable law."),
    ("9. Pets", "No pets are allowed without Landlord's prior written consent, which Landlord may "
     "withhold at its discretion."),
    ("10. Governing Law", "This Lease is governed by the laws of the State of California."),
]


def build_sample_pdf() -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    for item in CLAUSES:
        if item == "\f":
            pdf.add_page()
            continue
        heading, body = item
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 7, heading, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, body, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)
    return bytes(pdf.output())


if __name__ == "__main__":
    out = Path(__file__).with_name("sample_lease.pdf")
    out.write_bytes(build_sample_pdf())
    print(f"wrote {out}")
