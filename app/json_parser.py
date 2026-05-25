import re
from typing import List

from app.models import DonutParsed, DonutHeader, DonutItem, DonutSummary


def _to_float_or_none(value):
    try:
        cleaned = (
            str(value)
            .replace(" ", "")
            .replace("\u00A0", "")  # twarda spacja
            .replace(",", ".")
            .replace("$", "")
            .replace("€", "")
            .replace("PLN", "")
        )
        return float(cleaned)
    except (ValueError, TypeError, AttributeError):
        return None


def parse_donut_xml(text: str) -> DonutParsed:
    # ---------------- HEADER ----------------
    header_tags = [
        "invoice_no", "invoice_date", "seller", "client",
        "seller_tax_id", "client_tax_id", "iban"
    ]
    header_data = {}
    for tag in header_tags:
        m = re.search(fr"<s_{tag}>(.*?)</s_{tag}>", text)
        if m:
            header_data[tag] = m.group(1).strip()

    header = DonutHeader(**header_data)

    # ---------------- ITEMS ----------------
    items: List[DonutItem] = []

    items_block_match = re.search(r"<s_items>(.*?)</s_items>", text)
    if items_block_match:
        items_block = items_block_match.group(1)
        raw_items = items_block.split("<sep/>")

        for block in raw_items:
            block = block.strip()
            if not block:
                continue

            fields = [
                "item_desc", "item_qty", "item_net_price",
                "item_net_worth", "item_vat", "item_gross_worth"
            ]
            item_data = {}
            for f in fields:
                m = re.search(fr"<s_{f}>(.*?)</s_{f}>", block)
                if m:
                    val = m.group(1).strip()
                    if f == "item_desc":
                        item_data[f] = val
                    else:
                        item_data[f] = _to_float_or_none(val)

            items.append(DonutItem(**item_data))

    # ---------------- SUMMARY ----------------
    summary_tags = ["total_net_worth", "total_vat", "total_gross_worth"]
    summary_data = {"total_items": len(items)}
    for tag in summary_tags:
        m = re.search(fr"<s_{tag}>(.*?)</s_{tag}>", text)
        if m:
            summary_data[tag] = _to_float_or_none(m.group(1).strip())  # type: ignore

    summary = DonutSummary(**summary_data)

    return DonutParsed(header=header, items=items, summary=summary)
