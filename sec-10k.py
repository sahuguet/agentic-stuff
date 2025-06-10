from lxml import etree
import json
import sys

"""
The XML file contains all the facts for the 10K document.
The same fact may appear more than once, with a different `id`.
Each fact is linked to a context, which is usually a timestamp.
We retrieve the facts we care about, and we resolve the context to a human-readable date.
We output a JSON array of objects, one per fact.
"""


def resolve_context(root, ctx, ns):
    instant = root.xpath(f"//d:context[@id='{ctx}']//d:instant/text()", namespaces={'d': ns[None]})
    if len(instant) > 0:
        return instant[0]
    else:
        startDate = root.xpath(f"//d:context[@id='{ctx}']//d:startDate/text()", namespaces={'d': ns[None]})[0]
        endDate   = root.xpath(f"//d:context[@id='{ctx}']//d:endDate/text()", namespaces={'d': ns[None]})[0]
        return f"from {startDate} to {endDate}"

def process_xml_file(xml_file):
    root = etree.parse(xml_file)
    ns = root.getroot().nsmap

    name = root.xpath("//dei:EntityRegistrantName/text()", namespaces={'dei': ns['dei']})[0]
    cik = root.xpath("//dei:EntityCentralIndexKey/text()", namespaces={'dei': ns['dei']})[0]

    data = {}

    FACTS_TO_EXTRACT = [ "PropertyPlantAndEquipmentNet", "AssetsCurrent", "DepreciationDepletionAndAmortization" ]

    facts = root.xpath("|".join([f"//us-gaap:{fact}" for fact in FACTS_TO_EXTRACT]), namespaces={"us-gaap": ns["us-gaap"]})
    for fact in facts:
        property = fact.tag.split("}")[-1]
        context = fact.get("contextRef", "(no contextRef)")
        key = property + "@" + context
        if key in data:
            continue
        unit    = fact.get("unitRef",    "(no unitRef)")
        decimals = fact.get("decimals",  "(no decimals)")
        id = fact.get("id", "(no id)")
        value   = (fact.text or "").strip()
        data[key] = { "name": name, "cik": cik, "property": property, "date": resolve_context(root, context, ns), "value": value, "unit": unit, "decimals": decimals }

    return list(data.values())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sec-10k.py <xml_file1> [xml_file2 ...]")
        sys.exit(1)
    
    data = []
    for xml_file in sys.argv[1:]:
        try:
            data.extend(process_xml_file(xml_file))
        except Exception as e:
            print(f"Error processing {xml_file}: {str(e)}", file=sys.stderr)
            continue
    
    print(json.dumps(data, indent=4))





