import unittest

from prisma.enums import OwnerType
from pydash import group_by
from constants.company import COMPANY_MAP

from data.etl.entity.owner.owner import generate_clean_owner_map, OwnerTypeParser


class TestPatentScriptUtils(unittest.TestCase):
    def test_owner_type_parser(self):
        test_conditions = [
            {
                "name": "mayo clinic",
                "expected_output": OwnerType.HEALTH_SYSTEM,
            },
            {
                "name": "albert einstein college medicine",
                "expected_output": OwnerType.UNIVERSITY,
            },
            {
                "name": "wisconsin university",
                "expected_output": OwnerType.UNIVERSITY,
            },
            {
                "name": "pfizer",
                "expected_output": OwnerType.INDUSTRY_LARGE,
            },
            {
                "name": "random co, inc",
                "expected_output": OwnerType.INDUSTRY,
            },
            {
                "name": "us government",
                "expected_output": OwnerType.GOVERNMENTAL,
            },
            {
                "name": "random llc",
                "expected_output": OwnerType.INDUSTRY,
            },
            {
                "name": "bristol university research foundation",
                "expected_output": OwnerType.UNIVERSITY,
            },
        ]

        for test in test_conditions:
            name = test["name"]
            expected_output = test["expected_output"]

            result = OwnerTypeParser.find(name)
            print("Actual", result, "expected", expected_output)
            self.assertEqual(result, expected_output)

    def test_clean_assignees(self):
        test_conditions = [
            {
                "terms": [
                    "Pfizer Inc",
                    "Pfizer Inc",
                    "Bobs Pharmacy LLC",
                    "Bobs Pharmacy LLC",
                    "Bobs Pharmacy LLC inc CO",
                    "BBOB Labs",
                    "BioGen Ma",
                    "BioGen Mass",
                    "Biogen Inc",
                    "Biogen Inc",
                    "Charles River Laboratories Inc",
                    "Charles River Lab",
                    "Charles River Laboratories",
                    "Charles River Lab",
                    "Charles River Inc",
                    "PFIZER R&D UK LTD",
                    "Astrazeneca",
                    "Astrazeneca",
                    "Astrazeneca",
                    "Astrazeneca",
                    "Astrazeneca China",
                    "ASTRAZENECA (CHINA) CO LTD",
                    "ASTRAZENECA COLLABORATION VENTURES LLC",
                    "JAPAN PHARMA CO LTD",
                    "Matsushita Electric Ind Co Ltd",
                    "Matsushita Electric Ind Co Ltd",
                    "MATSUSHITA ENVIRONMENTAL & AIR",
                    "MATSUSHITA ELECTRIC IND CO LTD",
                    "US GOV CT DISEASE CONTR & PREV",
                    "US GOV NAT INST HEALTH",
                    "THE US GOV",
                    "siemens ag",
                    "siemens ag",
                    "siemens ag gmbh",
                    "siemens",
                    "Korea Advanced Institute Science And Technology",
                    "university of colorado",
                    "university of colorado",
                    "university of colorado at denver",
                    "university of colorado, denver",
                    "Dicerna Pharmaceuticals, Inc., a Novo Nordisk company",
                    "NOVO NORDISK",
                    "NOVO NORDISK",
                    "NOVO NORDISK",
                    "NOVO NORDISK BIOTECH INC",
                    "NOVO NORDISK HEALTHCARE AG",
                    "Genentech, Inc.",
                    "Genentech Inc.",
                    "Genentech Inc.",
                    "Genentech Inc.",
                    "Genentech Inc",
                    "GENENTECH INC",
                    "agency for innovation by science and technology",
                    "agency for innovation by science and technology",
                    "NAT SCIENCE AND TECHNOLOGY DEV AGENCY",
                    "AGENCY IND SCIENCE TECHN",
                    "C O AGENCY FOR SCIENCE TECHNOLOGY AND RES",
                    "U S ENVIRONMENTAL AGENCY",
                    "US Government",
                    "US Government",
                    "US Government",
                    "US ENVIRONMENTAL AGENCY",
                    "united states environmental agency",
                    "United States Environmental Protection Agency",
                    "US ENVIRONMENTAL PROTECTION AGENCY",
                    "janssen pharmaceuticals inc",
                    "JANSSEN BIOPHARMA INC",
                    "JANSSEN SCIENCES IRELAND UC",
                    "JANSSEN VACCINE & PREVENTION B V",
                    "JANSSEN R & D IRELAND",
                    "JANSSEN",
                    "JANSSEN",
                    "JANSSEN",
                    "JANSSEN PHARMACEUTICA N V",
                    "JANSSEN RES & DEVELOPMENT LLC",
                    "CA NAT RESEARCH COUNCIL",
                    "JANSSEN PHARMACEUTICA NV",
                    "SOUTH AFRICA NUCLEAR ENERGY COMPANY",
                    "ORTHO MCNEIL PHARM COMPANY",
                    "PROCTER & GAMBLE",
                    "PROCTER & GAMBLE",
                    "PROCTER & GAMBLE, Inc. CA",
                    "ABBOTT GMBH & CO KG",
                    "ABBOTT GMBH & KG CO",
                    "ABBOTT",
                    "ABBOTT",
                    "ABBOTT",
                    "ABBOTT",
                    "ABBOTT CO",
                    "ABBOTT CO",
                    "ABBOTT GMBH CO KG",
                    "ABBOTT LAB SHANGHAI COMPANY LTD",
                    "ABBOTT DIAGNOSTICS INT COMPANY",
                    "ABBOTT DIAGNOSTICS INT COMPANY",
                    "JOHNSON & JOHNSON INTERNAT",
                    "US GOV HEALTH & HUMAN SERV",
                    "GOVERNMENT OF UNITED STATES OF",
                    "GOVERNMENT OF THE U S A AS REPRESENTED BY THE SECRE OF THE DEPART OF HEALTH & HUMAN SERVICES",
                    "GOVERNING COUNCIL OF THE UNIV OF TORONTO",
                    "UNIVERSITY OF TORONTO",
                    "UNIVERSITY OF TORONTO",
                    "THE GOVERNING COUNCIL OF THE UNIVERSITY OF TORONTO",
                    "UNIV TORONTO MISSISSAUGA",
                    "GOVERNING COUNCIL UNIV TORONTO",
                    "DAINIPPON INK & CHEMICALS",
                    "DAINIPPON GMBH",
                    "DAINIPPON",
                    "DAINIPPON COMPANY",
                    "DAINIPPON PHARMACEUTICAL CO",
                    "DAINIPPON PHARMACEUTICAL CO",
                    "DAINIPPON PHARMACEUTICAL CO",
                    "MERCK & CO INC",
                    "MERCK CO",
                    "MERCK CO",
                    "MERCK SHARP & DOHME ANIMAL HEALTH S L",
                    "MERCK BIOSCIENCES AG GMBH",
                    "JAPAN SCIENCE & TECH CORP",
                    "MO R & D INC",
                    "MO R&D INC",
                    "MO research and development INC",
                    "QUEEN MARY and WESTFIELD COLLEGE",
                    "QUEEN MARY and WESTFIELD COLLEGE",
                    "QUEEN MARY & WESTFIELD COLLEGE",
                    "THE PROCTER & GAMBLE COMPANY",
                    "US GOV SEC TECH TRANSFER",
                ],
                "expected": {
                    "abbott": 6,
                    "agency for innovation by science and technology": 3,
                    "astrazeneca": 3,
                    "biogen inc": 2,
                    "bobs pharmaceutical llc": 2,
                    "charles river laboratory": 4,
                    "dainippon pharmaceutical co": 4,
                    "genentech inc": 4,
                    "governing council of the university of toronto": 4,
                    "janssen": 6,
                    "janssen research and development ireland": 2,
                    "japan pharmaceutical co ltd": 2,
                    "merck co": 2,
                    "novo nordisk": 3,
                    "other": 28,
                    "procter & gamble": 3,
                    "queen mary and westfield college": 2,
                    "siemens": 3,
                    "united states environmental agency": 5,
                    "united states government": 5,
                    "university of colorado at denver": 2,
                },
                "expected_overrides": {
                    "abbott": 7,
                    "agency for innovation by science and technology": 3,
                    "astrazeneca": 4,
                    "biogen": 3,
                    "bobs pharmaceutical llc": 2,
                    "charles river laboratory": 4,
                    "dainippon": 5,
                    "genentech": 4,
                    "janssen": 9,
                    "japan science & technology corp": 2,
                    "johnson & johnson": 1,
                    "matsushita": 3,
                    "merck": 4,
                    "novo nordisk": 4,
                    "other": 12,
                    "pfizer": 2,
                    "procter and gamble": 3,
                    "queen mary and westfield college": 2,
                    "siemens": 3,
                    "united states environmental agency": 5,
                    "united states government": 1,
                    "university of colorado at denver": 2,
                    "university of toronto": 4,
                    "us government": 6,
                },
            },
        ]

        for condition in test_conditions:
            grouped = group_by(condition["terms"], lambda x: x)
            counts = [len(v) for v in grouped.values()]
            terms = list(grouped.keys())

            for override, expected in zip(
                [{}, COMPANY_MAP],
                [condition["expected"], condition["expected_overrides"]],
            ):
                owner_map = generate_clean_owner_map(
                    terms, counts, overrides=override, solo_cluster_threshold=4
                )
                result = sorted([owner_map.get(term) or term for term in terms])
                stats = {term: result.count(term) for term in result}
                print("Actual: \n", stats)
                print("Expected: \n", expected)

                discrepancy = {
                    term: abs(stats.get(term, 0) - expected.get(term, 0))
                    for term in expected.keys()
                }
                total_discrepancy = sum(discrepancy.values())

                print("Discrepancies: \n", discrepancy)
                print("Total discrepancy:", total_discrepancy)
                self.assertLessEqual(total_discrepancy, 9)


if __name__ == "__main__":
    unittest.main()
