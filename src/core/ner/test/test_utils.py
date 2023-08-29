import unittest

from core.ner.cleaning import EntityCleaner
from core.ner.utils import rearrange_terms, normalize_by_pos


class TestNerUtils(unittest.TestCase):
    """
    TODO:
    - poly(lactide-co-glycolide) (PLGA)
    """

    def setUp(self):
        self.cleaner = EntityCleaner()

    def test_filter_common_terms(self):
        test_conditions = [
            {
                "terms": [
                    "vaccine candidates",
                    "PF-06863135",
                    "therapeutic options",
                    "COVID-19 mRNA vaccine",
                    "exception term",
                    "common term",
                ],
                "exception_list": ["exception"],
                "expected_output": [
                    "pf06863135",
                    "covid 19 mrna vaccine",
                    "exception term",
                ],
            },
            {
                "terms": [
                    "vaccine candidate",
                    "vaccine candidates",
                    "therapeutic options",
                    "therapeutic option",
                    "therapeutics option",
                    "COVID-19 mRNA vaccine",
                    "common term",
                ],
                "exception_list": [],
                "expected_output": ["covid 19 mrna vaccine"],
            },
            # Add more test conditions as needed
        ]

        for condition in test_conditions:
            terms = condition["terms"]
            exception_list = condition["exception_list"]
            expected_output = condition["expected_output"]

            result = self.cleaner.clean(terms, exception_list, True)
            print("Actual", result, "expected", expected_output)
            self.assertEqual(result, expected_output)

    def test_clean_entities(self):
        test_conditions = [
            {
                "input": "OPSUMIT ®",
                "expected": "opsumit",
            },
            {
                "input": "OPSUMITA®",
                "expected": "opsumita",
            },
            {
                "input": "OPSUMITB®, other product",
                "expected": "opsumitb, other product",
            },
            {
                "input": "/OPSUMITC ®",
                "expected": "opsumitc",
            },
            {
                "input": "5-ht1a inhibitors",
                "expected": "5 ht1a inhibitor",  # TODO
            },
            {
                "input": "1-(3-aminophenyl)-6,8-dimethyl-5-(4-iodo-2-fluoro-phenylamino)-3-cyclopropyl-1h,6h-pyrido[4,3-d]pyridine-2,4,7-trione derivatives",
                "expected": "1-(3-aminophenyl)-6,8-dimethyl-5-(4-iodo-2-fluoro-phenylamino)-3-cyclopropyl-1h,6h-pyrido[4,3-d]pyridine-2,4,7-trione derivative",
            },
            {
                "input": "(meth)acrylic acid polymer",
                "expected": "methacrylic acid polymer",
            },
            {
                "input": "metabotropic glutamate receptor (mGluR) antagonists",
                "expected": "metabotropic glutamate antagonist",
            },
            {
                "input": "poly(isoprene)",
                "expected": "polyisoprene",
            },
            {
                "input": "poly(isoprene-co-butadiene)",
                "expected": "polyisoprene co-butadiene",
            },
            {
                "input": "The γc-family Interleukin-2 (IL-2), Interleukin-9 (IL-9), and Interleukin-15 (IL-15)",
                "expected": "the γc family il2, il9, and il15",
            },
            {
                "input": "Familial Alzheimer Disease (FAD)",
                "expected": "familial alzheimer disease",
            },
            {
                "input": "il36 pathway inhibitors",
                "expected": "il36 inhibitor",
            },
            {
                "input": "anti-il36r antibodies",
                "expected": "anti il36r antibody",
            },
            {
                "input": "human il-36r agonist ligands il36α",
                "expected": "human il36r agonist il36α",  # TODO
            },
            {
                "input": "GLP-1 receptor agonists",
                "expected": "glp1 agonist",
            },
            {
                "input": "tgf-beta 1 accessory receptor",
                "expected": "tgf β 1 accessory receptor",
            },
            {
                "input": "angiotensin-ii",
                "expected": "angiotensin ii",
            },
            {
                "input": "receptor activator of NF-kB ligand",
                "expected": "nfkb activator",
            },
            {
                "input": "chimeric antibody-T cell receptor",
                "expected": "chimeric antibody t-cell receptor",
            },
            {
                "input": "β1-adrenoreceptor gene",
                "expected": "β1 adrenoreceptor gene",
            },
            {
                "input": "5-hydroxytryptamine-3 receptor antagonist",
                "expected": "5 hydroxytryptamine 3 antagonist",
            },
            {
                "input": "glucagon-like peptide-2 receptors",
                "expected": "glucagon like peptide 2 receptor",
            },
            {
                "input": "T-cell receptor CD-28",
                "expected": "t-cell receptor cd28",
            },
            {
                "input": "abl kinase inhibition",
                "expected": "abl kinase inhibitor",
            },
            {
                "input": "akt (protein kinase b) inhibitor",
                "expected": "akt (kinase b) inhibitor",  # TODO
            },
        ]

        for condition in test_conditions:
            input = condition["input"]
            expected = condition["expected"]

            result = self.cleaner.clean([input], [], True)
            print("Actual", result, "expected", [expected])
            self.assertEqual(result, [expected])

    def test_rearrange_of(self):
        test_conditions = [
            {
                "input": "related diseases of abnormal pulmonary function",
                "expected": "abnormal pulmonary function related diseases",
            },
            {
                "input": "suturing lacerations of the meniscus",
                "expected": "meniscus suturing lacerations",
            },
            {"input": "rejection of a transplant", "expected": "transplant rejection"},
            {
                "input": "diseases treatable by inhibition of FGFR",
                "expected": "diseases treatable by FGFR inhibition",
            },
            {
                "input": "disorders of cell proliferation",
                "expected": "cell proliferation disorders",
            },
            {
                "input": "diseases associated with expression of GU Protein",
                "expected": "GU Protein expression diseases",
            },
            {
                "input": "conditions characterized by up-regulation of IL-10",
                "expected": "conditions characterized by il-10 up-regulation",
            },
            {"input": "alleviation of tumors", "expected": "tumor alleviation"},
            {
                "input": "diseases of the respiratory tract",
                "expected": "respiratory tract diseases",
            },
            {
                "input": "diseases mediated by modulation of voltage-gated sodium channels",
                "expected": "diseases mediated by voltage-gated sodium channel modulation",
            },
            {
                "input": "conditions associated with production of IL-1 and IL-6",
                "expected": "IL-1 and IL-6 production conditions",
            },
            {
                "input": "inhibitors of phosphatidylinositol 3-kinase gamma",
                "expected": "phosphatidylinositol 3-kinase gamma inhibitors",
            },
            {
                "input": "inhibitors of the interaction between mdm2 and XYZ",
                "expected": "interaction inhibitors",  # TODO
            },
            {
                "input": "middle-of-the night insomnia",
                "expected": "-the night insomnia middle-",  # TODO eek should be "middle of the night insomnia",
            },
            {
                "input": "disorders mediated by neurofibrillary tangles",
                "expected": "disorders mediated by neurofibrillary tangles",  # ok but ideally 'neurofibrillary tangle mediated disorders'
            },
            {
                "input": "inhibitors for use in the treatment of blood-borne cancers",
                "expected": "inhibitors for use in blood-borne cancer the treatment",  # TODO: "blood-borne cancer treatment inhibitors",
            },
            {
                "input": "antibody against urokinase receptor",
                "expected": "urokinase receptor antibody",
            },
        ]

        for condition in test_conditions:
            input = condition["input"]
            expected = condition["expected"]

            result = list(rearrange_terms([input]))[0]
            if result != expected:
                print(f"Actual: '{result}', expected: '{expected}'")

            self.assertEqual(result, expected)

    def test_pos_dash(self):
        test_conditions = [
            {
                "input": "APoE-4",
                "expected": "APoE4",
            },
            {
                "input": "HIV-1 infection",
                "expected": "HIV1 infection",
            },
            {
                "input": "sodium channel-mediated diseases",
                "expected": "sodium channel mediated diseases",
            },
            {
                "input": "neuronal hypo-kinetic disease",
                "expected": "neuronal hypo kinetic disease",  # TODO
            },
            {
                "input": "Loeys-Dietz syndrome",
                "expected": "Loeys Dietz syndrome",
            },
            {
                "input": "sleep-wake cycles",
                "expected": "sleep wake cycles",
            },
            {
                "input": "low-grade prostate cancer",
                "expected": "low grade prostate cancer",
            },
            {
                "input": "non-insulin dependent diabetes mellitus",
                "expected": "non insulin dependent diabetes mellitus",
            },
            {
                "input": "T-cell lymphoblastic leukemia",
                "expected": "T cell lymphoblastic leukemia",
            },
            {
                "input": "T-cell",
                "expected": "T cell",
            },
            {
                "input": "MAGE-A3 gene",
                "expected": "MAGEA3 gene",
            },
            {
                "input": "Bcr-Abl kinase",
                "expected": "Bcr Abl kinase",
            },
            {
                "input": "HLA-C gene",
                "expected": "HLAC gene",
            },
            {
                "input": "IL-6",
                "expected": "IL6",
            },
            {
                "input": "interleukin-6",
                "expected": "interleukin 6",
            },
            {
                "input": "Alzheimer's disease",
                "expected": "Alzheimer disease",
            },
            {
                "input": "graft-versus-host-disease",
                "expected": "graft versus host disease",
            },
            {
                "input": "graft -versus - host - disease (gvhd)",  # IRL example
                "expected": "graft  versus host disease (gvhd)",
            },
            {
                "input": "(-)-ditoluoyltartaric acid",
                "expected": "(-)-ditoluoyltartaric acid",
            },
            {
                "input": "(+)-hydrocodone",
                "expected": "(+)-hydrocodone",
            },
            {
                "input": "(6e)-8-hydroxygeraniol",
                "expected": "(6e)-8 hydroxygeraniol",
            },
            {
                "input": "(6R,S)-5-formyltetrahydrofolate",
                "expected": "(6R,S)-5 formyltetrahydrofolate",  # TODO
            },
            {
                "input": "mild-to-moderate alzheimer's disease",
                "expected": "mild to moderate alzheimer disease",
            },
            {
                "input": "Anti-cd73, anti-pd-l1 antibodies and chemotherapy for treating tumors",
                "expected": "Anti cd73, anti pdl1 antibodies and chemotherapy for treating tumors",
            },
        ]

        for condition in test_conditions:
            input = condition["input"]
            expected = condition["expected"]

            result = list(normalize_by_pos([input]))[0]
            if result != expected:
                print(f"Actual: '{result}', expected: '{expected}'")

            self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
