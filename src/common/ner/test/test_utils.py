import unittest

from common.ner.cleaning import EntityCleaner
from common.ner.utils import rearrange_terms, normalize_by_pos, lemmatize_tail


class TestNerUtils(unittest.TestCase):
    def setUp(self):
        self.clean = EntityCleaner()

    # def test_filter_common_terms(self):
    #     test_conditions = [
    #         {
    #             "terms": [
    #                 "vaccine candidates",
    #                 "PF-06863135",
    #                 "therapeutic options",
    #                 "COVID-19 mRNA vaccine",
    #                 "exception term",
    #                 "common term",
    #             ],
    #             "exception_list": ["exception"],
    #             "expected_output": [
    #                 "PF-06863135",
    #                 "COVID-19 mRNA vaccine",
    #                 "exception term",
    #             ],
    #         },
    #         {
    #             "terms": [
    #                 "vaccine candidate",
    #                 "vaccine candidates",
    #                 "therapeutic options",
    #                 "therapeutic option",
    #                 "therapeutics option",
    #                 "COVID-19 mRNA vaccine",
    #                 "common term",
    #             ],
    #             "exception_list": [],
    #             "expected_output": ["COVID-19 mRNA vaccine"],
    #         },
    #         # Add more test conditions as needed
    #     ]

    #     for condition in test_conditions:
    #         terms = condition["terms"]
    #         exception_list = condition["exception_list"]
    #         expected_output = condition["expected_output"]

    #         result = clean(terms, exception_list)
    #         print("Actual", result, "expected", expected_output)
    #         self.assertEqual(result, expected_output)

    # def test_clean_entities(self):
    #     test_conditions = [
    #         {
    #             "input": "OPSUMIT ®",
    #             "expected": "opsumit",
    #         },
    #         {
    #             "input": "OPSUMIT®",
    #             "expected": "opsumit",
    #         },
    #         {
    #             "input": "OPSUMIT®, other product",
    #             "expected": "opsumit, other product",
    #         },
    #         {
    #             "input": "/OPSUMIT ®",
    #             "expected": "OPSUMIT",
    #         },
    #         {
    #             "input": "5-ht1a inhibitors",
    #             "expected": "5-ht1a inhibitor",
    #         },
    #         {
    #             "input": "1-(3-aminophenyl)-6,8-dimethyl-5-(4-iodo-2-fluoro-phenylamino)-3-cyclopropyl-1h,6h-pyrido[4,3-d]pyridine-2,4,7-trione derivatives",
    #             "expected": "1-(3-aminophenyl)-6,8-dimethyl-5-(4-iodo-2-fluoro-phenylamino)-3-cyclopropyl-1h,6h-pyrido[4,3-d]pyridine-2,4,7-trione derivative",
    #         },
    #     ]

    #     for condition in test_conditions:
    #         input = condition["input"]
    #         expected = condition["expected"]

    #         result = clean([input])
    #         print("Actual", result, "expected", [expected])
    #         self.assertEqual(result, [expected])

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
                "expected": "IL-1 and il-6 production conditions",
            },
            # {
            #     "input": "disorders mediated by neurofibrillary tangles",
            #     "expected": "neurofibrillary tangle mediated disorders",
            # }
            # {
            #     "input": "inhibitors for use in the treatment of blood-borne cancers",
            #     "expected": "blood-borne cancer treatment inhibitors",
            # }
        ]

        for condition in test_conditions:
            input = condition["input"]
            expected = condition["expected"]

            result = rearrange_terms(input)
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
                "expected": "graft  versus host disease (gvhd)",  # dup space removed elsewhere
            },
        ]

        for condition in test_conditions:
            input = condition["input"]
            expected = condition["expected"]

            result = normalize_by_pos(input)
            if result != expected:
                print(f"Actual: '{result}', expected: '{expected}'")

            self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
