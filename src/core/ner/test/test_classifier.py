import unittest

from core.ner.classifier import classify_by_keywords, classify_string
from constants.patents import get_patent_attribute_map
from typings.trials import TERMINATION_KEYWORD_MAP, TerminationReason


PATENT_ATTRIBUTE_MAP = get_patent_attribute_map()


class TestClassifier(unittest.TestCase):
    def test_trial_termination_classification(self):
        test_conditions = [
            {
                "text": "This trial failed due to business reasons",
                "expected_output": [TerminationReason.BUSINESS],
            },
            {
                "text": "Insufficient Accrual",
                "expected_output": [TerminationReason.ENROLLMENT],
            },
        ]

        km = TERMINATION_KEYWORD_MAP
        for condition in test_conditions:
            text = condition["text"]
            expected_output = condition["expected_output"]

            result = classify_string(text, km)
            print("Actual", result, "expected", expected_output)
            self.assertEqual(result, expected_output)

    def test_classify_by_keywords(self):
        test_conditions = [
            {
                "docs": [
                    "Binding molecule specific to lrig-1 protein, and use thereof. The present invention relates to a binding molecule which can bind specifically to Lrig-1 protein which is a protein present on the surface of a regulator T cell. The binding molecule provided by the present invention can effectively prevent, alleviate or treat immunity-related diseases, such as autoimmune disease, graft versus host disease, organ transplant rejection, asthma, atopy, or acute or chronic inflammatory disease, which are diseases induced by the excessive activation and expression of diverse immune cells and inflammatory cells, by activating the function of regulator T cells. In addition, the binding molecule, preferably antibody, specific to Lrig-1 protein, according to the present invention, has an excellent binding force and can be more effectively targeted to Lrig-1 protein, compared with existing antibodies to Lrig-1 which have been commercially sold.",
                ],
                "attribute_map": PATENT_ATTRIBUTE_MAP,
                "expected_output": [["COMPOUND_OR_MECHANISM"]],
            },
            {
                "docs": [
                    "Food supplement for combating asthma made from medicinal plants. The food supplement for combating asthma contains certain nutrients such as vitamins, minerals and unsaturated fatty acids that combat infections and inflammations of the respiratory system, soothe and act favourably on dilation of the bronchi, and provide relief to persons suffering from breathing difficulties caused by congestion or lack of air."
                ],
                "attribute_map": PATENT_ATTRIBUTE_MAP,
                "expected_output": [["COMPOUND_OR_MECHANISM", "NUTRITIONAL"]],
            },
            {
                "docs": [
                    "Sensor devices and systems for monitoring markers in breath. The disclosure relates to devices, systems and methods for detecting markers in breath, more specifically volatile and non-volatile markers associated with pulmonary diseases such as, for example, asthma, chronic obstructive pulmonary disease (COPD), or cystic fibrosis (CF), in exhaled breath condensate (EBC)."
                ],
                "attribute_map": PATENT_ATTRIBUTE_MAP,
                "expected_output": [["DIAGNOSTIC", "PROCESS"]],  # "DEVICE"
            },
            {
                "docs": [
                    "In vitro method for identifying severity levels in patients with bronchial asthma. The present invention shows that miR-185-5p exhibits higher levels of relative expression (2 -ΔCt , where ΔCt= Ct miRNA of interest (miR-185-5p) -Ct endogenous miRNA) in asthma patients than in healthy individuals. Moreover, the levels of expression of miR-185-5p differ not only between healthy individuals and asthma patients, but also between healthy individuals and each of the subgroups of asthma patients (intermittent, persistent mild, moderate and severe)"
                ],
                "attribute_map": PATENT_ATTRIBUTE_MAP,
                "expected_output": [["COMPOUND_OR_MECHANISM", "DIAGNOSTIC"]],
            },
            {
                "docs": [
                    "Fc receptor-mediated drug delivery. Provided are methods and compositions for modulating an immune response or for treating a disease or condition in a subject, such as cancer, infection, autoimmune disease, allergy, and asthma."
                ],
                "attribute_map": PATENT_ATTRIBUTE_MAP,
                "expected_output": [
                    ["COMPOUND_OR_MECHANISM", "METHOD_OF_ADMINISTRATION"]
                ],
            },
        ]

        for condition in test_conditions:
            docs = condition["docs"]
            attribute_map = condition["attribute_map"]
            expected_output = condition["expected_output"]

            result = classify_by_keywords(docs, attribute_map)
            print("Actual", result, "expected", expected_output)
            self.assertEqual(result, expected_output)
