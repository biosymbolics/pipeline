"""
NER util tests
"""
import unittest

from common.ner.cleaning import normalize_entity_names


# CC-90010
# CK1α Degrader
# BMS-986158
# CD3xPSCA Bispecific
# Anti-CCR8
# Anti-Fucosyl GM1
# PKCθ Inhibitor
# cendakimab
# CD33 NKE
# BCMA NKE --Relapsed/Refractory Multiple Myeloma BET Inhibitor
# eIF2b Activator: Evotec


class TestNerUtils(unittest.TestCase):
    def test_normalize_entity_name(self):
        """
        Tests entity name normalization
        """
        test_cases = [
            {
                "entity_str": "Lorbrena/Lorviqua (lorlatinib)",
                "expected": "Lorbrena/lorviqua",
            },
            {
                "entity_str": "Myfembree(relugolix fixed dose combination)(a)",
                "expected": "Myfembree",
            },
            {
                "entity_str": "Paxlovid(e) (nirmatrelvir [PF-07321332]; ritonavir)",
                "expected": "Paxlovid",
            },
            {
                "entity_str": "Cibinqo(abrocitinib)",
                "expected": "Cibinqo",
            },
            {
                "entity_str": "Nyvepria(pegfilgrastim-apgf)",
                "expected": "Nyvepria",
            },
            # {
            #     "entity_str": "Braftovi (encorafenib) and Mektovi (binimetinib)(c) ",
            #     "expected": "Braftovi and Mektovi",
            # },
            # Comirnaty/BNT162b2(PF-07302048)(a)
            # Immunization to prevent COVID-19 (booster)
            # zavegepant(intranasal)
            # Endometriosis (combination with estradiol and norethindrone acetate)
            # Prevnar 20(Vaccine)(h)
        ]

        for test in test_cases:
            result = normalize_entity_names([test["entity_str"]])
            self.assertEqual(result, [test["expected"]])
