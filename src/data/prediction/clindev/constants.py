from data.prediction.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_OPTIMIZER_CLASS,
    DEFAULT_SAVE_FREQUENCY,
    DEFAULT_TRUE_THRESHOLD,
)

CHECKPOINT_PATH = "clindev_model_checkpoints"


BATCH_SIZE = 32  # DEFAULT_BATCH_SIZE
DEVICE = "mps"
EMBEDDING_DIM = 16
LR = 1e-4
MAX_ITEMS_PER_CAT = 10  # DEFAULT_MAX_ITEMS_PER_CAT
OPTIMIZER_CLASS = DEFAULT_OPTIMIZER_CLASS
SAVE_FREQUENCY = DEFAULT_SAVE_FREQUENCY
TRUE_THRESHOLD = DEFAULT_TRUE_THRESHOLD

SINGLE_SELECT_CATEGORICAL_FIELDS: list[str] = [
    "phase",
    "sponsor_type",
    "enrollment",
    # "facilities", ??
    # "countries" ??
]
Y1_CATEGORICAL_FIELDS: list[str] = [
    "design",
    "masking",
    "randomization",
    "comparison_type",
    # "hypothesis_type"
    # "termination_reason",
    # dropout_count
]
MULTI_SELECT_CATEGORICAL_FIELDS: list[str] = [
    "conditions",  # typically only the specific condition
    "mesh_conditions",  # normalized; includes ancestors
    "interventions",
]
TEXT_FIELDS: list[str] = []


QUANTITATIVE_FIELDS: list[str] = [
    "start_date",
]

# Side note: enrollment + duration have very low correlation, high covariance
# (low corr is perhaps why it doesn't offer much loss reduction)
QUANTITATIVE_TO_CATEGORY_FIELDS: list[str] = [
    "enrollment",
    "duration",
    # dropout_count
]
Y2_FIELD = "duration"  # TODO: include distance measure


# new data (mesh conditions, comparator type) - BIG improvement (stage2 CEL was 0.32 before)
# INFO:__main__:Stage1 design: {"0": {"precision": 0.9843843843843844, "recall": 0.9669616519174041, "f1-score": 0.9755952380952375}, "1": {"precision": 0.9891186071817193, "recall": 0.9926677067082683, "f1-score": 0.9908899789768739}, "2": {"precision": 0.9829857299670691, "recall": 0.9867768595041322, "f1-score": 0.9848776464118773}, "macro avg": {"precision": 0.9854962405110577, "recall": 0.9821354060432682, "f1-score": 0.9837876211613296}}
# INFO:__main__:Stage1 design: 0.9871975806451613
# INFO:__main__:Stage1 masking: {"0": {"precision": 0.9020141535111595, "recall": 0.9495702005730658, "f1-score": 0.9251814628699044}, "1": {"precision": 0.985723698660224, "recall": 0.9885462555066079, "f1-score": 0.987132959419333}, "2": {"precision": 0.9076595744680851, "recall": 0.9543624161073826, "f1-score": 0.9304252998909481}, "3": {"precision": 0.888268156424581, "recall": 0.6694736842105263, "f1-score": 0.7635054021608638}, "4": {"precision": 0.8600973236009732, "recall": 0.7643243243243243, "f1-score": 0.8093875214653685}, "macro avg": {"precision": 0.9087525813330046, "recall": 0.8652553761443814, "f1-score": 0.8831265291612835}}
# INFO:__main__:Stage1 masking: 0.9378024193548387
# INFO:__main__:Stage1 randomization: {"0": {"precision": 0.950228832951945, "recall": 0.9916417910447761, "f1-score": 0.970493718959976}, "1": {"precision": 0.9488721804511279, "recall": 0.8887323943661972, "f1-score": 0.9178181818181812}, "2": {"precision": 0.9930731317437059, "recall": 0.9893828798938288, "f1-score": 0.9912245712006378}, "macro avg": {"precision": 0.9640580483822596, "recall": 0.9565856884349341, "f1-score": 0.9598454906595983}}
# INFO:__main__:Stage1 randomization: 0.9825604838709677
# INFO:__main__:Stage1 comparison_type: {"0": {"precision": 0.9757113115891742, "recall": 0.9959858323494687, "f1-score": 0.985744332788034}, "1": {"precision": 0.9722991689750693, "recall": 0.9915254237288136, "f1-score": 0.9818181818181814}, "2": {"precision": 0.7435897435897436, "recall": 0.15675675675675677, "f1-score": 0.2589285714285712}, "3": {"precision": 0.9090909090909091, "recall": 0.16666666666666666, "f1-score": 0.2816901408450701}, "4": {"precision": 0.9743452699091395, "recall": 0.9961748633879781, "f1-score": 0.9851391515806532}, "5": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "macro avg": {"precision": 0.7625060671923393, "recall": 0.5511849238149473, "f1-score": 0.5822200630767517}}
# INFO:__main__:Stage1 comparison_type: 0.9735887096774194
# INFO:__main__:Stage2: {"0": {"precision": 0.6451388888888889, "recall": 0.78895966029724, "f1-score": 0.7098376313276021}, "1": {"precision": 0.486331569664903, "recall": 0.6027322404371585, "f1-score": 0.5383113714006827}, "2": {"precision": 0.5856443719412724, "recall": 0.4835016835016835, "f1-score": 0.5296938399114712}, "3": {"precision": 0.4344073647871116, "recall": 0.6371308016877637, "f1-score": 0.5165925419089971}, "4": {"precision": 0.5444664031620553, "recall": 0.58, "f1-score": 0.5616717635066253}, "5": {"precision": 0.5540201005025126, "recall": 0.5547169811320755, "f1-score": 0.5543683218101817}, "6": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "7": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "8": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "9": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}, "macro avg": {"precision": 0.32500086989467436, "recall": 0.3647041367055921, "f1-score": 0.341047546986556}}
# INFO:__main__:Stage2: 0.5469758064516129
# INFO:__main__:Saved checkpoint checkpoint_95.pt
