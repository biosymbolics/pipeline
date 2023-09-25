from data.prediction.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_OPTIMIZER_CLASS,
    DEFAULT_SAVE_FREQUENCY,
    DEFAULT_TRUE_THRESHOLD,
)

CHECKPOINT_PATH = "clindev_model_checkpoints"


BATCH_SIZE = DEFAULT_BATCH_SIZE
DEVICE = "mps"
EMBEDDING_DIM = 16
LR = 1e-6
MAX_ITEMS_PER_CAT = 5  # DEFAULT_MAX_ITEMS_PER_CAT
OPTIMIZER_CLASS = DEFAULT_OPTIMIZER_CLASS
SAVE_FREQUENCY = DEFAULT_SAVE_FREQUENCY
TRUE_THRESHOLD = DEFAULT_TRUE_THRESHOLD

SINGLE_SELECT_CATEGORICAL_FIELDS: list[str] = [
    "phase",
    "sponsor_type",
    # "comparator",
    # "facilities", ??
    # "countries" ??
]
Y1_CATEGORICAL_FIELDS: list[str] = [
    "design",
    "masking",
    "randomization",
]
MULTI_SELECT_CATEGORICAL_FIELDS: list[str] = [
    "conditions",
    "interventions",
    # "termination_reason",
]
TEXT_FIELDS: list[str] = []
QUANTITATIVE_FIELDS: list[str] = [
    "enrollment",
    "start_date",
]  # enrollment + duration have very low correlation, high covariance
Y2_FIELD = "duration"


# with start date
# INFO:root:Stage1 design Metrics: {'precision': 0.3601883086205383, 'recall': 0.28252967520760774, 'f1-score': 0.25608948897399253}
# INFO:root:Stage1 design Accuracy: 0.5561899038461539
# INFO:root:Stage1 masking Metrics: {'precision': 0.27554986048907154, 'recall': 0.26145247795739185, 'f1-score': 0.2324802204045405}
# INFO:root:Stage1 masking Accuracy: 0.5302483974358975
# INFO:root:Stage1 randomization Metrics: {'precision': 0.44463306265185776, 'recall': 0.28662344452298383, 'f1-score': 0.27285090366887976}
# INFO:root:Stage1 randomization Accuracy: 0.7072315705128205
# INFO:root:Stage2 MAE: 290.42245092147436

# (only 2000c)
# INFO:root:Stage1 design Metrics: {'precision': 0.41458479521617725, 'recall': 0.29075154497309463, 'f1-score': 0.27267735527793896}
# INFO:root:Stage1 design Accuracy: 0.5693359375
# INFO:root:Stage1 masking Metrics: {'precision': 0.3358596375427413, 'recall': 0.2867235204906746, 'f1-score': 0.2654897214192193}
# INFO:root:Stage1 masking Accuracy: 0.51416015625
# INFO:root:Stage1 randomization Metrics: {'precision': 0.54058229352347, 'recall': 0.3827675189722388, 'f1-score': 0.36977309664495445}
# INFO:root:Stage1 randomization Accuracy: 0.744140625
# INFO:root:Stage2 MAE: 162.5795135498047

# learnable embeddings, sized
# INFO:root:Stage1 design Metrics: {'precision': 0.2875904265960235, 'recall': 0.2652472435978736, 'f1-score': 0.22339165077887083}
# INFO:root:Stage1 design Accuracy: 0.310546875
# INFO:root:Stage1 masking Metrics: {'precision': 0.31644165358451076, 'recall': 0.2776932722945374, 'f1-score': 0.20352623877540613}
# INFO:root:Stage1 masking Accuracy: 0.31787109375
# INFO:root:Stage1 randomization Metrics: {'precision': 0.5785, 'recall': 0.37414965986394555, 'f1-score': 0.3552587237980494}
# INFO:root:Stage1 randomization Accuracy: 0.74169921875
# INFO:root:Stage2 MAE: 155.61978149414062

# higher weight decay
# INFO:root:Stage1 design Metrics: {'precision': 0.3044067955148667, 'recall': 0.211263168118451, 'f1-score': 0.16326977421929}
# INFO:root:Stage1 design Accuracy: 0.52294921875
# INFO:root:Stage1 masking Metrics: {'precision': 0.2604082484174232, 'recall': 0.2567331770846363, 'f1-score': 0.19891699273139474}
# INFO:root:Stage1 masking Accuracy: 0.2978515625
# INFO:root:Stage1 randomization Metrics: {'precision': 0.4572053831715986, 'recall': 0.3592052870167855, 'f1-score': 0.3332139953930379}
# INFO:root:Stage1 randomization Accuracy: 0.7275390625
# INFO:root:Stage2 MAE: 156.21859741210938

# 10,000
# INFO:root:Stage1 design Metrics: {'precision': 0.18245140297881934, 'recall': 0.18110410312022598, 'f1-score': 0.157358222229351}
# INFO:root:Stage1 design Accuracy: 0.5765224358974359
# INFO:root:Stage1 masking Metrics: {'precision': 0.1314500884096779, 'recall': 0.17943782271280673, 'f1-score': 0.13533546050584264}
# INFO:root:Stage1 masking Accuracy: 0.4795673076923077
# INFO:root:Stage1 randomization Metrics: {'precision': 0.40118861024033436, 'recall': 0.27401801140094983, 'f1-score': 0.24998637689635714}
# INFO:root:Stage1 randomization Accuracy: 0.7006209935897436
# INFO:root:Stage2 MAE: 351.2637970753205
